# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Iterator
import copy
import logging
import multiprocessing
import os
import random
import re
import string
import time
from typing import Any
import uuid

from kubeflow_trainer_api import models
from kubernetes import client, config, watch
from kubernetes.stream import stream

import kubeflow.common.constants as common_constants
from kubeflow.common.types import KubernetesBackendConfig
import kubeflow.common.utils as common_utils
from kubeflow.trainer.backends.base import RuntimeBackend
import kubeflow.trainer.backends.kubernetes.utils as utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class KubernetesBackend(RuntimeBackend):
    def __init__(self, cfg: KubernetesBackendConfig):
        if cfg.namespace is None:
            cfg.namespace = common_utils.get_default_target_namespace(cfg.context)

        # If client configuration is not set, use kube-config to access Kubernetes APIs.
        if cfg.client_configuration is None:
            # Load kube-config or in-cluster config.
            if cfg.config_file or not common_utils.is_running_in_k8s():
                config.load_kube_config(config_file=cfg.config_file, context=cfg.context)
            else:
                config.load_incluster_config()

        k8s_client = client.ApiClient(cfg.client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)

        self.namespace = cfg.namespace

        # Perform control-plane version metadata verification.
        self.verify_backend()

    def verify_backend(self) -> None:
        """Verify that the Trainer control plane exposes version metadata.

        This check only ensures that the public control-plane ConfigMap exists
        and contains a ``kubeflow_trainer_version`` field. It does not
        enforce version compatibility and never raises.
        """

        system_namespace = os.getenv("KUBEFLOW_SYSTEM_NAMESPACE", "kubeflow-system")
        config_map_name = "kubeflow-trainer-public"

        try:
            _ = self.core_api.read_namespaced_config_map(
                name=config_map_name,
                namespace=system_namespace,
            ).data["kubeflow_trainer_version"]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Trainer control-plane version info is not available: "
                f"unable to read 'kubeflow_trainer_version' from ConfigMap "
                f"'{config_map_name}' in namespace '{system_namespace}': {e}. "
                "If your Trainer control-plane is installed in a different namespace, "
                "set the KUBEFLOW_SYSTEM_NAMESPACE environment variable accordingly."
            )
            return

    def list_runtimes(self) -> list[types.Runtime]:
        """List available runtimes, preferring namespaced over cluster-scoped for duplicates.

        If a TrainingRuntime with the same name exists in both the namespace and cluster scope,
        only the namespaced runtime is returned. Cluster-scoped runtimes are still returned
        when there is no namespaced runtime with the same name.
        """
        result: list[types.Runtime] = []

        cluster_thread = self.custom_api.list_cluster_custom_object(
            constants.GROUP,
            constants.VERSION,
            constants.CLUSTER_TRAINING_RUNTIME_PLURAL,
            async_req=True,
        )

        namespace_thread = self.custom_api.list_namespaced_custom_object(
            constants.GROUP,
            constants.VERSION,
            self.namespace,
            constants.TRAINING_RUNTIME_PLURAL,
            async_req=True,
        )

        # Fetch namespace-scoped TrainingRuntimes.
        namespace_runtimes = None
        try:
            namespace_runtimes = models.TrainerV1alpha1TrainingRuntimeList.from_dict(
                namespace_thread.get(common_constants.DEFAULT_TIMEOUT)
            )
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(f"Timeout to list {types.RuntimeKind.TRAINING_RUNTIME}s") from e
        except client.ApiException as e:
            if e.status != 404:
                raise RuntimeError(f"Failed to list {types.RuntimeKind.TRAINING_RUNTIME}s") from e
        except Exception as e:
            raise RuntimeError(f"Failed to list {types.RuntimeKind.TRAINING_RUNTIME}s") from e

        # Fetch cluster-scoped ClusterTrainingRuntimes.
        cluster_runtimes = None
        try:
            cluster_runtimes = models.TrainerV1alpha1ClusterTrainingRuntimeList.from_dict(
                cluster_thread.get(common_constants.DEFAULT_TIMEOUT)
            )
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {types.RuntimeKind.CLUSTER_TRAINING_RUNTIME}s"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {types.RuntimeKind.CLUSTER_TRAINING_RUNTIME}s"
            ) from e

        # Collect runtimes in a map, preferring namespaced over cluster-scoped
        runtimes_by_name = {}

        # Add namespaced runtimes first (they have priority)
        if namespace_runtimes:
            for runtime in namespace_runtimes.items:
                if runtime.metadata and runtime.metadata.name:
                    runtimes_by_name[runtime.metadata.name] = runtime

        # Add cluster runtimes only if not already present
        if cluster_runtimes:
            for runtime in cluster_runtimes.items:
                if runtime.metadata and runtime.metadata.name:
                    runtimes_by_name.setdefault(
                        runtime.metadata.name,
                        runtime,
                    )

        try:
            for runtime in runtimes_by_name.values():
                if not (
                    runtime.metadata
                    and runtime.metadata.labels
                    and constants.RUNTIME_FRAMEWORK_LABEL in runtime.metadata.labels
                ):
                    logger.warning(
                        "Runtime %s missing %s label",
                        runtime.metadata.name if runtime.metadata else "<unknown>",
                        constants.RUNTIME_FRAMEWORK_LABEL,
                    )
                    continue

                result.append(self.__get_runtime_from_cr(runtime))
        except Exception:
            logger.exception(
                "Failed to parse runtime %s",
                runtime.metadata.name if runtime.metadata else "<unknown>",
            )
            raise
        return result

    def get_runtime(self, name: str) -> types.Runtime:
        """Prefer namespaced runtime, fall back to cluster-scoped only if it does not exist"""
        try:
            ns_thread = self.custom_api.get_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINING_RUNTIME_PLURAL,
                name,
                async_req=True,
            )
            runtime = models.TrainerV1alpha1TrainingRuntime.from_dict(
                ns_thread.get(common_constants.DEFAULT_TIMEOUT)
            )
            return self.__get_runtime_from_cr(runtime)

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get {types.RuntimeKind.TRAINING_RUNTIME}: {self.namespace}/{name}"
            ) from e
        except client.ApiException as e:
            if e.status != 404:
                raise RuntimeError(
                    f"Failed to get {types.RuntimeKind.TRAINING_RUNTIME}: {self.namespace}/{name}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get {types.RuntimeKind.TRAINING_RUNTIME}: {self.namespace}/{name}"
            ) from e

        try:
            cluster_thread = self.custom_api.get_cluster_custom_object(
                constants.GROUP,
                constants.VERSION,
                constants.CLUSTER_TRAINING_RUNTIME_PLURAL,
                name,
                async_req=True,
            )
            runtime = models.TrainerV1alpha1ClusterTrainingRuntime.from_dict(
                cluster_thread.get(common_constants.DEFAULT_TIMEOUT)
            )
            return self.__get_runtime_from_cr(runtime)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get {types.RuntimeKind.CLUSTER_TRAINING_RUNTIME}: {name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get Runtime: {name} (checked both namespaced and cluster scope)"
            ) from e

    def get_runtime_packages(self, runtime: types.Runtime):
        if runtime.trainer.trainer_type == types.TrainerType.BUILTIN_TRAINER:
            raise ValueError("Cannot get Runtime packages for BuiltinTrainer")

        # Create a deepcopy of the runtime to avoid modifying the original command.
        runtime_copy = copy.deepcopy(runtime)

        # Run mpirun only within the single process.
        if runtime_copy.trainer.command[0] == "mpirun":
            mpi_command = list(constants.MPI_COMMAND)
            mpi_command[1:3] = ["-np", "1"]
            runtime_copy.trainer.set_command(tuple(mpi_command))

        def print_packages():
            import shutil
            import subprocess
            import sys

            # Print Python version.
            print(f"Python: {sys.version}")

            # Print Python packages.
            if shutil.which("pip"):
                pip_list = subprocess.run(["pip", "list"], capture_output=True, text=True)
                print(pip_list.stdout)
            else:
                print("Unable to get installed packages: pip command not found")

            # Print nvidia-smi if GPUs are available.
            if shutil.which("nvidia-smi"):
                print("Available GPUs on the single training node")
                nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                print(nvidia_smi.stdout)

        # Create the TrainJob and wait until it completes.
        # If Runtime trainer has GPU resources use them, otherwise run TrainJob with 1 CPU.
        job_name = None
        try:
            job_name = self.train(
                runtime=runtime_copy,
                trainer=types.CustomTrainer(
                    func=print_packages,
                    num_nodes=1,
                    resources_per_node=(
                        {"cpu": 1} if runtime_copy.trainer.device != "gpu" else None
                    ),
                ),
            )
            self.wait_for_job_status(job_name)
            print("\n".join(self.get_job_logs(name=job_name)))
        finally:
            if job_name is not None:
                try:
                    self.delete_job(job_name)
                except Exception:
                    logger.exception(
                        "Failed to delete temporary TrainJob %s/%s created by get_runtime_packages",
                        self.namespace,
                        job_name,
                    )

    def train(
        self,
        runtime: str | types.Runtime | None = None,
        initializer: types.Initializer | None = None,
        trainer: types.CustomTrainer
        | types.CustomTrainerContainer
        | types.BuiltinTrainer
        | None = None,
        options: list | None = None,
    ) -> str:
        # Process options to extract configuration
        job_spec = {}
        labels = None
        annotations = None
        name = None
        trainer_overrides = {}
        runtime_patches = None
        active_deadline_seconds = None

        if options:
            for option in options:
                option(job_spec, trainer, self)

            metadata_section = job_spec.get("metadata", {})
            labels = metadata_section.get("labels")
            annotations = metadata_section.get("annotations")
            name = metadata_section.get("name")

            # Extract spec-level configurations
            spec_section = job_spec.get("spec", {})
            trainer_overrides = spec_section.get("trainer", {})
            runtime_patches = spec_section.get("runtimePatches")
            active_deadline_seconds = spec_section.get("activeDeadlineSeconds")

        # Generate unique name for the TrainJob if not provided
        train_job_name = name or (
            random.choice(string.ascii_lowercase)
            + uuid.uuid4().hex[: constants.JOB_NAME_UUID_LENGTH]
        )

        # Build the TrainJob spec using the common _get_trainjob_spec method
        trainjob_spec = self._get_trainjob_spec(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            trainer_overrides=trainer_overrides,
            runtime_patches=runtime_patches,
            active_deadline_seconds=active_deadline_seconds,
        )

        # Build the TrainJob.
        train_job = models.TrainerV1alpha1TrainJob(
            apiVersion=constants.API_VERSION,
            kind=constants.TRAINJOB_KIND,
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name=train_job_name, labels=labels, annotations=annotations
            ),
            spec=trainjob_spec,
        )

        # Create the TrainJob.
        try:
            self.custom_api.create_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                train_job.to_dict(),
            )
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to create {constants.TRAINJOB_KIND}: {self.namespace}/{train_job_name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {constants.TRAINJOB_KIND}: {self.namespace}/{train_job_name}"
            ) from e

        logger.debug(
            f"{constants.TRAINJOB_KIND} {self.namespace}/{train_job_name} has been created"
        )

        return train_job_name

    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        result = []
        try:
            thread = self.custom_api.list_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                async_req=True,
            )

            trainjob_list = models.TrainerV1alpha1TrainJobList.from_dict(
                thread.get(common_constants.DEFAULT_TIMEOUT)
            )

            if not trainjob_list:
                return result

            for trainjob in trainjob_list.items:
                # If runtime object is set, we check the TrainJob's runtime reference.
                if (
                    runtime is not None
                    and trainjob.spec
                    and trainjob.spec.runtime_ref
                    and trainjob.spec.runtime_ref.name != runtime.name
                ):
                    continue

                result.append(self.__get_trainjob_from_cr(trainjob))

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {constants.TRAINJOB_KIND}s in namespace: {self.namespace}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {constants.TRAINJOB_KIND}s in namespace: {self.namespace}"
            ) from e

        return result

    def get_job(self, name: str) -> types.TrainJob:
        """Get the TrainJob object"""

        try:
            thread = self.custom_api.get_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                name,
                async_req=True,
            )

            trainjob = models.TrainerV1alpha1TrainJob.from_dict(
                thread.get(common_constants.DEFAULT_TIMEOUT)  # type: ignore
            )

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            ) from e

        return self.__get_trainjob_from_cr(trainjob)  # type: ignore

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        """Get the TrainJob logs"""
        # Get the TrainJob Pod name.
        pod_name = None
        for c in self.get_job(name).steps:
            if c.status != constants.POD_PENDING and c.name == step:
                pod_name = c.pod_name
                break
        if pod_name is None:
            return

        # Remove the number for the node step.
        container_name = re.sub(r"-\d+$", "", step)
        yield from self._read_pod_logs(
            pod_name=pod_name, container_name=container_name, follow=follow
        )

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: list[Callable[[types.TrainJob], None]] | None = None,
    ) -> types.TrainJob:
        job_statuses = {
            constants.TRAINJOB_CREATED,
            constants.TRAINJOB_RUNNING,
            constants.TRAINJOB_COMPLETE,
            constants.TRAINJOB_FAILED,
        }
        if not status.issubset(job_statuses):
            raise ValueError(f"Expected status {status} must be a subset of {job_statuses}")

        if polling_interval <= 0:
            raise ValueError(
                f"Polling interval must be a positive number, got polling_interval={polling_interval}"
            )
        if polling_interval >= timeout:
            raise ValueError(
                f"Polling interval must be strictly less than timeout. "
                f"Received polling_interval={polling_interval}, timeout={timeout}"
            )

        for _ in range(round(timeout / polling_interval)):
            # Check the status after event is generated for the TrainJob's Pods.
            trainjob = self.get_job(name)
            logger.debug(f"TrainJob {name}, status {trainjob.status}")

            # Invoke callbacks if provided
            if callbacks:
                for callback in callbacks:
                    callback(trainjob)

            # Raise an error if TrainJob is Failed and it is not the expected status.
            if (
                constants.TRAINJOB_FAILED not in status
                and trainjob.status == constants.TRAINJOB_FAILED
            ):
                raise RuntimeError(f"TrainJob {name} is Failed")

            # Return the TrainJob if it reaches the expected status.
            if trainjob.status in status:
                return trainjob

            time.sleep(polling_interval)

        raise TimeoutError(f"Timeout waiting for TrainJob {name} to reach status: {status} status")

    def delete_job(self, name: str):
        try:
            self.custom_api.delete_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                name=name,
            )
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to delete {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            ) from e

        logger.debug(f"{constants.TRAINJOB_KIND} {self.namespace}/{name} has been deleted")

    def connect(
        self,
        runtime: str | types.Runtime | None = None,
        initializer: types.Initializer | None = None,
        trainer: types.CustomTrainerContainer | None = None,
        options: list | None = None,
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJobSession:
        """Create an interactive TrainJob and connect a session to it.

        The TrainJob keeps running, so multiple commands can be executed in the returned
        session. If the options don't include a `TrainerCommand`, the launcher Pod is kept
        alive with the default `sleep infinity` command.

        Args:
            runtime: reference to one of the existing runtimes. It can accept the runtime
                name or Runtime object. Defaults to the torch-distributed runtime if not provided.
            initializer: Optional configuration for the dataset and model initializers.
            trainer: Optional configuration for a CustomTrainerContainer. CustomTrainer and
                BuiltinTrainer are not supported, since their training process exits when the
                training finishes.
            options: Optional list of options to configure the TrainJob, e.g. a
                `TrainerCommand` to override the default keep-alive command.
            timeout: The maximum time to wait for the interactive TrainJob to be ready in seconds.
            polling_interval: The interval to poll the TrainJob's Pods in seconds.

        Returns:
            `TrainJobSession`: The session to execute commands in the TrainJob.

        Raises:
            ValueError: If the trainer is not a CustomTrainerContainer or the timeout values
                are invalid.
            TimeoutError: If the interactive TrainJob is not ready within the timeout.
            RuntimeError: If the interactive TrainJob can't be created or its Pod has already
                finished.
        """
        if trainer is not None and not isinstance(trainer, types.CustomTrainerContainer):
            raise ValueError(
                "Trainer must be a CustomTrainerContainer to create an interactive TrainJob. "
                "CustomTrainer and BuiltinTrainer are not supported, since their training "
                "process exits when the training finishes."
            )

        common_utils.validate_wait_for_job_status(polling_interval, timeout)

        # Import the option here to avoid circular imports at module load time.
        from kubeflow.trainer.options.kubernetes import TrainerCommand

        options = list(options) if options else []
        if not any(isinstance(option, TrainerCommand) for option in options):
            logger.debug(
                f"Setting the default command for the interactive {constants.TRAINJOB_KIND}: "
                f"{' '.join(constants.INTERACTIVE_COMMAND)}"
            )
            options.append(TrainerCommand(list(constants.INTERACTIVE_COMMAND)))

        name = self.train(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            options=options,
        )

        logger.debug(
            f"Waiting for the interactive {constants.TRAINJOB_KIND} "
            f"{self.namespace}/{name} to be ready"
        )

        last_pod_phase = None
        for _ in range(round(timeout / polling_interval)):
            pod = self._get_trainer_pod(name)
            if pod is not None:
                last_pod_phase = pod.status.phase if pod.status else None
                if last_pod_phase == constants.POD_RUNNING:
                    return types.TrainJobSession(name=name, backend=self)
                if last_pod_phase in {constants.POD_SUCCEEDED, constants.POD_FAILED}:
                    raise RuntimeError(
                        f"TrainJob's Pod {pod.metadata.name} reached the {last_pod_phase} phase "
                        "before an interactive session could be established. Make sure the "
                        "trainer Pod keeps running, e.g. by omitting the TrainerCommand option "
                        "to use the default "
                        f"'{' '.join(constants.INTERACTIVE_COMMAND)}' command."
                    )
            time.sleep(polling_interval)

        raise TimeoutError(
            f"Timeout waiting for the interactive {constants.TRAINJOB_KIND} "
            f"{self.namespace}/{name} to be ready"
            + (f" (last Pod phase: {last_pod_phase})" if last_pod_phase else "")
            + f". The {constants.TRAINJOB_KIND} was created and can be deleted with "
            f"delete_job('{name}')"
        )

    def exec_command(
        self,
        name: str,
        command: str | list[str],
        container: str | None = None,
    ) -> types.ExecResult:
        """Execute a command in the TrainJob's trainer Pod.

        Args:
            name: The name of the TrainJob to execute the command in.
            command: The command to execute. A string command is executed with `sh -c`.
                A list of arguments is executed directly.
            container: The name of the container the command is executed in. Defaults to the
                TrainJob's trainer container.

        Returns:
            `ExecResult`: The standard output, standard error and exit code of the command.

        Raises:
            ValueError: If the command is empty.
            RuntimeError: If the TrainJob's Pod can't be found, is not running, or the command
                can't be executed.
        """
        command = ["sh", "-c", command] if isinstance(command, str) else list(command)

        if not command:
            raise ValueError("Command must not be empty")

        pod = self._get_trainer_pod(name)
        if pod is None:
            raise RuntimeError(
                f"Failed to find the trainer Pod for {constants.TRAINJOB_KIND}: "
                f"{self.namespace}/{name}"
            )

        pod_phase = pod.status.phase if pod.status else None
        if pod_phase != constants.POD_RUNNING:
            raise RuntimeError(
                f"{constants.TRAINJOB_KIND}'s Pod {pod.metadata.name} is not Running, "
                f"its phase is {pod_phase}"
            )

        container_name = container or self._get_exec_container_name(pod)

        try:
            response = stream(
                self.core_api.connect_get_namespaced_pod_exec,
                pod.metadata.name,
                self.namespace,
                container=container_name,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                # Return the WebSocket client to stream the output of the command.
                _preload_content=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to exec the command in {constants.TRAINJOB_KIND}'s Pod "
                f"{self.namespace}/{pod.metadata.name}"
            ) from e

        stdout: list[str] = []
        stderr: list[str] = []
        try:
            while response.is_open():
                response.update(timeout=1)
                if response.peek_stdout():
                    stdout.append(response.read_stdout())
                if response.peek_stderr():
                    stderr.append(response.read_stderr())

            try:
                returncode = response.returncode
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to get the command's return code: {e}")
                returncode = None
        finally:
            response.close()

        return types.ExecResult(
            stdout="".join(stdout),
            stderr="".join(stderr),
            returncode=returncode,
        )

    def wait_for_job_deletion(
        self,
        name: str,
        timeout: int = 180,
        polling_interval: int = 2,
    ) -> None:
        """Wait for all Pods of the deleted TrainJob to be removed.

        Waiting for the Pods to be deleted ensures that a new TrainJob with the same name can
        be created right after. Otherwise, old Pods (e.g. with SSH keys) might conflict with
        the new TrainJob.

        Args:
            name: The name of the deleted TrainJob.
            timeout: The maximum time to wait for the Pods to be deleted in seconds.
            polling_interval: The interval to poll the Pods in seconds.

        Raises:
            ValueError: If the timeout values are invalid.
            RuntimeError: If the TrainJob's Pods can't be listed.
        """
        common_utils.validate_wait_for_job_status(polling_interval, timeout)

        for _ in range(round(timeout / polling_interval)):
            pods = self._list_job_pods(name)
            if not pods:
                logger.debug(
                    f"All Pods of {constants.TRAINJOB_KIND} {self.namespace}/{name} "
                    "have been deleted"
                )
                return
            logger.debug(
                f"Waiting for {len(pods)} Pods of {constants.TRAINJOB_KIND} "
                f"{self.namespace}/{name} to be deleted"
            )
            time.sleep(polling_interval)

        logger.warning(
            f"Timeout waiting for all Pods of {constants.TRAINJOB_KIND} "
            f"{self.namespace}/{name} to be deleted after {timeout} seconds"
        )

    def get_job_events(self, name: str) -> list[types.Event]:
        # Get all pod names related to this TrainJob
        trainjob = self.get_job(name)

        # Create set of all TrainJob-related resource names
        trainjob_resources = {name}
        for step in trainjob.steps:
            trainjob_resources.add(step.name)
            if step.pod_name:
                trainjob_resources.add(step.pod_name)

        events = []
        try:
            # Retrieve events from the namespace
            event_response: models.IoK8sApiCoreV1EventList = self.core_api.list_namespaced_event(
                namespace=self.namespace,
                async_req=True,
            ).get(common_constants.DEFAULT_TIMEOUT)

            # Filter events related to this TrainJob or its pods
            for event in event_response.items:
                if not (event.metadata and event.involved_object and event.first_timestamp):
                    continue

                involved_object = event.involved_object

                # Check if event is related to TrainJob resources
                if (
                    involved_object.kind in {constants.TRAINJOB_KIND, "JobSet", "Job", "Pod"}
                    and involved_object.name in trainjob_resources
                ):
                    events.append(
                        types.Event(
                            involved_object_kind=involved_object.kind,
                            involved_object_name=involved_object.name,
                            message=event.message or "",
                            reason=event.reason or "",
                            event_time=event.first_timestamp,
                        )
                    )

            # Sort events by first occurrence time
            events.sort(key=lambda e: e.event_time)

            return events
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout getting {constants.TRAINJOB_KIND} events: {self.namespace}/{name}"
            ) from e

    def __get_runtime_from_cr(
        self,
        runtime_cr: models.TrainerV1alpha1ClusterTrainingRuntime
        | models.TrainerV1alpha1TrainingRuntime,
    ) -> types.Runtime:
        if not (
            runtime_cr.metadata
            and runtime_cr.metadata.name
            and runtime_cr.spec
            and runtime_cr.spec.ml_policy
            and runtime_cr.spec.template.spec
            and runtime_cr.spec.template.spec.replicated_jobs
        ):
            raise Exception(
                f"{runtime_cr} is invalid — missing one or more required fields: "
                f"metadata.name, spec.ml_policy, spec.template.spec.replicated_jobs."
            )

        if not (
            runtime_cr.metadata.labels
            and constants.RUNTIME_FRAMEWORK_LABEL in runtime_cr.metadata.labels
        ):
            raise Exception(
                f"Runtime {runtime_cr.metadata.name} must have "
                f"{constants.RUNTIME_FRAMEWORK_LABEL} label"
            )

        return types.Runtime(
            name=runtime_cr.metadata.name,
            kind=types.RuntimeKind(runtime_cr.kind),
            trainer=utils.get_runtime_trainer(
                runtime_cr.metadata.labels[constants.RUNTIME_FRAMEWORK_LABEL],
                runtime_cr.spec.template.spec.replicated_jobs,
                runtime_cr.spec.ml_policy,
            ),
        )

    def _read_pod_logs(self, pod_name: str, container_name: str, follow: bool) -> Iterator[str]:
        """Read logs from a pod container."""
        try:
            if follow:
                log_stream = watch.Watch().stream(
                    self.core_api.read_namespaced_pod_log,
                    name=pod_name,
                    namespace=self.namespace,
                    container=container_name,
                    follow=True,
                )

                # Stream logs incrementally.
                yield from log_stream  # type: ignore
            else:
                logs = self.core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.namespace,
                    container=container_name,
                )

                yield from logs.splitlines()

        except Exception as e:
            raise RuntimeError(
                f"Failed to read logs for the pod {self.namespace}/{pod_name}"
            ) from e

    def _list_job_pods(self, name: str) -> list:
        """List the Pods created for the TrainJob."""
        try:
            response = self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=constants.POD_LABEL_SELECTOR.format(trainjob_name=name),
                async_req=True,
            ).get(common_constants.DEFAULT_TIMEOUT)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {constants.TRAINJOB_KIND}'s Pods: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {constants.TRAINJOB_KIND}'s Pods: {self.namespace}/{name}"
            ) from e

        return list(response.items) if response and response.items else []

    def _get_trainer_pod(self, name: str):
        """Get the Pod that runs the trainer command for the TrainJob.

        The trainer Pod is marked with the `trainer.kubeflow.org/trainjob-ancestor-step`
        label. It is the launcher Pod for the MPI runtimes and the node Pod for the other
        runtimes. The node Pods of the MPI runtimes only run the SSH server, so commands
        must never be executed in them.
        """
        pods = self._list_job_pods(name)

        trainer_pods = [
            pod
            for pod in pods
            if pod.metadata
            and pod.metadata.labels
            and pod.metadata.labels.get(constants.TRAINJOB_ANCESTOR_LABEL) == constants.TRAINER
        ]
        if not trainer_pods:
            # Fall back to the launcher Pod for the runtimes that don't set the ancestor label.
            trainer_pods = [
                pod
                for pod in pods
                if pod.metadata
                and pod.metadata.labels
                and pod.metadata.labels.get(constants.JOBSET_RJOB_NAME_LABEL) == constants.LAUNCHER
            ]

        if not trainer_pods:
            return None

        # If the Pod was restarted, use the newest one.
        return max(
            trainer_pods,
            key=lambda pod: (
                pod.metadata is not None and pod.metadata.creation_timestamp is not None,
                pod.metadata.creation_timestamp if pod.metadata else None,
            ),
        )

    def _get_exec_container_name(self, pod) -> str:
        """Get the name of the container commands are executed in."""
        containers = pod.spec.containers if pod.spec and pod.spec.containers else []
        if not containers:
            raise RuntimeError(
                f"{constants.TRAINJOB_KIND}'s Pod {pod.metadata.name} has no containers"
            )

        # The trainer container is always named 'node' in the TrainJob runtimes.
        for container in containers:
            if container.name == constants.NODE:
                return container.name

        return containers[0].name

    def __get_trainjob_from_cr(
        self,
        trainjob_cr: models.TrainerV1alpha1TrainJob,
    ) -> types.TrainJob:
        if not (
            trainjob_cr.metadata
            and trainjob_cr.metadata.name
            and trainjob_cr.metadata.namespace
            and trainjob_cr.spec
            and trainjob_cr.metadata.creation_timestamp
        ):
            raise Exception(f"TrainJob CR is invalid: {trainjob_cr}")

        name = trainjob_cr.metadata.name
        namespace = trainjob_cr.metadata.namespace

        runtime = self.get_runtime(trainjob_cr.spec.runtime_ref.name)

        # Construct the TrainJob from the CR.
        trainjob = types.TrainJob(
            name=name,
            creation_timestamp=trainjob_cr.metadata.creation_timestamp,
            runtime=runtime,
            steps=[],
            # Number of nodes is taken from TrainJob or TrainingRuntime
            num_nodes=(
                trainjob_cr.spec.trainer.num_nodes
                if trainjob_cr.spec.trainer and trainjob_cr.spec.trainer.num_nodes
                else runtime.trainer.num_nodes
            ),
            status=constants.TRAINJOB_CREATED,  # The default TrainJob status.
        )

        # Add the TrainJob components, e.g. trainer nodes and initializer.
        try:
            response = self.core_api.list_namespaced_pod(
                namespace,
                label_selector=constants.POD_LABEL_SELECTOR.format(trainjob_name=name),
                async_req=True,
            ).get(common_constants.DEFAULT_TIMEOUT)

            # Convert Pod to the correct format.
            # This is required to convert Pod's container resources into API object from str
            pod_list = models.IoK8sApiCoreV1PodList.from_dict(response.to_dict())
            if not pod_list:
                return trainjob

            sorted_pods = sorted(
                pod_list.items,
                key=lambda pod: (
                    pod.metadata is not None and pod.metadata.creation_timestamp is not None,
                    pod.metadata.creation_timestamp if pod.metadata else None,
                ),
                reverse=True,
            )
            seen_step_keys: set[str] = set()
            for pod in sorted_pods:
                # Pod must have labels to detect the TrainJob step.
                # Every Pod always has a single TrainJob step.
                if not (pod.metadata and pod.metadata.name and pod.metadata.labels and pod.spec):
                    raise Exception(f"TrainJob Pod is invalid: {pod}")

                role = pod.metadata.labels[constants.JOBSET_RJOB_NAME_LABEL]
                step_key = role
                if role in {constants.LAUNCHER, constants.NODE}:
                    step_key = f"{role}-{pod.metadata.labels[constants.JOB_INDEX_LABEL]}"

                if step_key in seen_step_keys:
                    continue
                seen_step_keys.add(step_key)

                # Get the Initializer step.
                if role in {
                    constants.DATASET_INITIALIZER,
                    constants.MODEL_INITIALIZER,
                }:
                    trainjob.steps.append(
                        utils.get_trainjob_initializer_step(
                            pod.metadata.name,
                            pod.spec,
                            pod.status,
                        )
                    )
                # Get the Node step.
                elif role in {
                    constants.LAUNCHER,
                    constants.NODE,
                }:
                    trainjob.steps.append(
                        utils.get_trainjob_node_step(
                            pod.metadata.name,
                            pod.spec,
                            pod.status,
                            trainjob.runtime,
                            role,
                            int(pod.metadata.labels[constants.JOB_INDEX_LABEL]),
                        )
                    )
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {constants.TRAINJOB_KIND}'s steps: {namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {constants.TRAINJOB_KIND}'s steps: {namespace}/{name}"
            ) from e

        # Update the TrainJob status from its conditions.
        if trainjob_cr.status and trainjob_cr.status.conditions:
            for c in trainjob_cr.status.conditions:
                if (
                    c.type == constants.TRAINJOB_COMPLETE
                    and c.status == "True"
                    or c.type == constants.TRAINJOB_FAILED
                    and c.status == "True"
                ):
                    trainjob.status = c.type
        else:
            # The TrainJob running status is defined when all training node (e.g. Pods) are
            # running or succeeded.
            num_running_nodes = sum(
                1
                for step in trainjob.steps
                if step.name.startswith(constants.NODE)
                and (
                    step.status == constants.TRAINJOB_RUNNING
                    or step.status == constants.POD_SUCCEEDED
                )
            )

            if trainjob.num_nodes == num_running_nodes:
                trainjob.status = constants.TRAINJOB_RUNNING

        return trainjob

    def _get_trainjob_spec(
        self,
        runtime: str | types.Runtime | None = None,
        initializer: types.Initializer | None = None,
        trainer: types.CustomTrainer
        | types.CustomTrainerContainer
        | types.BuiltinTrainer
        | None = None,
        trainer_overrides: dict[str, Any] | None = None,
        runtime_patches: list[dict[str, Any]] | None = None,
        active_deadline_seconds: int | None = None,
    ) -> models.TrainerV1alpha1TrainJobSpec:
        """Get TrainJob spec from the given parameters."""

        if runtime is None:
            runtime = self.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)
        elif isinstance(runtime, str):
            runtime = self.get_runtime(runtime)

        # Build the Trainer.
        trainer_cr = models.TrainerV1alpha1Trainer()

        if trainer:
            # If users choose to use a custom training script.
            if isinstance(trainer, (types.CustomTrainer, types.CustomTrainerContainer)):
                if runtime.trainer.trainer_type != types.TrainerType.CUSTOM_TRAINER:
                    raise ValueError(f"CustomTrainer can't be used with {runtime} runtime")
                trainer_cr = utils.get_trainer_cr_from_custom_trainer(runtime, trainer)

            # If users choose to use a builtin trainer for post-training.
            elif isinstance(trainer, types.BuiltinTrainer):
                if runtime.trainer.trainer_type != types.TrainerType.BUILTIN_TRAINER:
                    raise ValueError(f"BuiltinTrainer can't be used with {runtime} runtime")
                trainer_cr = utils.get_trainer_cr_from_builtin_trainer(
                    runtime, trainer, initializer
                )

            else:
                raise ValueError(
                    f"The trainer type {type(trainer)} is not supported. "
                    "Please use CustomTrainer, CustomTrainerContainer, or BuiltinTrainer."
                )

        # Apply trainer overrides if trainer was not provided but overrides exist
        if trainer_overrides:
            if "command" in trainer_overrides:
                trainer_cr.command = trainer_overrides["command"]
            if "args" in trainer_overrides:
                trainer_cr.args = trainer_overrides["args"]

        # Convert runtime patches dicts to native model objects.
        runtime_patch_models = None
        if runtime_patches:
            runtime_patch_models = [
                models.TrainerV1alpha1RuntimePatch.from_dict(p) for p in runtime_patches
            ]

        trainjob_spec = models.TrainerV1alpha1TrainJobSpec(
            runtimeRef=models.TrainerV1alpha1RuntimeRef(name=runtime.name, kind=runtime.kind.value),
            trainer=trainer_cr if trainer_cr != models.TrainerV1alpha1Trainer() else None,
            runtimePatches=runtime_patch_models,
            activeDeadlineSeconds=active_deadline_seconds,
        )

        # Add initializer if users define it.
        if initializer and (initializer.dataset or initializer.model):
            trainjob_spec.initializer = models.TrainerV1alpha1Initializer(
                dataset=utils.get_dataset_initializer(initializer.dataset)
                if initializer.dataset
                else None,
                model=utils.get_model_initializer(initializer.model) if initializer.model else None,
            )

        return trainjob_spec
