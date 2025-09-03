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

import copy
import logging
import multiprocessing
import random
import string
import time
import uuid
from typing import Optional, Union, Iterator
import re

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types
from kubeflow.trainer.utils import utils
from kubeflow_trainer_api import models
from kubernetes import client, config, watch
from kubeflow.trainer.backends.base import ExecutionBackend
from kubeflow.trainer.backends.kubernetes import types as k8s_types

logger = logging.getLogger(__name__)


class KubernetesBackend(ExecutionBackend):
    def __init__(
        self,
        cfg: k8s_types.KubernetesBackendConfig,
    ):
        if cfg.namespace is None:
            cfg.namespace = utils.get_default_target_namespace(cfg.context)

        # If client configuration is not set, use kube-config to access Kubernetes APIs.
        if cfg.client_configuration is None:
            # Load kube-config or in-cluster config.
            if cfg.config_file or not utils.is_running_in_k8s():
                config.load_kube_config(config_file=cfg.config_file, context=cfg.context)
            else:
                config.load_incluster_config()

        k8s_client = client.ApiClient(cfg.client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)

        self.namespace = cfg.namespace

    def list_runtimes(self) -> list[types.Runtime]:
        result = []
        try:
            thread = self.custom_api.list_cluster_custom_object(
                constants.GROUP,
                constants.VERSION,
                constants.CLUSTER_TRAINING_RUNTIME_PLURAL,
                async_req=True,
            )

            runtime_list = models.TrainerV1alpha1ClusterTrainingRuntimeList.from_dict(
                thread.get(constants.DEFAULT_TIMEOUT)
            )

            if not runtime_list:
                return result

            for runtime in runtime_list.items:
                if not (
                    runtime.metadata
                    and runtime.metadata.labels
                    and constants.RUNTIME_FRAMEWORK_LABEL in runtime.metadata.labels
                ):
                    logger.warning(
                        f"Runtime {runtime.metadata.name} must have "  # type: ignore
                        f"{constants.RUNTIME_FRAMEWORK_LABEL} label."
                    )
                    continue
                result.append(self.__get_runtime_from_crd(runtime))

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {constants.CLUSTER_TRAINING_RUNTIME_KIND}s "
                f"in namespace: {self.namespace}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {constants.CLUSTER_TRAINING_RUNTIME_KIND}s "
                f"in namespace: {self.namespace}"
            ) from e

        return result

    def get_runtime(self, name: str) -> types.Runtime:
        """Get the the Runtime object"""

        try:
            thread = self.custom_api.get_cluster_custom_object(
                constants.GROUP,
                constants.VERSION,
                constants.CLUSTER_TRAINING_RUNTIME_PLURAL,
                name,
                async_req=True,
            )

            runtime = models.TrainerV1alpha1ClusterTrainingRuntime.from_dict(
                thread.get(constants.DEFAULT_TIMEOUT)  # type: ignore
            )

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get {constants.CLUSTER_TRAINING_RUNTIME_PLURAL}: "
                f"{self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get {constants.CLUSTER_TRAINING_RUNTIME_PLURAL}: "
                f"{self.namespace}/{name}"
            ) from e

        return self.__get_runtime_from_crd(runtime)  # type: ignore

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
            import subprocess
            import shutil
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
        job_name = self.train(
            runtime=runtime_copy,
            trainer=types.CustomTrainer(
                func=print_packages,
                num_nodes=1,
                resources_per_node=({"cpu": 1} if runtime_copy.trainer.device != "gpu" else None),
            ),
        )

        self.wait_for_job_status(job_name)
        print("\n".join(self.get_job_logs(name=job_name)))
        self.delete_job(job_name)

    def train(
        self,
        runtime: Optional[types.Runtime] = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[Union[types.CustomTrainer, types.BuiltinTrainer]] = None,
    ) -> str:
        if runtime is None:
            runtime = self.get_runtime(constants.TORCH_RUNTIME)

        # Generate unique name for the TrainJob.
        # TODO (andreyvelich): Discuss this TrainJob name generation.
        train_job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]

        # Build the Trainer.
        trainer_crd = models.TrainerV1alpha1Trainer()

        if trainer:
            # If users choose to use a custom training function.
            if isinstance(trainer, types.CustomTrainer):
                if runtime.trainer.trainer_type != types.TrainerType.CUSTOM_TRAINER:
                    raise ValueError(f"CustomTrainer can't be used with {runtime} runtime")
                trainer_crd = utils.get_trainer_crd_from_custom_trainer(runtime, trainer)

            # If users choose to use a builtin trainer for post-training.
            elif isinstance(trainer, types.BuiltinTrainer):
                if runtime.trainer.trainer_type != types.TrainerType.BUILTIN_TRAINER:
                    raise ValueError(f"BuiltinTrainer can't be used with {runtime} runtime")
                trainer_crd = utils.get_trainer_crd_from_builtin_trainer(
                    runtime, trainer, initializer
                )

            else:
                raise ValueError(
                    f"The trainer type {type(trainer)} is not supported. "
                    "Please use CustomTrainer or BuiltinTrainer."
                )

        train_job = models.TrainerV1alpha1TrainJob(
            apiVersion=constants.API_VERSION,
            kind=constants.TRAINJOB_KIND,
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(name=train_job_name),
            spec=models.TrainerV1alpha1TrainJobSpec(
                runtimeRef=models.TrainerV1alpha1RuntimeRef(name=runtime.name),
                trainer=(trainer_crd if trainer_crd != models.TrainerV1alpha1Trainer() else None),
                initializer=(
                    models.TrainerV1alpha1Initializer(
                        dataset=utils.get_dataset_initializer(initializer.dataset),
                        model=utils.get_model_initializer(initializer.model),
                    )
                    if isinstance(initializer, types.Initializer)
                    else None
                ),
            ),
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

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> list[types.TrainJob]:
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
                thread.get(constants.DEFAULT_TIMEOUT)
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

                result.append(self.__get_trainjob_from_crd(trainjob))

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
                thread.get(constants.DEFAULT_TIMEOUT)  # type: ignore
            )

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            ) from e

        return self.__get_trainjob_from_crd(trainjob)  # type: ignore

    def get_job_logs(
        self,
        name: str,
        follow: Optional[bool] = False,
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
                for logline in log_stream:
                    yield logline  # type:ignore
            else:
                logs = self.core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.namespace,
                    container=container_name,
                )

                for line in logs.splitlines():
                    yield line

        except Exception as e:
            raise RuntimeError(
                f"Failed to read logs for the pod {self.namespace}/{pod_name}"
            ) from e

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJob:
        job_statuses = {
            constants.TRAINJOB_CREATED,
            constants.TRAINJOB_RUNNING,
            constants.TRAINJOB_COMPLETE,
            constants.TRAINJOB_FAILED,
        }
        if not status.issubset(job_statuses):
            raise ValueError(f"Expected status {status} must be a subset of {job_statuses}")

        if polling_interval > timeout:
            raise ValueError(
                f"Polling interval {polling_interval} must be less than timeout: {timeout}"
            )

        for _ in range(round(timeout / polling_interval)):
            # Check the status after event is generated for the TrainJob's Pods.
            trainjob = self.get_job(name)
            logger.debug(f"TrainJob {name}, status {trainjob.status}")

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

    def __get_runtime_from_crd(
        self,
        runtime_crd: models.TrainerV1alpha1ClusterTrainingRuntime,
    ) -> types.Runtime:
        if not (
            runtime_crd.metadata
            and runtime_crd.metadata.name
            and runtime_crd.spec
            and runtime_crd.spec.ml_policy
            and runtime_crd.spec.template.spec
            and runtime_crd.spec.template.spec.replicated_jobs
        ):
            raise Exception(f"ClusterTrainingRuntime CRD is invalid: {runtime_crd}")

        if not (
            runtime_crd.metadata.labels
            and constants.RUNTIME_FRAMEWORK_LABEL in runtime_crd.metadata.labels
        ):
            raise Exception(
                f"Runtime {runtime_crd.metadata.name} must have "
                f"{constants.RUNTIME_FRAMEWORK_LABEL} label"
            )

        return types.Runtime(
            name=runtime_crd.metadata.name,
            trainer=utils.get_runtime_trainer(
                runtime_crd.metadata.labels[constants.RUNTIME_FRAMEWORK_LABEL],
                runtime_crd.spec.template.spec.replicated_jobs,
                runtime_crd.spec.ml_policy,
            ),
        )

    def __get_trainjob_from_crd(
        self,
        trainjob_crd: models.TrainerV1alpha1TrainJob,
    ) -> types.TrainJob:
        if not (
            trainjob_crd.metadata
            and trainjob_crd.metadata.name
            and trainjob_crd.metadata.namespace
            and trainjob_crd.spec
            and trainjob_crd.metadata.creation_timestamp
        ):
            raise Exception(f"TrainJob CRD is invalid: {trainjob_crd}")

        name = trainjob_crd.metadata.name
        namespace = trainjob_crd.metadata.namespace

        runtime = self.get_runtime(trainjob_crd.spec.runtime_ref.name)

        # Construct the TrainJob from the CRD.
        trainjob = types.TrainJob(
            name=name,
            creation_timestamp=trainjob_crd.metadata.creation_timestamp,
            runtime=runtime,
            steps=[],
            # Number of nodes is taken from TrainJob or TrainingRuntime
            num_nodes=(
                trainjob_crd.spec.trainer.num_nodes
                if trainjob_crd.spec.trainer and trainjob_crd.spec.trainer.num_nodes
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
            ).get(constants.DEFAULT_TIMEOUT)

            # Convert Pod to the correct format.
            pod_list = models.IoK8sApiCoreV1PodList.from_dict(response.to_dict())
            if not pod_list:
                return trainjob

            for pod in pod_list.items:
                # Pod must have labels to detect the TrainJob step.
                # Every Pod always has a single TrainJob step.
                if not (pod.metadata and pod.metadata.name and pod.metadata.labels and pod.spec):
                    raise Exception(f"TrainJob Pod is invalid: {pod}")

                # Get the Initializer step.
                if pod.metadata.labels[constants.JOBSET_RJOB_NAME_LABEL] in {
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
                elif pod.metadata.labels[constants.JOBSET_RJOB_NAME_LABEL] in {
                    constants.LAUNCHER,
                    constants.NODE,
                }:
                    trainjob.steps.append(
                        utils.get_trainjob_node_step(
                            pod.metadata.name,
                            pod.spec,
                            pod.status,
                            trainjob.runtime,
                            pod.metadata.labels[constants.JOBSET_RJOB_NAME_LABEL],
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
        if trainjob_crd.status and trainjob_crd.status.conditions:
            for c in trainjob_crd.status.conditions:
                if c.type == constants.TRAINJOB_COMPLETE and c.status == "True":
                    trainjob.status = c.type
                elif c.type == constants.TRAINJOB_FAILED and c.status == "True":
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
