# Copyright 2026 The Kubeflow Authors.
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

"""Trainer-native backend for the Kubeflow Optimizer.

This backend creates OptimizationJob resources from the `trainer.kubeflow.org/v1alpha1`
API group introduced by KEP-2605 in kubeflow/trainer, instead of Katib Experiment
resources. Hyperparameter suggestions are injected into trials by the OptimizationJob
controller via `KUBEFLOW_TRAINER_OPT_<NAME>` environment variables, so the trial
template is submitted without any parameter placeholder substitution.
"""

from collections.abc import Callable, Iterator
import copy
import datetime
import logging
import multiprocessing
import random
import string
import time
from typing import Any
import uuid

from kubernetes import client, config

import kubeflow.common.constants as common_constants
import kubeflow.common.utils as common_utils
from kubeflow.optimizer.backends.base import RuntimeBackend
from kubeflow.optimizer.backends.trainer_native import utils
from kubeflow.optimizer.backends.trainer_native.types import TrainerNativeBackendConfig
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import BaseAlgorithm, RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
    Result,
    Trial,
    TrialConfig,
)
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend as TrainerBackend
import kubeflow.trainer.constants.constants as trainer_constants
from kubeflow.trainer.types.types import Event, TrainJobTemplate

logger = logging.getLogger(__name__)


class TrainerNativeBackend(RuntimeBackend):
    """Backend that manages trainer-native OptimizationJob resources."""

    def __init__(self, cfg: TrainerNativeBackendConfig):
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
        self.trainer_backend = TrainerBackend(cfg)

    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        search_space: dict[str, Any],
        trial_config: TrialConfig | None = None,
        objectives: list[Objective] | None = None,
        algorithm: BaseAlgorithm | None = None,
    ) -> str:
        # Generate unique name for the OptimizationJob.
        optimization_job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]

        # Validate search_space
        if not search_space:
            raise ValueError("Search space must be set.")

        # Set defaults.
        objectives = objectives or [Objective()]
        algorithm = algorithm or RandomSearch()
        trial_config = trial_config or TrialConfig()

        if trial_config.max_failed_trials is not None:
            raise ValueError(
                "max_failed_trials is not supported by the trainer-native OptimizationJob"
            )

        trial_template = copy.deepcopy(trial_template)

        # The OptimizationJob controller injects hyperparameter suggestions into every
        # trial via KUBEFLOW_TRAINER_OPT_<NAME> environment variables, so the trial
        # template is used as-is without parameter placeholder substitution.
        optimization_job = {
            "apiVersion": trainer_constants.API_VERSION,
            "kind": constants.OPTIMIZATION_JOB_KIND,
            "metadata": {"name": optimization_job_name},
            "spec": {
                "objectives": utils.get_crd_objectives(objectives),
                "searchAlgorithm": utils.get_crd_search_algorithm(algorithm),
                "parameters": utils.get_crd_parameters(search_space),
                "numTrials": trial_config.num_trials,
                "parallelTrials": trial_config.parallel_trials,
                "trainJobTemplate": {
                    "spec": self.trainer_backend._get_trainjob_spec(
                        runtime=trial_template.runtime,
                        trainer=trial_template.trainer,
                        initializer=trial_template.initializer,
                    ).to_dict(),
                },
            },
        }

        # Create the OptimizationJob.
        try:
            self.custom_api.create_namespaced_custom_object(
                trainer_constants.GROUP,
                trainer_constants.VERSION,
                self.namespace,
                constants.OPTIMIZATION_JOB_PLURAL,
                optimization_job,
            )
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to create {constants.OPTIMIZATION_JOB_KIND}: "
                f"{self.namespace}/{optimization_job_name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {constants.OPTIMIZATION_JOB_KIND}: "
                f"{self.namespace}/{optimization_job_name}"
            ) from e

        logger.debug(
            f"{constants.OPTIMIZATION_JOB_KIND} {self.namespace}/{optimization_job_name} "
            "has been created"
        )

        return optimization_job_name

    def list_jobs(self) -> list[OptimizationJob]:
        """List of the created OptimizationJobs"""
        result = []

        try:
            thread = self.custom_api.list_namespaced_custom_object(
                trainer_constants.GROUP,
                trainer_constants.VERSION,
                self.namespace,
                constants.OPTIMIZATION_JOB_PLURAL,
                async_req=True,
            )

            optimization_job_list = thread.get(common_constants.DEFAULT_TIMEOUT)

            for optimization_job in optimization_job_list.get("items", []):
                result.append(self.__get_optimization_job_from_cr(optimization_job))

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {constants.OPTIMIZATION_JOB_KIND}s in namespace: {self.namespace}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {constants.OPTIMIZATION_JOB_KIND}s in namespace: {self.namespace}"
            ) from e

        return result

    def get_job(self, name: str) -> OptimizationJob:
        """Get the OptimizationJob object"""
        optimization_job = self.__get_optimization_job_cr(name)
        return self.__get_optimization_job_from_cr(optimization_job)

    def get_job_logs(
        self,
        name: str,
        trial_name: str | None = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get the OptimizationJob logs from a Trial"""
        # Determine what trial to get logs from.
        if trial_name is None:
            # Get logs from the best current trial.
            best_trial = self._get_best_trial(name)
            if best_trial is None:
                return
            trial_name = best_trial.name
            logger.debug(f"Getting logs from trial: {trial_name}")

        # Trials are plain TrainJobs, so delegate to the trainer backend.
        yield from self.trainer_backend.get_job_logs(name=trial_name, follow=follow)

    def get_best_results(self, name: str) -> Result | None:
        """Get the best hyperparameters and metrics from an OptimizationJob"""
        best_trial = self._get_best_trial(name)

        if best_trial is None:
            return None

        return Result(
            parameters=best_trial.parameters,
            metrics=best_trial.metrics,
        )

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.OPTIMIZATION_JOB_COMPLETE},
        timeout: int = 3600,
        polling_interval: int = 2,
        callbacks: list[Callable[[OptimizationJob], None]] | None = None,
    ) -> OptimizationJob:
        job_statuses = {
            constants.OPTIMIZATION_JOB_CREATED,
            constants.OPTIMIZATION_JOB_RUNNING,
            constants.OPTIMIZATION_JOB_COMPLETE,
            constants.OPTIMIZATION_JOB_FAILED,
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
            optimization_job = self.get_job(name)
            logger.debug(
                f"{constants.OPTIMIZATION_JOB_KIND} {name}, status {optimization_job.status}"
            )

            # Invoke callbacks if provided
            if callbacks:
                for callback in callbacks:
                    callback(optimization_job)

            if (
                constants.OPTIMIZATION_JOB_FAILED not in status
                and optimization_job.status == constants.OPTIMIZATION_JOB_FAILED
            ):
                raise RuntimeError(f"{constants.OPTIMIZATION_JOB_KIND} {name} is Failed")

            if optimization_job.status in status:
                return optimization_job

            time.sleep(polling_interval)

        raise TimeoutError(
            f"Timeout waiting for {constants.OPTIMIZATION_JOB_KIND} {name} to reach status: "
            f"{status}"
        )

    def delete_job(self, name: str):
        """Delete the OptimizationJob"""

        try:
            self.custom_api.delete_namespaced_custom_object(
                trainer_constants.GROUP,
                trainer_constants.VERSION,
                self.namespace,
                constants.OPTIMIZATION_JOB_PLURAL,
                name=name,
            )
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to delete {constants.OPTIMIZATION_JOB_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {constants.OPTIMIZATION_JOB_KIND}: {self.namespace}/{name}"
            ) from e

        logger.debug(f"{constants.OPTIMIZATION_JOB_KIND} {self.namespace}/{name} has been deleted")

    def get_job_events(self, name: str) -> list[Event]:
        # Get the OptimizationJob to ensure it exists.
        self.get_job(name)

        # Create set of all OptimizationJob-related resource names. Individual trial
        # names are not tracked in the OptimizationJob status yet, so only the best
        # trial's TrainJob is included.
        # TODO(kubeflow/trainer#3856): include all trials once the trial history is
        # stored natively in the OptimizationJob status.
        optimization_job_resources = {name}
        best_trial = self._get_best_trial(name)
        if best_trial is not None:
            optimization_job_resources.add(best_trial.name)

        events = []
        try:
            # Retrieve events from the namespace
            event_response = self.core_api.list_namespaced_event(
                namespace=self.namespace,
                async_req=True,
            ).get(common_constants.DEFAULT_TIMEOUT)

            # Filter events related to OptimizationJob resources
            for event in event_response.items:
                if not (event.metadata and event.involved_object and event.first_timestamp):
                    continue

                involved_object = event.involved_object

                # Check if event is related to OptimizationJob resources
                if (
                    involved_object.kind
                    in {constants.OPTIMIZATION_JOB_KIND, trainer_constants.TRAINJOB_KIND}
                    and involved_object.name in optimization_job_resources
                ):
                    events.append(
                        Event(
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
                f"Timeout getting {constants.OPTIMIZATION_JOB_KIND} events: {self.namespace}/{name}"
            ) from e

    def _get_best_trial(self, name: str) -> Trial | None:
        """Get the best current Trial for the OptimizationJob"""
        optimization_job = self.__get_optimization_job_cr(name)

        # The best trial is tracked in status.result of the OptimizationJob.
        result = (optimization_job.get("status") or {}).get("result") or {}
        trainjob_name = result.get("trainJobName")
        if not trainjob_name:
            return None

        parameters = {
            pa["name"]: pa["value"]
            for pa in result.get("parameters", [])
            if pa.get("name") is not None and pa.get("value") is not None
        }

        trainjob = self.trainer_backend.get_job(name=trainjob_name)

        # TODO(kubeflow/trainer#3856): populate metrics once the trial history is
        # stored natively in the OptimizationJob status.
        return Trial(
            name=trainjob_name,
            parameters=parameters,
            trainjob=trainjob,
        )

    def __get_optimization_job_cr(self, name: str) -> dict[str, Any]:
        """Get the OptimizationJob CR from Kubernetes API"""
        try:
            thread = self.custom_api.get_namespaced_custom_object(
                trainer_constants.GROUP,
                trainer_constants.VERSION,
                self.namespace,
                constants.OPTIMIZATION_JOB_PLURAL,
                name,
                async_req=True,
            )

            optimization_job = thread.get(common_constants.DEFAULT_TIMEOUT)

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get {constants.OPTIMIZATION_JOB_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get {constants.OPTIMIZATION_JOB_KIND}: {self.namespace}/{name}"
            ) from e

        return optimization_job

    def __get_optimization_job_from_cr(
        self,
        optimization_job_cr: dict[str, Any],
    ) -> OptimizationJob:
        metadata = optimization_job_cr.get("metadata") or {}
        spec = optimization_job_cr.get("spec") or {}

        if not (
            metadata.get("name")
            and metadata.get("creationTimestamp")
            and spec.get("parameters")
            and spec.get("objectives")
        ):
            raise ValueError(
                f"{constants.OPTIMIZATION_JOB_KIND} CR is invalid: {optimization_job_cr}"
            )

        optimization_job = OptimizationJob(
            name=metadata["name"],
            search_space=utils.get_search_space_from_crd(spec["parameters"]),
            objectives=utils.get_objectives_from_crd(spec["objectives"]),
            algorithm=utils.get_algorithm_from_crd(spec.get("searchAlgorithm") or {}),
            trial_config=TrialConfig(
                num_trials=spec.get("numTrials", 1),
                parallel_trials=spec.get("parallelTrials", 1),
                max_failed_trials=None,
            ),
            # TODO(kubeflow/trainer#3856): populate trials once the trial history is
            # stored natively in the OptimizationJob status.
            trials=[],
            creation_timestamp=datetime.datetime.fromisoformat(
                metadata["creationTimestamp"].replace("Z", "+00:00")
            ),
            status=constants.OPTIMIZATION_JOB_CREATED,  # The default OptimizationJob status.
        )

        # Update the OptimizationJob status from the CR conditions.
        conditions = (optimization_job_cr.get("status") or {}).get("conditions") or []
        for c in conditions:
            if c.get("status") != "True":
                continue
            if c.get("type") == constants.OPTIMIZATION_JOB_COMPLETE:
                optimization_job.status = constants.OPTIMIZATION_JOB_COMPLETE
            elif c.get("type") == constants.OPTIMIZATION_JOB_FAILED:
                optimization_job.status = constants.OPTIMIZATION_JOB_FAILED

        return optimization_job
