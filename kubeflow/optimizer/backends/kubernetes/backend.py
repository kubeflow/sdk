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

from collections.abc import Iterator
import logging
import multiprocessing
import random
import string
import time
from typing import Any, Optional
import uuid

from kubeflow_katib_api import models
from kubernetes import client, config

import kubeflow.common.constants as common_constants
from kubeflow.common.types import KubernetesBackendConfig
import kubeflow.common.utils as common_utils
from kubeflow.optimizer.backends.base import RuntimeBackend
from kubeflow.optimizer.backends.kubernetes import utils
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Metric,
    Objective,
    OptimizationJob,
    Result,
    Trial,
    TrialConfig,
)
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend as TrainerBackend
import kubeflow.trainer.constants.constants as trainer_constants
from kubeflow.trainer.types.types import TrainJobTemplate

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
        self.trainer_backend = TrainerBackend(cfg)

    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        search_space: dict[str, Any],
        trial_config: Optional[TrialConfig] = None,
        objectives: Optional[list[Objective]] = None,
        algorithm: Optional[RandomSearch] = None,
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

        # Iterate over search space to build the following values:
        # experiment.spec.parameters to define distribution and feasible space.
        # experiment.spec.trialTemplate.trialParameters to reference parameters in Trials.
        # Trainer function arguments for the appropriate substitution.
        parameters_spec = []
        trial_parameters = []
        if trial_template.trainer.func_args is None:
            trial_template.trainer.func_args = {}

        for param_name, param_spec in search_space.items():
            param_spec.name = param_name
            parameters_spec.append(param_spec)

            trial_parameters.append(
                models.V1beta1TrialParameterSpec(
                    name=param_name,
                    reference=param_name,
                )
            )

            trial_template.trainer.func_args[param_name] = f"${{trialParameters.{param_name}}}"

        # Build the Experiment.
        experiment = models.V1beta1Experiment(
            apiVersion=constants.API_VERSION,
            kind=constants.EXPERIMENT_KIND,
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(name=optimization_job_name),
            spec=models.V1beta1ExperimentSpec(
                # Trial template and parameters.
                trialTemplate=models.V1beta1TrialTemplate(
                    retain=True,
                    primaryContainerName=trainer_constants.NODE,
                    trialParameters=trial_parameters,
                    trialSpec={
                        "apiVersion": trainer_constants.API_VERSION,
                        "kind": trainer_constants.TRAINJOB_KIND,
                        "spec": self.trainer_backend._get_trainjob_spec(
                            runtime=trial_template.runtime,
                            trainer=trial_template.trainer,
                            initializer=trial_template.initializer,
                        ).to_dict(),
                    },
                ),
                parameters=parameters_spec,
                # Trial Configs.
                maxTrialCount=trial_config.num_trials,
                parallelTrialCount=trial_config.parallel_trials,
                maxFailedTrialCount=trial_config.max_failed_trials,
                # Objective specification.
                objective=models.V1beta1ObjectiveSpec(
                    objectiveMetricName=objectives[0].metric,
                    type=objectives[0].direction.value,
                    additionalMetricNames=[obj.metric for obj in objectives[1:]]
                    if len(objectives) > 1
                    else None,
                ),
                # Algorithm specification.
                algorithm=algorithm._to_katib_spec(),
            ),
        )

        # Create the Experiment.
        try:
            self.custom_api.create_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.EXPERIMENT_PLURAL,
                experiment.to_dict(),
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
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.EXPERIMENT_PLURAL,
                async_req=True,
            )

            optimization_job_list = models.V1beta1ExperimentList.from_dict(
                thread.get(common_constants.DEFAULT_TIMEOUT)
            )

            if not optimization_job_list:
                return result

            for optimization_job in optimization_job_list.items:
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
        optimization_job = self.__get_experiment_cr(name)
        return self.__get_optimization_job_from_cr(optimization_job)

    def get_job_logs(
        self,
        name: str,
        trial_name: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get the OptimizationJob logs from a Trial"""
        # Determine what trial to get logs from.
        if trial_name is None:
            # Get logs from the best current trial.
            best_trial = self._get_best_trial(name)
            if best_trial is None:
                # Get first trial if available.
                optimization_job = self.get_job(name)
                if not optimization_job.trials:
                    return
                trial_name = optimization_job.trials[0].name
            else:
                trial_name = best_trial.name
            logger.debug(f"Getting logs from trial: {trial_name}")

        # Get the Trial's Pod name.
        pod_name = None
        step = trainer_constants.NODE + "-0"
        for c in self.trainer_backend.get_job(trial_name).steps:
            if c.status != trainer_constants.POD_PENDING and c.name == step:
                pod_name = c.pod_name
                break
        if pod_name is None:
            return

        container_name = constants.METRICS_COLLECTOR_CONTAINER
        yield from self.trainer_backend._read_pod_logs(
            pod_name=pod_name, container_name=container_name, follow=follow
        )

    def get_best_results(self, name: str) -> Optional[Result]:
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
    ) -> OptimizationJob:
        job_statuses = {
            constants.OPTIMIZATION_JOB_CREATED,
            constants.OPTIMIZATION_JOB_RUNNING,
            constants.OPTIMIZATION_JOB_COMPLETE,
            constants.OPTIMIZATION_JOB_FAILED,
        }

        if not status.issubset(job_statuses):
            raise ValueError(f"Expected status {status} must be a subset of {job_statuses}")

        if polling_interval > timeout:
            raise ValueError(
                f"Polling interval {polling_interval} must be less than timeout: {timeout}"
            )

        for _ in range(round(timeout / polling_interval)):
            optimization_job = self.get_job(name)
            logger.debug(
                f"{constants.OPTIMIZATION_JOB_KIND} {name}, status {optimization_job.status}"
            )

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
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.EXPERIMENT_PLURAL,
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

    def _get_best_trial(self, name: str) -> Optional[Trial]:
        """Get the best current Trial for the OptimizationJob"""
        optimization_job = self.__get_experiment_cr(name)

        # Get the best trial from currentOptimalTrial
        if (
            optimization_job.status
            and optimization_job.status.current_optimal_trial
            and optimization_job.status.current_optimal_trial.best_trial_name
        ):
            best_trial_name = optimization_job.status.current_optimal_trial.best_trial_name

            parameters = {}
            if optimization_job.status.current_optimal_trial.parameter_assignments:
                parameters = {
                    pa.name: pa.value
                    for pa in optimization_job.status.current_optimal_trial.parameter_assignments
                    if pa.name is not None and pa.value is not None
                }

            metrics = []
            if (
                optimization_job.status.current_optimal_trial.observation
                and optimization_job.status.current_optimal_trial.observation.metrics
            ):
                metrics = [
                    Metric(name=m.name, latest=m.latest, max=m.max, min=m.min)
                    for m in optimization_job.status.current_optimal_trial.observation.metrics
                    if m.name is not None
                    and m.latest is not None
                    and m.max is not None
                    and m.min is not None
                ]

            trainjob = self.trainer_backend.get_job(name=best_trial_name)

            return Trial(
                name=best_trial_name,
                parameters=parameters,
                metrics=metrics,
                trainjob=trainjob,
            )

        return None

    def __get_experiment_cr(self, name: str) -> models.V1beta1Experiment:
        """Get the Experiment CR from Kubernetes API"""
        try:
            thread = self.custom_api.get_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.EXPERIMENT_PLURAL,
                name,
                async_req=True,
            )

            optimization_job = models.V1beta1Experiment.from_dict(
                thread.get(common_constants.DEFAULT_TIMEOUT)  # type: ignore
            )

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
        optimization_job_cr: models.V1beta1Experiment,
    ) -> OptimizationJob:
        if not (
            optimization_job_cr.metadata
            and optimization_job_cr.metadata.name
            and optimization_job_cr.metadata.namespace
            and optimization_job_cr.spec
            and optimization_job_cr.spec.parameters
            and optimization_job_cr.spec.objective
            and optimization_job_cr.spec.algorithm
            and optimization_job_cr.spec.max_trial_count
            and optimization_job_cr.spec.parallel_trial_count
            and optimization_job_cr.metadata.creation_timestamp
        ):
            raise Exception(
                f"{constants.OPTIMIZATION_JOB_KIND} CR is invalid: {optimization_job_cr}"
            )

        optimization_job = OptimizationJob(
            name=optimization_job_cr.metadata.name,
            search_space=utils.get_search_space_from_katib_spec(
                optimization_job_cr.spec.parameters
            ),
            objectives=utils.get_objectives_from_katib_spec(optimization_job_cr.spec.objective),
            algorithm=utils.get_algorithm_from_katib_spec(optimization_job_cr.spec.algorithm),
            trial_config=TrialConfig(
                num_trials=optimization_job_cr.spec.max_trial_count,
                parallel_trials=optimization_job_cr.spec.parallel_trial_count,
                max_failed_trials=optimization_job_cr.spec.max_failed_trial_count,
            ),
            trials=self.__get_trials_from_job(optimization_job_cr.metadata.name),
            creation_timestamp=optimization_job_cr.metadata.creation_timestamp,
            status=constants.OPTIMIZATION_JOB_CREATED,  # The default OptimizationJob status.
        )

        # Update the OptimizationJob status from Experiment conditions.
        if optimization_job_cr.status and optimization_job_cr.status.conditions:
            for c in optimization_job_cr.status.conditions:
                if c.type == constants.EXPERIMENT_SUCCEEDED and c.status == "True":
                    optimization_job.status = constants.OPTIMIZATION_JOB_COMPLETE
                elif c.type == constants.OPTIMIZATION_JOB_FAILED and c.status == "True":
                    optimization_job.status = constants.OPTIMIZATION_JOB_FAILED
                else:
                    for trial in optimization_job.trials:
                        if trial.trainjob.status == trainer_constants.TRAINJOB_RUNNING:
                            optimization_job.status = constants.OPTIMIZATION_JOB_RUNNING

        return optimization_job

    def __get_trials_from_job(self, optimization_job_name: str) -> list[Trial]:
        result = []
        try:
            thread = self.custom_api.list_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRIAL_PLURAL,
                label_selector=f"{constants.EXPERIMENT_LABEL}={optimization_job_name}",
                async_req=True,
            )

            trial_list = models.V1beta1TrialList.from_dict(
                thread.get(common_constants.DEFAULT_TIMEOUT)
            )

            if not trial_list:
                return result

            for t in trial_list.items:
                if not (t.metadata and t.metadata.name and t.spec and t.spec.parameter_assignments):
                    raise ValueError(f"{constants.TRIAL_KIND} CR is invalid: {t}")

                # Trial name is equal to the TrainJob name.
                trial = Trial(
                    name=t.metadata.name,
                    parameters={
                        pa.name: pa.value
                        for pa in t.spec.parameter_assignments
                        if pa.name is not None and pa.value is not None
                    },
                    trainjob=self.trainer_backend.get_job(name=t.metadata.name),
                )
                if t.status and t.status.observation and t.status.observation.metrics:
                    trial.metrics = [
                        Metric(name=m.name, latest=m.latest, max=m.max, min=m.min)
                        for m in t.status.observation.metrics
                        if m.name is not None
                        and m.latest is not None
                        and m.max is not None
                        and m.min is not None
                    ]

                result.append(trial)

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {constants.TRIAL_KIND}s in namespace: {self.namespace}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {constants.TRIAL_KIND}s in namespace: {self.namespace}"
            ) from e

        return result
