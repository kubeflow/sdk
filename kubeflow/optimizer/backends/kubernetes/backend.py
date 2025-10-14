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

import logging
import multiprocessing
import random
import string
from typing import Any, Optional
import uuid

from kubeflow_katib_api import models
from kubernetes import client, config

import kubeflow.common.constants as common_constants
import kubeflow.common.types as common_types
import kubeflow.common.utils as common_utils
from kubeflow.optimizer.backends.base import ExecutionBackend
from kubeflow.optimizer.backends.kubernetes import utils
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
    Trial,
    TrialConfig,
)
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend as TrainerBackend
import kubeflow.trainer.constants.constants as trainer_constants

logger = logging.getLogger(__name__)


class KubernetesBackend(ExecutionBackend):
    def __init__(
        self,
        cfg: common_types.KubernetesBackendConfig,
    ):
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
        trial_template: common_types.TrainJobTemplate,
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
                        ),
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
                f"Timeout to create OptimizationJob: {self.namespace}/{optimization_job_name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create OptimizationJob: {self.namespace}/{optimization_job_name}"
            ) from e

        logger.debug(f"OptimizationJob {self.namespace}/{optimization_job_name} has been created")

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
                result.append(self.__get_optimization_job_from_crd(optimization_job))

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list OptimizationJobs in namespace: {self.namespace}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list OptimizationJobs in namespace: {self.namespace}"
            ) from e

        return result

    def get_job(self, name: str) -> OptimizationJob:
        """Get the OptimizationJob object"""

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
            raise TimeoutError(f"Timeout to get OptimizationJob: {self.namespace}/{name}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to get OptimizationJob: {self.namespace}/{name}") from e

        return self.__get_optimization_job_from_crd(optimization_job)  # type: ignore

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
            raise TimeoutError(f"Timeout to delete OptimizationJob: {self.namespace}/{name}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to delete OptimizationJob: {self.namespace}/{name}") from e

        logger.debug(f"OptimizationJob {self.namespace}/{name} has been deleted")

    def __get_optimization_job_from_crd(
        self,
        optimization_job_crd: models.V1beta1Experiment,
    ) -> OptimizationJob:
        if not (
            optimization_job_crd.metadata
            and optimization_job_crd.metadata.name
            and optimization_job_crd.metadata.namespace
            and optimization_job_crd.spec
            and optimization_job_crd.spec.parameters
            and optimization_job_crd.spec.objective
            and optimization_job_crd.spec.algorithm
            and optimization_job_crd.spec.max_trial_count
            and optimization_job_crd.spec.parallel_trial_count
            and optimization_job_crd.metadata.creation_timestamp
        ):
            raise Exception(f"OptimizationJob CRD is invalid: {optimization_job_crd}")

        optimization_job = OptimizationJob(
            name=optimization_job_crd.metadata.name,
            search_space=utils.get_search_space_from_katib_spec(
                optimization_job_crd.spec.parameters
            ),
            objectives=utils.get_objectives_from_katib_spec(optimization_job_crd.spec.objective),
            algorithm=utils.get_algorithm_from_katib_spec(optimization_job_crd.spec.algorithm),
            trial_config=TrialConfig(
                num_trials=optimization_job_crd.spec.max_trial_count,
                parallel_trials=optimization_job_crd.spec.parallel_trial_count,
                max_failed_trials=optimization_job_crd.spec.max_failed_trial_count,
            ),
            trials=self.__get_trials_from_crd(optimization_job_crd.metadata.name),
            creation_timestamp=optimization_job_crd.metadata.creation_timestamp,
            status=constants.OPTIMIZATION_JOB_CREATED,  # The default OptimizationJob status.
        )

        # Update the OptimizationJob status from Experiment conditions.
        if optimization_job_crd.status and optimization_job_crd.status.conditions:
            for c in optimization_job_crd.status.conditions:
                if c.type == constants.EXPERIMENT_SUCCEEDED and c.status == "True":
                    optimization_job.status = constants.OPTIMIZATION_JOB_COMPLETE
                elif c.type == constants.OPTIMIZATION_JOB_FAILED and c.status == "True":
                    optimization_job.status = constants.OPTIMIZATION_JOB_FAILED
                else:
                    for trial in optimization_job.trials:
                        if trial.trainjob.status == trainer_constants.TRAINJOB_RUNNING:
                            optimization_job.status = constants.OPTIMIZATION_JOB_RUNNING

        return optimization_job

    def __get_trials_from_crd(self, optimization_job_name: str) -> list[Trial]:
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

            for trial in trial_list.items:
                if not (trial.metadata and trial.metadata.name):
                    raise ValueError(f"Trial CRD is invalid: {trial}")

                # Trial name is equal to the TrainJob name.
                result.append(
                    Trial(trainjob=self.trainer_backend.get_job(name=trial.metadata.name))
                )

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(f"Timeout to list Trials in namespace: {self.namespace}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to list Trials in namespace: {self.namespace}") from e

        return result
