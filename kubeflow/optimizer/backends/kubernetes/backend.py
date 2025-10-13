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

import kubeflow.common.types as common_types
import kubeflow.common.utils as common_utils
from kubeflow.optimizer.backends.base import ExecutionBackend
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import Objective, TrialConfig
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
        self.cfg = cfg

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
                        "spec": TrainerBackend(cfg=self.cfg)._get_trainjob_spec(
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
