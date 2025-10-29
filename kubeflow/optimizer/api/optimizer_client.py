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
from typing import Any, Optional

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.optimizer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.optimizer.types.algorithm_types import BaseAlgorithm
from kubeflow.optimizer.types.optimization_types import Objective, OptimizationJob, TrialConfig
from kubeflow.trainer.types.types import TrainJobTemplate

logger = logging.getLogger(__name__)


class OptimizerClient:
    def __init__(
        self,
        backend_config: Optional[KubernetesBackendConfig] = None,
    ):
        """Initialize a Kubeflow Optimizer client.

        Args:
            backend_config: Backend configuration. Either KubernetesBackendConfig or None to use
                default config class. Defaults to KubernetesBackendConfig.

        Raises:
            ValueError: Invalid backend configuration.

        """
        # Set the default backend config.
        if not backend_config:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config '{backend_config}'")

    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        trial_config: Optional[TrialConfig] = None,
        search_space: dict[str, Any],
        objectives: Optional[list[Objective]] = None,
        algorithm: Optional[BaseAlgorithm] = None,
    ) -> str:
        """Create an OptimizationJob for hyperparameter tuning.

        Args:
            trial_template: The TrainJob template defining the training script.
            trial_config: Optional configuration to run Trials.
            objectives: List of objectives to optimize.
            search_space: Dictionary mapping parameter names to Search specifications using
                Search.uniform(), Search.loguniform(), Search.choice(), etc.
            algorithm: The optimization algorithm to use. Defaults to RandomSearch.

        Returns:
            The unique name of the Experiment that has been generated.

        Raises:
            ValueError: Input arguments are invalid.
            TimeoutError: Timeout to create Experiment.
            RuntimeError: Failed to create Experiment.
        """
        return self.backend.optimize(
            trial_template=trial_template,
            trial_config=trial_config,
            objectives=objectives,
            search_space=search_space,
            algorithm=algorithm,
        )

    def list_jobs(self) -> list[OptimizationJob]:
        """List of the created OptimizationJobs

        Returns:
            List of created OptimizationJobs. If no OptimizationJob exist,
                an empty list is returned.

        Raises:
            TimeoutError: Timeout to list OptimizationJobs.
            RuntimeError: Failed to list OptimizationJobs.
        """

        return self.backend.list_jobs()

    def get_job(self, name: str) -> OptimizationJob:
        """Get the OptimizationJob object

        Args:
            name: Name of the OptimizationJob.

        Returns:
            A OptimizationJob object.

        Raises:
            TimeoutError: Timeout to get a OptimizationJob.
            RuntimeError: Failed to get a OptimizationJob.
        """

        return self.backend.get_job(name=name)

    def delete_job(self, name: str):
        """Delete the OptimizationJob.

        Args:
            name: Name of the OptimizationJob.

        Raises:
            TimeoutError: Timeout to delete OptimizationJob.
            RuntimeError: Failed to delete OptimizationJob.
        """
        return self.backend.delete_job(name=name)
