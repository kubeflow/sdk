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
from typing import Optional

import kubeflow.common.types as common_types
from kubeflow.optimizer.backends.kubernetes.backend import KubernetesBackend

logger = logging.getLogger(__name__)


class OptimizerClient:
    def __init__(
        self,
        backend_config: Optional[common_types.KubernetesBackendConfig] = None,
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
            backend_config = common_types.KubernetesBackendConfig()

        if isinstance(backend_config, common_types.KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config '{backend_config}'")

    def optimize(
        self,
    ) -> str:
        """Create a OptimizationJob. You can configure the TrainJob using one of these trainers:

        - CustomTrainer: Runs training with a user-defined function that fully encapsulates the
            training process.
        - BuiltinTrainer: Uses a predefined trainer with built-in post-training logic, requiring
            only parameter configuration.

        Args:
            runtime: Optional reference to one of the existing runtimes. Defaults to the
                torch-distributed runtime if not provided.
            initializer: Optional configuration for the dataset and model initializers.
            trainer: Optional configuration for a CustomTrainer or BuiltinTrainer. If not specified,
                the TrainJob will use the runtime's default values.

        Returns:
            The unique name of the TrainJob that has been generated.

        Raises:
            ValueError: Input arguments are invalid.
            TimeoutError: Timeout to create TrainJobs.
            RuntimeError: Failed to create TrainJobs.
        """
        return self.backend.optimize()
