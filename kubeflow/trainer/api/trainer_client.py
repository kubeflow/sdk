# Copyright 2024 The Kubeflow Authors.
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
from typing import Dict, List, Optional, Set, Union

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig


logger = logging.getLogger(__name__)


class TrainerClient:
    def __init__(
        self,
        backend_config: KubernetesBackendConfig = KubernetesBackendConfig(),
    ):
        """Initialize a Kubeflow Trainer client.

        Args:
            backend_config: Backend configuration. Either KubernetesBackendConfig or
                            LocalProcessBackendConfig, or None to use the backend's
                            default config class. Defaults to KubernetesBackendConfig.
        """
        # initialize training backend
        if isinstance(backend_config, KubernetesBackendConfig):
            self.__backend = KubernetesBackend(backend_config)
        else:
            raise ValueError("Invalid backend config '{}'".format(backend_config))

    def list_runtimes(self) -> types.Runtime:
        """List of the available Runtimes.

        Returns:
            List[Runtime]: List of available training runtimes.
                If no runtimes exist, an empty list is returned.

        Raises:
            TimeoutError: Timeout to list Runtimes.
            RuntimeError: Failed to list Runtimes.
        """
        return self.__backend.list_runtimes()

    def get_runtime(self, name: str) -> types.Runtime:
        """Get the Runtime object
        Args:
            name: Name of the runtime.
            Returns:
                types.TrainingRuntime: Runtime object.
        """
        return self.__backend.get_runtime(name=name)

    def get_runtime_packages(self, runtime: types.Runtime):
        """
        Print the installed Python packages for the given Runtime. If Runtime has GPUs it also
        prints available GPUs on the single training node.

        Args:
            runtime: Reference to one of existing Runtimes.

        Raises:
            ValueError: Input arguments are invalid.
            RuntimeError: Failed to get Runtime.

        """
        return self.__backend.get_runtime_packages(runtime=runtime)

    def train(
        self,
        runtime: types.Runtime = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[Union[types.CustomTrainer, types.BuiltinTrainer]] = None,
    ) -> str:
        """
        Create the TrainJob. You can configure these types of training task:
        - Custom Training Task: Training with a self-contained function that encapsulates
            the entire model training process, e.g. `CustomTrainer`.
        - Config-driven Task with Existing Trainer: Training with a trainer that already includes
            the post-training logic, requiring only parameter adjustments, e.g. `BuiltinTrainer`.
        Args:
            runtime: Reference to one of existing Runtimes.
            initializer:
                Configuration for the dataset and model initializers.
            trainer:
                Configuration for Custom Training Task or Config-driven Task with Builtin Trainer.
        Returns:
            str: The unique name of the TrainJob that has been generated.
        Raises:
            ValueError: Input arguments are invalid.
            TimeoutError: Timeout to create TrainJobs.
            RuntimeError: Failed to create TrainJobs.
        """
        return self.__backend.train(runtime=runtime, initializer=initializer, trainer=trainer)

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> List[types.TrainJob]:
        """List of all TrainJobs.

        Returns:
            List: List of created TrainJobs.
                If no TrainJob exist, an empty list is returned.

        Raises:
            TimeoutError: Timeout to list TrainJobs.
            RuntimeError: Failed to list TrainJobs.
        """
        return self.__backend.list_jobs(runtime=runtime)

    def get_job(self, name: str) -> types.TrainJob:
        """Get the TrainJob object"""
        return self.__backend.get_job(name=name)

    def get_job_logs(
        self,
        name: str,
        follow: Optional[bool] = False,
        step: str = constants.NODE,
        node_rank: int = 0,
    ) -> Dict[str, str]:
        """Get the logs from TrainJob"""
        return self.__backend.get_job_logs(name=name, follow=follow, step=step, node_rank=node_rank)

    def wait_for_job_status(
        self,
        name: str,
        status: Set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJob:
        """Wait for TrainJob to reach the desired status

        Args:
            name: Name of the TrainJob.
            status: Set of expected statuses. It must be subset of Created, Running, Complete, and
                Failed statuses.
            timeout: How many seconds to wait until TrainJob reaches one of the expected conditions.
            polling_interval: The polling interval in seconds to check TrainJob status.

        Returns:
            TrainJob: The training job that reaches the desired status.

        Raises:
            ValueError: The input values are incorrect.
            RuntimeError: Failed to get TrainJob or TrainJob reaches unexpected Failed status.
            TimeoutError: Timeout to wait for TrainJob status.
        """
        return self.__backend.wait_for_job_status(
            name=name,
            status=status,
            timeout=timeout,
            polling_interval=polling_interval,
        )

    def delete_job(self, name: str):
        """Delete the TrainJob.

        Args:
            name: Name of the TrainJob.

        Raises:
            TimeoutError: Timeout to delete TrainJob.
            RuntimeError: Failed to delete TrainJob.
        """
        return self.__backend.delete_job(name=name)
