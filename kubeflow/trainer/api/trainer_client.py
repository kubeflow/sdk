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

from collections.abc import Callable, Iterator
import logging
from typing import Optional, Union

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.backends.container.backend import ContainerBackend
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.backends.localprocess.backend import (
    LocalProcessBackend,
    LocalProcessBackendConfig,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class TrainerClient:
    def __init__(
        self,
        backend_config: Optional[
            Union[
                KubernetesBackendConfig,
                LocalProcessBackendConfig,
                ContainerBackendConfig,
            ]
        ] = None,
    ):
        """Initialize a Kubeflow Trainer client.

        Args:
            backend_config: Backend configuration. Either KubernetesBackendConfig,
                            LocalProcessBackendConfig, ContainerBackendConfig,
                            or None to use the backend's default config class.
                            Defaults to KubernetesBackendConfig.

        Raises:
            ValueError: Invalid backend configuration.

        """
        # Set the default backend config.
        if not backend_config:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        elif isinstance(backend_config, LocalProcessBackendConfig):
            self.backend = LocalProcessBackend(backend_config)
        elif isinstance(backend_config, ContainerBackendConfig):
            self.backend = ContainerBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config '{backend_config}'")

    def list_runtimes(self) -> list[types.Runtime]:
        """List of the available runtimes.

        Returns:
            A list of available training runtimes. If no runtimes exist, an empty list is returned.

        Raises:
            TimeoutError: Timeout to list runtimes.
            RuntimeError: Failed to list runtimes.
        """
        return self.backend.list_runtimes()

    def get_runtime(self, name: str) -> types.Runtime:
        """Get the runtime object
        Args:
            name: Name of the runtime.

        Returns:
            A runtime object.
        """
        return self.backend.get_runtime(name=name)

    def get_runtime_packages(self, runtime: types.Runtime):
        """Print the installed Python packages for the given runtime. If a runtime has GPUs it also
        prints available GPUs on the single training node.

        Args:
            runtime: Reference to one of existing runtimes.

        Raises:
            ValueError: Input arguments are invalid.
            RuntimeError: Failed to get Runtime.

        """
        return self.backend.get_runtime_packages(runtime=runtime)

    def train(
        self,
        runtime: Optional[Union[str, types.Runtime]] = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[
            Union[types.CustomTrainer, types.CustomTrainerContainer, types.BuiltinTrainer]
        ] = None,
        options: Optional[list] = None,
    ) -> str:
        """Create a TrainJob. You can configure the TrainJob using one of these trainers:

        - CustomTrainer: Runs training with a user-defined function that fully encapsulates the
            training process.
        - CustomTrainerContainer: Runs training with a user-defined image that fully encapsulates
            the training process.
        - BuiltinTrainer: Uses a predefined trainer with built-in post-training logic, requiring
            only parameter configuration.

        Args:
            runtime: Optional reference to one of the existing runtimes. It can accept the runtime
                name or Runtime object from the `get_runtime()` API.
                Defaults to the torch-distributed runtime if not provided.
            initializer: Optional configuration for the dataset and model initializers.
            trainer: Optional configuration for a CustomTrainer, CustomTrainerContainer, or
                BuiltinTrainer. If not specified, the TrainJob will use the
                runtime's default values.
            options: Optional list of configuration options to apply to the TrainJob.
                Options can be imported from kubeflow.trainer.options.

        Returns:
            The unique name of the TrainJob that has been generated.

        Raises:
            ValueError: Input arguments are invalid.
            TimeoutError: Timeout to create TrainJobs.
            RuntimeError: Failed to create TrainJobs.
        """
        return self.backend.train(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            options=options,
        )

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> list[types.TrainJob]:
        """List of the created TrainJobs. If a runtime is specified, only TrainJobs associated with
        that runtime are returned.

        Args:
            runtime: Reference to one of the existing runtimes.

        Returns:
            List of created TrainJobs. If no TrainJob exist, an empty list is returned.

        Raises:
            TimeoutError: Timeout to list TrainJobs.
            RuntimeError: Failed to list TrainJobs.
        """
        return self.backend.list_jobs(runtime=runtime)

    def get_job(self, name: str) -> types.TrainJob:
        """Get the TrainJob object

        Args:
            name: Name of the TrainJob.

        Returns:
            A TrainJob object.

        Raises:
            TimeoutError: Timeout to get a TrainJob.
            RuntimeError: Failed to get a TrainJob.
        """

        return self.backend.get_job(name=name)

    def get_job_logs(
        self,
        name: str,
        step: str = constants.NODE + "-0",
        follow: Optional[bool] = False,
    ) -> Iterator[str]:
        """Get logs from a specific step of a TrainJob.

        You can watch for the logs in realtime as follows:
        ```python
        from kubeflow.trainer import TrainerClient

        for logline in TrainerClient().get_job_logs(name="s8d44aa4fb6d", follow=True):
            print(logline)
        ```

        Args:
            name: Name of the TrainJob.
            step: Step of the TrainJob to collect logs from, like dataset-initializer or node-0.
            follow: Whether to stream logs in realtime as they are produced.

        Returns:
            Iterator of log lines.


        Raises:
            TimeoutError: Timeout to get a TrainJob.
            RuntimeError: Failed to get a TrainJob.
        """
        return self.backend.get_job_logs(name=name, follow=follow, step=step)

    def get_job_events(self, name: str) -> list[types.Event]:
        """Get events for a TrainJob.

        This provides additional clarity about the state of the TrainJob
        when logs alone are not sufficient. Events include information about
        pod state changes, errors, and other significant occurrences.

        Args:
            name: Name of the TrainJob.

        Returns:
            A list of Event objects associated with the TrainJob.

        Raises:
            TimeoutError: Timeout to get a TrainJob events.
            RuntimeError: Failed to get a TrainJob events.
        """
        return self.backend.get_job_events(name=name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: Optional[list[Callable[[types.TrainJob], None]]] = None,
    ) -> types.TrainJob:
        """Wait for a TrainJob to reach a desired status.

        Args:
            name: Name of the TrainJob.
            status: Expected statuses. Must be a subset of Created, Running, Complete, and
                Failed statuses.
            timeout: Maximum number of seconds to wait for the TrainJob to reach one of the
                expected statuses.
            polling_interval: The polling interval in seconds to check TrainJob status.
            callbacks: Optional list of callback functions to be invoked after each polling
                interval. Each callback should accept a single argument: the TrainJob object.

        Returns:
            A TrainJob object that reaches the desired status.

        Raises:
            ValueError: The input values are incorrect.
            RuntimeError: Failed to get TrainJob or TrainJob reaches unexpected Failed status.
            TimeoutError: Timeout to wait for TrainJob status.
        """
        return self.backend.wait_for_job_status(
            name=name,
            status=status,
            timeout=timeout,
            polling_interval=polling_interval,
            callbacks=callbacks,
        )

    def delete_job(self, name: str):
        """Delete the TrainJob.

        Args:
            name: Name of the TrainJob.

        Raises:
            TimeoutError: Timeout to delete TrainJob.
            RuntimeError: Failed to delete TrainJob.
        """
        return self.backend.delete_job(name=name)
