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

from kubeflow.common.types import KubernetesBackendConfig
import kubeflow.common.utils as common_utils
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
        backend_config: KubernetesBackendConfig
        | LocalProcessBackendConfig
        | ContainerBackendConfig
        | None = None,
    ):
        """Initialize a Kubeflow Trainer client.

        Args:
            backend_config: Backend configuration. Either KubernetesBackendConfig,
                LocalProcessBackendConfig, ContainerBackendConfig, or None to use
                the backend's default config class. Defaults to None (uses KubernetesBackendConfig).

        Raises:
            ValueError: If the backend configuration is invalid.

        Examples:
            >>> from kubeflow.trainer import TrainerClient, LocalProcessBackendConfig
            >>> client = TrainerClient(backend_config=LocalProcessBackendConfig())
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
        """List the available training runtimes.

        This method lists available runtimes, preferring namespace-scoped over
        cluster-scoped runtimes in case of duplicate names.

        Returns:
            A list of training runtimes from namespace-scoped
            and cluster-scoped resources. Returns an empty list if no runtimes are found.

        Raises:
            TimeoutError: Timeout occurred while listing runtimes.
            RuntimeError: Failed to list runtimes.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> runtimes = client.list_runtimes()
            >>> for runtime in runtimes:
            ...     print(runtime.name)
        """
        return self.backend.list_runtimes()

    def get_runtime(self, name: str) -> types.Runtime:
        """Get a specific training runtime.

        Args:
            name: Name of the runtime.

        Returns:
            A runtime object. If both namespace-scoped and
            cluster-scoped runtimes exist with the same name, the namespace-scoped
            runtime is returned.

        Raises:
            TimeoutError: Timeout occurred while retrieving the runtime.
            RuntimeError: Failed to get the runtime.
            ValueError: If the runtime does not exist (LocalProcess and Container backends).

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> runtime = client.get_runtime("torch-distributed")
            >>> print(runtime.trainer.image)
        """
        return self.backend.get_runtime(name=name)

    def get_runtime_packages(self, runtime: types.Runtime):
        """Print the installed Python packages and GPU info for the given runtime.

        If a runtime has GPUs, it also prints available GPUs on the training node.

        Args:
            runtime: Reference to an existing training runtime.

        Raises:
            ValueError: If the input runtime is invalid.
            RuntimeError: Failed to retrieve packages for the runtime.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> runtime = client.get_runtime("torch-distributed")
            >>> client.get_runtime_packages(runtime)
        """
        return self.backend.get_runtime_packages(runtime=runtime)

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
        """Create and submit a TrainJob.

        You can configure the TrainJob using one of these trainers:

        - CustomTrainer: Runs training with a user-defined function that fully encapsulates the
          training process.
        - CustomTrainerContainer: Runs training with a user-defined image that fully encapsulates
          the training process.
        - BuiltinTrainer: Uses a predefined trainer with built-in post-training logic, requiring
          only parameter configuration.

        Args:
            runtime: Optional reference to one of the existing runtimes. It can accept the runtime
                name or Runtime object. Defaults to the torch-distributed runtime if not provided.
            initializer: Optional configuration for the dataset and model initializers.
            trainer: Optional configuration for a CustomTrainer, CustomTrainerContainer, or
                BuiltinTrainer. If not specified, the TrainJob will use the runtime's
                default values.
            options: Optional list of configuration options to apply to the TrainJob.
                Options can be imported from kubeflow.trainer.options.

        Returns:
            The unique name generated for the TrainJob.

        Raises:
            ValueError: If input arguments are invalid.
            TimeoutError: Timeout occurred while creating the TrainJob.
            RuntimeError: Failed to create the TrainJob.

        Examples:
            >>> from kubeflow.trainer import TrainerClient, CustomTrainer
            >>> def train_fn():
            ...     print("Training function")
            >>> client = TrainerClient()
            >>> job_id = client.train(runtime="torch-distributed", trainer=CustomTrainer(func=train_fn))
            >>> print(job_id)
        """
        return self.backend.train(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            options=options,
        )

    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        """List created TrainJobs.

        Args:
            runtime: Optional reference to one of the existing runtimes. If specified,
                only TrainJobs associated with that runtime are returned.

        Returns:
            List of created TrainJobs. If no TrainJobs exist, an
            empty list is returned.

        Raises:
            TimeoutError: Timeout occurred while listing TrainJobs.
            RuntimeError: Failed to list TrainJobs.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> jobs = client.list_jobs()
            >>> for job in jobs:
            ...     print(job.name)
        """
        return self.backend.list_jobs(runtime=runtime)

    def get_job(self, name: str) -> types.TrainJob:
        """Get the TrainJob object by name.

        Args:
            name: Name of the TrainJob.

        Returns:
            A TrainJob object.

        Raises:
            TimeoutError: Timeout occurred while getting the TrainJob.
            RuntimeError: Failed to get the TrainJob.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> job = client.get_job("s8d44aa4fb6d")
            >>> print(job.status)
        """

        return self.backend.get_job(name=name)

    def get_job_logs(
        self,
        name: str,
        step: str = constants.NODE + "-0",
        follow: bool | None = False,
    ) -> Iterator[str]:
        """Get logs from a specific step of a TrainJob.

        Args:
            name: Name of the TrainJob.
            step: Step of the TrainJob to collect logs from, like "dataset-initializer"
                or "node-0". Defaults to "node-0".
            follow: Whether to stream logs in realtime as they are produced. Defaults to False.

        Returns:
            Iterator of log lines.

        Raises:
            TimeoutError: Timeout occurred while getting the TrainJob logs.
            RuntimeError: Failed to get the TrainJob logs.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> for logline in client.get_job_logs(name="s8d44aa4fb6d", follow=True):
            ...     print(logline)
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
            TimeoutError: Timeout occurred while getting the TrainJob events.
            RuntimeError: Failed to get the TrainJob events.
            NotImplementedError: If the client uses the LocalProcess or Container backend.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> events = client.get_job_events("s8d44aa4fb6d")
            >>> for event in events:
            ...     print(f"[{event.event_time}] {event.message}")
        """
        return self.backend.get_job_events(name=name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: list[Callable[[types.TrainJob], None]] | None = None,
    ) -> types.TrainJob:
        """Wait for a TrainJob to reach a desired status.

        Args:
            name: Name of the TrainJob.
            status: Expected statuses. Must be a subset of Created, Running, Complete, and
                Failed statuses. Defaults to Complete.
            timeout: Maximum number of seconds to wait for the TrainJob to reach one of the
                expected statuses. Defaults to 600.
            polling_interval: The polling interval in seconds to check TrainJob status.
                Defaults to 2.
            callbacks: Optional list of callback functions to be invoked after each polling
                interval. Each callback should accept a single argument: the TrainJob object.

        Returns:
            A TrainJob object that has reached one of the desired statuses.

        Raises:
            ValueError: The input values are incorrect.
            RuntimeError: Failed to get TrainJob or TrainJob reaches unexpected Failed status.
            TimeoutError: Timeout occurred while waiting for the TrainJob status.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> job = client.wait_for_job_status(name="s8d44aa4fb6d")
            >>> print(job.status)
        """
        common_utils.validate_wait_for_job_status(polling_interval, timeout)

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
            TimeoutError: Timeout occurred while deleting the TrainJob.
            RuntimeError: Failed to delete the TrainJob.

        Examples:
            >>> from kubeflow.trainer import TrainerClient
            >>> client = TrainerClient()
            >>> client.delete_job("s8d44aa4fb6d")
        """
        return self.backend.delete_job(name=name)
