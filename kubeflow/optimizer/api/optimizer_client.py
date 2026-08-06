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

from collections.abc import Callable, Iterator
import logging
from typing import Any

from kubeflow.common.types import KubernetesBackendConfig
import kubeflow.common.utils as common_utils
from kubeflow.optimizer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import BaseAlgorithm
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
    Result,
    TrialConfig,
)
from kubeflow.trainer.types.types import Event, TrainJobTemplate

logger = logging.getLogger(__name__)


class OptimizerClient:
    def __init__(
        self,
        backend_config: KubernetesBackendConfig | None = None,
    ):
        """Initialize a Kubeflow Optimizer client.

        Args:
            backend_config: Backend configuration. Either KubernetesBackendConfig
                or None to use the default config class.
                Defaults to None (uses KubernetesBackendConfig).

        Raises:
            ValueError: If the backend configuration is invalid.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
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
        trial_config: TrialConfig | None = None,
        search_space: dict[str, Any],
        objectives: list[Objective] | None = None,
        algorithm: BaseAlgorithm | None = None,
    ) -> str:
        """Create and submit an OptimizationJob for hyperparameter tuning.

        Args:
            trial_template: The TrainJob template defining the training script.
            trial_config: Optional configuration to run Trials.
            search_space: Dictionary mapping parameter names to Search specifications using
                Search.uniform(), Search.loguniform(), Search.choice(), etc.
            objectives: Optional list of objectives to optimize. Defaults to minimizing the
                "loss" metric.
            algorithm: The optimization algorithm to use. Defaults to RandomSearch.

        Returns:
            The unique name generated for the OptimizationJob (Experiment).

        Raises:
            ValueError: If input arguments are invalid.
            TimeoutError: Timeout occurred while creating the Experiment.
            RuntimeError: Failed to create the Experiment.

        Examples:
            >>> from kubeflow.trainer import TrainJobTemplate, CustomTrainer
            >>> from kubeflow.optimizer import OptimizerClient, Search, TrialConfig
            >>> def train_fn(learning_rate, num_epochs):
            ...     pass
            >>> template = TrainJobTemplate(runtime="torch-distributed", trainer=CustomTrainer(func=train_fn))
            >>> client = OptimizerClient()
            >>> opt_id = client.optimize(
            ...     trial_template=template,
            ...     trial_config=TrialConfig(num_trials=5, parallel_trials=2),
            ...     search_space={
            ...         "learning_rate": Search.loguniform(0.001, 0.1),
            ...         "num_epochs": Search.choice([5, 10]),
            ...     },
            ... )
            >>> print(opt_id)
        """
        return self.backend.optimize(
            trial_template=trial_template,
            trial_config=trial_config,
            objectives=objectives,
            search_space=search_space,
            algorithm=algorithm,
        )

    def list_jobs(self) -> list[OptimizationJob]:
        """List created OptimizationJobs.

        Returns:
            List of created OptimizationJobs. If no OptimizationJobs exist,
            an empty list is returned.

        Raises:
            TimeoutError: Timeout occurred while listing OptimizationJobs.
            RuntimeError: Failed to list OptimizationJobs.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
            >>> jobs = client.list_jobs()
            >>> for job in jobs:
            ...     print(job.name)
        """

        return self.backend.list_jobs()

    def get_job(self, name: str) -> OptimizationJob:
        """Get the OptimizationJob object by name.

        Args:
            name: Name of the OptimizationJob.

        Returns:
            An OptimizationJob object.

        Raises:
            TimeoutError: Timeout occurred while getting the OptimizationJob.
            RuntimeError: Failed to get the OptimizationJob.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
            >>> job = client.get_job("opt-12345")
            >>> print(job.status)
        """

        return self.backend.get_job(name=name)

    def get_job_logs(
        self,
        name: str,
        trial_name: str | None = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a specific trial of an OptimizationJob.

        Args:
            name: Name of the OptimizationJob.
            trial_name: Optional name of a specific Trial. If not provided, logs from the
                current best trial are returned. If no best trial is available yet, logs
                from the first trial are returned.
            follow: Whether to stream logs in realtime as they are produced. Defaults to False.

        Returns:
            Iterator of log lines.

        Raises:
            TimeoutError: Timeout occurred while getting the OptimizationJob logs.
            RuntimeError: Failed to get the OptimizationJob logs.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
            >>> for line in client.get_job_logs(name="opt-12345"):
            ...     print(line)
        """
        return self.backend.get_job_logs(name=name, trial_name=trial_name, follow=follow)

    def get_best_results(self, name: str) -> Result | None:
        """Get the best hyperparameters and metrics from an OptimizationJob.

        This method retrieves the optimal hyperparameters and their corresponding metrics
        from the best trial found during the optimization process.

        Args:
            name: Name of the OptimizationJob.

        Returns:
            A Result object containing the best hyperparameters and metrics,
            or None if no best trial is available yet.

        Raises:
            TimeoutError: Timeout occurred while getting the best results.
            RuntimeError: Failed to get the best results for the OptimizationJob.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
            >>> best_res = client.get_best_results("opt-12345")
            >>> if best_res:
            ...     print(best_res.parameters)
        """

        return self.backend.get_best_results(name=name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.OPTIMIZATION_JOB_COMPLETE},
        timeout: int = 3600,
        polling_interval: int = 2,
        callbacks: list[Callable[[OptimizationJob], None]] | None = None,
    ) -> OptimizationJob:
        """Wait for an OptimizationJob to reach a desired status.

        Args:
            name: Name of the OptimizationJob.
            status: Expected statuses. Must be a subset of Created, Running, Complete, and
                Failed statuses. Defaults to Complete.
            timeout: Maximum number of seconds to wait for the OptimizationJob to reach one of the
                expected statuses. Defaults to 3600.
            polling_interval: The polling interval in seconds to check OptimizationJob status.
                Defaults to 2.
            callbacks: Optional list of callback functions to be invoked after each polling
                interval. Each callback should accept a single argument: the OptimizationJob object.

        Returns:
            An OptimizationJob object that reaches the desired status.

        Raises:
            ValueError: The input values are incorrect.
            RuntimeError: Failed to get OptimizationJob or OptimizationJob reaches unexpected
                Failed status.
            TimeoutError: Timeout occurred while waiting for the OptimizationJob status.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
            >>> job = client.wait_for_job_status(name="opt-12345")
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
        """Delete the OptimizationJob.

        Args:
            name: Name of the OptimizationJob.

        Raises:
            TimeoutError: Timeout occurred while deleting the OptimizationJob.
            RuntimeError: Failed to delete the OptimizationJob.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
            >>> client.delete_job("opt-12345")
        """
        return self.backend.delete_job(name=name)

    def get_job_events(self, name: str) -> list[Event]:
        """Get events for an OptimizationJob.

        This provides additional clarity about the state of the OptimizationJob
        when logs alone are not sufficient. Events include information about
        trial state changes, errors, and other significant occurrences.

        Args:
            name: Name of the OptimizationJob.

        Returns:
            A list of Event objects associated with the OptimizationJob.

        Raises:
            TimeoutError: Timeout occurred while getting the OptimizationJob events.
            RuntimeError: Failed to get the OptimizationJob events.

        Examples:
            >>> from kubeflow.optimizer import OptimizerClient
            >>> client = OptimizerClient()
            >>> events = client.get_job_events("opt-12345")
            >>> for event in events:
            ...     print(f"[{event.event_time}] {event.message}")
        """
        return self.backend.get_job_events(name=name)
