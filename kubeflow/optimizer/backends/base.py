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

"""Abstract runtime backend interface for the Kubeflow Optimizer."""

import abc
from collections.abc import Callable, Iterator
from typing import Any

from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
    Result,
    TrialConfig,
)
from kubeflow.trainer.types.types import Event, TrainJobTemplate


class RuntimeBackend(abc.ABC):
    """Abstract base class for Kubeflow Optimizer runtime backends."""

    @abc.abstractmethod
    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        search_space: dict[str, Any],
        trial_config: TrialConfig | None = None,
        objectives: list[Objective] | None = None,
        algorithm: RandomSearch | None = None,
    ) -> str:
        """Create an OptimizationJob for hyperparameter tuning."""
        raise NotImplementedError()

    @abc.abstractmethod
    def list_jobs(self) -> list[OptimizationJob]:
        """List the created OptimizationJobs."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job(self, name: str) -> OptimizationJob:
        """Get an OptimizationJob by name."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job_logs(
        self,
        name: str,
        trial_name: str | None,
        follow: bool,
    ) -> Iterator[str]:
        """Get logs from a Trial of an OptimizationJob."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_best_results(self, name: str) -> Result | None:
        """Get the best hyperparameters and metrics from an OptimizationJob."""
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.OPTIMIZATION_JOB_COMPLETE},
        timeout: int = 3600,
        polling_interval: int = 2,
        callbacks: list[Callable[[OptimizationJob], None]] | None = None,
    ) -> OptimizationJob:
        """Wait for an OptimizationJob to reach a desired status."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job_events(self, name: str) -> list[Event]:
        """Get events for an OptimizationJob."""
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_job(self, name: str) -> None:
        """Delete an OptimizationJob."""
        raise NotImplementedError()
