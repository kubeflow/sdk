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

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union

import kubeflow.common.constants as common_constants
from kubeflow.optimizer.types.algorithm_types import GridSearch, RandomSearch
from kubeflow.optimizer.types.search_types import CategoricalSearchSpace, ContinuousSearchSpace
from kubeflow.trainer.types.types import TrainJob


# Direction for optimization objective
class Direction(Enum):
    """Direction for optimization objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


# Configuration for the objective metric
@dataclass
class Objective:
    """Objective configuration for hyperparameter optimization.

    Args:
        metric (`str`): The name of the metric to optimize. Defaults to "loss".
        direction (`Direction`): Whether to maximize or minimize the metric. Defaults to "minimize".
    """

    metric: str = "loss"
    direction: Direction = Direction.MINIMIZE

    def __post_init__(self):
        if isinstance(self.direction, str):
            self.direction = Direction(self.direction)


# Configuration for trial execution
@dataclass
class TrialConfig:
    """Trial configuration for hyperparameter optimization.

    Args:
        num_trials (`int`): Number of trials to run. Defaults to 10.
        parallel_trials (`int`): Number of trials to run in parallel. Defaults to 1.
        max_failed_trials (`Optional[int]`): Maximum number of failed trials before stopping.
    """

    num_trials: int = 10
    parallel_trials: int = 1
    max_failed_trials: Optional[int] = None


@dataclass
class Metric:
    name: str
    min: str
    max: str
    latest: str


# Representation of the single trial
@dataclass
class Trial:
    """Representation for a trial.

    Args:
        name (`str`): The name of the Trial.
        parameters (`dict[str, str]`): Hyperparameters assigned to this Trial.
        metrics (`list[Metric]`): Observed metrics for this Trial. The metrics are collected
            only for completed Trials.
        trainjob (`TrainJob`): Representation of the TrainJob
    """

    name: str
    parameters: dict[str, str]
    trainjob: TrainJob
    metrics: list[Metric] = field(default_factory=list)


# Representation for the OptimizationJob
@dataclass
class OptimizationJob:
    """Representation for an optimization job.

    Args:
        name (`str`): The name of the OptimizationJob.
        objectives (`list[Objective]`): The objective configuration. Currently, only the
            first metric defined in the objectives list is optimized. Any additional metrics are
            collected and displayed in the Trial results.
        algorithm (`RandomSearch`): The algorithm configuration.
        trial_config (`TrialConfig`): The trial configuration.
        trials (`list[Trial]`): The list of created Trials.
        creation_timestamp (`datetime`): The creation timestamp.
        status (`str`): The current status of the optimization job.
    """

    name: str
    search_space: dict[str, Union[ContinuousSearchSpace, CategoricalSearchSpace]]
    objectives: list[Objective]
    algorithm: Union[GridSearch, RandomSearch]
    trial_config: TrialConfig
    trials: list[Trial]
    creation_timestamp: datetime
    status: str = common_constants.UNKNOWN
