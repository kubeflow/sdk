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

import abc
from typing import Any, Optional

from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import Objective, OptimizationJob, TrialConfig
from kubeflow.trainer.types.types import TrainJobTemplate


class RuntimeBackend(abc.ABC):
    @abc.abstractmethod
    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        search_space: dict[str, Any],
        trial_config: Optional[TrialConfig] = None,
        objectives: Optional[list[Objective]] = None,
        algorithm: Optional[RandomSearch] = None,
    ) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def list_jobs(self) -> list[OptimizationJob]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job(self, name: str) -> OptimizationJob:
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_job(self, name: str):
        raise NotImplementedError()
