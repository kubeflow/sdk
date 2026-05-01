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
from dataclasses import dataclass, fields
from typing import Any

from kubeflow_katib_api import models


def algorithm_to_katib_spec(obj: Any) -> models.V1beta1AlgorithmSpec:
    """Convert any dataclass-based algorithm to a Katib AlgorithmSpec"""
    settings = []
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is not None:
            settings.append(
                models.V1beta1AlgorithmSetting(
                    name=f.name,
                    value=str(value),
                )
            )

    return models.V1beta1AlgorithmSpec(
        algorithmName=obj.algorithm_name,
        algorithmSettings=settings or None,
    )


# Base implementation for the search algorithm.
class BaseAlgorithm(abc.ABC):
    @property
    @abc.abstractmethod
    def algorithm_name(self) -> str:
        pass

    @abc.abstractmethod
    def _to_katib_spec(self):
        raise NotImplementedError()


@dataclass
class GridSearch(BaseAlgorithm):
    """Grid search algorithm."""

    @property
    def algorithm_name(self) -> str:
        return "grid"

    def _to_katib_spec(self):
        return algorithm_to_katib_spec(self)


@dataclass
class RandomSearch(BaseAlgorithm):
    """Random search algorithm.

    Args:
        random_state (`Optional[int]`): Random seed for reproducibility.
    """

    random_state: int | None = None

    @property
    def algorithm_name(self) -> str:
        return "random"

    def _to_katib_spec(self):
        return algorithm_to_katib_spec(self)


# Registry of supported search algorithms.
ALGORITHM_REGISTRY = {
    GridSearch().algorithm_name: GridSearch,
    RandomSearch().algorithm_name: RandomSearch,
}
