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

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from kubeflow_katib_api import models as katib_models
from pydantic import PrivateAttr

import kubeflow.optimizer.constants.constants as constants


def _is_python_literal(value: Any) -> bool:
    """Check that `str(value)` is a Python literal that evaluates back to the value."""
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (bool, int))


class ChoiceParameterSpec(katib_models.V1beta1ParameterSpec):
    """Categorical ParameterSpec that remembers the type of its values.

    Katib stores every categorical value as a string, so the original types are lost in the
    spec itself. They are needed to pass numbers and booleans to the training function as
    numbers and booleans rather than as strings.
    """

    _literal_values: bool = PrivateAttr(default=False)

    @property
    def literal_values(self) -> bool:
        """Whether every value was given as a finite number or a boolean."""
        return self._literal_values


# Search space distribution helpers
class Search:
    """Helper class for defining search space parameters."""

    @staticmethod
    def uniform(min: float, max: float) -> katib_models.V1beta1ParameterSpec:
        """Sample a float value uniformly between `min` and `max`.

        Args:
            min: Lower boundary for the float value.
            max: Upper boundary for the float value.

        Returns:
            Katib ParameterSpec object.
        """
        return katib_models.V1beta1ParameterSpec(
            parameterType=constants.DOUBLE_PARAMETER,
            feasibleSpace=katib_models.V1beta1FeasibleSpace(
                min=str(min), max=str(max), distribution=Distribution.UNIFORM.value
            ),
        )

    @staticmethod
    def loguniform(min: float, max: float) -> katib_models.V1beta1ParameterSpec:
        """Sample a float value with log-uniform distribution between `min` and `max`.

        Args:
            min: Lower boundary for the float value.
            max: Upper boundary for the float value.

        Returns:
            Katib ParameterSpec object.
        """
        return katib_models.V1beta1ParameterSpec(
            parameterType=constants.DOUBLE_PARAMETER,
            feasibleSpace=katib_models.V1beta1FeasibleSpace(
                min=str(min), max=str(max), distribution=Distribution.LOG_UNIFORM.value
            ),
        )

    @staticmethod
    def choice(values: list) -> katib_models.V1beta1ParameterSpec:
        """Sample a categorical value from the list.

        The training function receives the value with its original type when every value is
        a number or a boolean. Otherwise, every value is passed as a string.

        Args:
            values: List of categorical values.

        Returns:
            Katib ParameterSpec object.
        """
        spec = ChoiceParameterSpec(
            parameterType=constants.CATEGORICAL_PARAMETERS,
            feasibleSpace=katib_models.V1beta1FeasibleSpace(list=[str(v) for v in values]),
        )
        spec._literal_values = bool(values) and all(_is_python_literal(v) for v in values)
        return spec


# Distribution for the search space.
class Distribution(Enum):
    UNIFORM = "uniform"
    LOG_UNIFORM = "logUniform"


@dataclass
class ContinuousSearchSpace:
    min: float | int
    max: float | int
    distribution: Distribution


@dataclass
class CategoricalSearchSpace:
    choices: list
