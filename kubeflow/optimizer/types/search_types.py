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

from kubeflow_katib_api import models as katib_models

import kubeflow.optimizer.constants.constants as constants


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

        Args:
            values: List of categorical values.

        Returns:
            Katib ParameterSpec object.
        """
        return katib_models.V1beta1ParameterSpec(
            parameterType=constants.CATEGORICAL_PARAMETERS,
            feasibleSpace=katib_models.V1beta1FeasibleSpace(list=[str(v) for v in values]),
        )


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
