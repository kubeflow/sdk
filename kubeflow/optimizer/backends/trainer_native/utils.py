# Copyright 2026 The Kubeflow Authors.
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

"""Converters between SDK types and the trainer-native OptimizationJob CRD.

The trainer-native OptimizationJob (trainer.kubeflow.org/v1alpha1) has no released
generated Python models yet, so specs are built and parsed as plain dictionaries.
Source of truth: pkg/apis/trainer/v1alpha1/optimizationjob_types.go in kubeflow/trainer.
"""

from typing import Any

from kubeflow_katib_api import models as katib_models

from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import BaseAlgorithm, GridSearch, RandomSearch
from kubeflow.optimizer.types.optimization_types import Direction, Objective
from kubeflow.optimizer.types.search_types import (
    CategoricalSearchSpace,
    ContinuousSearchSpace,
    Distribution,
)


def get_crd_parameters(
    search_space: dict[str, katib_models.V1beta1ParameterSpec],
) -> list[dict[str, Any]]:
    """Convert the SDK search space into OptimizationJob `spec.parameters`.

    Args:
        search_space: Dictionary mapping parameter names to `Search.uniform()`,
            `Search.loguniform()`, or `Search.choice()` specifications.

    Returns:
        List of parameter dictionaries for the OptimizationJob spec.

    Raises:
        ValueError: The search space specification is invalid.
    """
    parameters = []
    for param_name, param_spec in search_space.items():
        feasible_space = param_spec.feasible_space
        if param_spec.parameter_type == constants.CATEGORICAL_PARAMETERS:
            if not (feasible_space and feasible_space.list):
                raise ValueError(f"Categorical parameter '{param_name}' is invalid: {param_spec}")
            crd_search_space = {
                constants.SEARCH_SPACE_CATEGORICAL: {
                    "choices": [str(v) for v in feasible_space.list],
                },
            }
        else:
            if not (feasible_space and feasible_space.min and feasible_space.max):
                raise ValueError(f"Continuous parameter '{param_name}' is invalid: {param_spec}")
            distribution_key = (
                constants.SEARCH_SPACE_LOG_UNIFORM
                if feasible_space.distribution == Distribution.LOG_UNIFORM.value
                else constants.SEARCH_SPACE_UNIFORM
            )
            # The CRD validates min/max as decimal-pattern strings.
            crd_search_space = {
                distribution_key: {
                    "min": str(feasible_space.min),
                    "max": str(feasible_space.max),
                    "type": constants.PARAMETER_TYPE_FLOAT,
                },
            }

        parameters.append({"name": param_name, "searchSpace": crd_search_space})

    return parameters


def get_search_space_from_crd(
    parameters: list[dict[str, Any]],
) -> dict[str, ContinuousSearchSpace | CategoricalSearchSpace]:
    """Convert OptimizationJob `spec.parameters` into SDK search space types."""
    search_space: dict[str, ContinuousSearchSpace | CategoricalSearchSpace] = {}
    for parameter in parameters:
        name = parameter.get("name")
        crd_search_space = parameter.get("searchSpace") or {}
        if not name or not crd_search_space:
            raise ValueError(f"OptimizationJob parameter is invalid: {parameter}")

        if constants.SEARCH_SPACE_CATEGORICAL in crd_search_space:
            categorical = crd_search_space[constants.SEARCH_SPACE_CATEGORICAL]
            search_space[name] = CategoricalSearchSpace(
                choices=[str(v) for v in categorical.get("choices", [])]
            )
        elif constants.SEARCH_SPACE_LOG_UNIFORM in crd_search_space:
            log_uniform = crd_search_space[constants.SEARCH_SPACE_LOG_UNIFORM]
            search_space[name] = ContinuousSearchSpace(
                min=float(log_uniform["min"]),
                max=float(log_uniform["max"]),
                distribution=Distribution.LOG_UNIFORM,
            )
        elif constants.SEARCH_SPACE_UNIFORM in crd_search_space:
            uniform = crd_search_space[constants.SEARCH_SPACE_UNIFORM]
            search_space[name] = ContinuousSearchSpace(
                min=float(uniform["min"]),
                max=float(uniform["max"]),
                distribution=Distribution.UNIFORM,
            )
        else:
            raise ValueError(f"OptimizationJob search space is invalid: {crd_search_space}")

    return search_space


def get_crd_objectives(objectives: list[Objective]) -> list[dict[str, Any]]:
    """Convert SDK objectives into OptimizationJob `spec.objectives`.

    Raises:
        ValueError: More than one objective is set. The trainer-native OptimizationJob
            supports exactly one objective.
    """
    if len(objectives) != 1:
        raise ValueError(
            "The trainer-native OptimizationJob supports exactly one objective, "
            f"got {len(objectives)}"
        )

    return [
        {
            "metric": objective.metric,
            # SDK direction values are lowercase, the CRD enum is Maximize/Minimize.
            "direction": objective.direction.value.capitalize(),
        }
        for objective in objectives
    ]


def get_objectives_from_crd(objectives: list[dict[str, Any]]) -> list[Objective]:
    """Convert OptimizationJob `spec.objectives` into SDK objectives."""
    result = []
    for objective in objectives:
        metric = objective.get("metric")
        if not metric:
            raise ValueError(f"OptimizationJob objective is invalid: {objective}")
        result.append(
            Objective(
                metric=metric,
                direction=Direction(objective.get("direction", "Minimize").lower()),
            )
        )
    return result


def get_crd_search_algorithm(algorithm: BaseAlgorithm) -> dict[str, Any]:
    """Convert an SDK algorithm into OptimizationJob `spec.searchAlgorithm`.

    Raises:
        ValueError: The algorithm is not supported by the trainer-native OptimizationJob.
    """
    if isinstance(algorithm, RandomSearch):
        random: dict[str, Any] = {}
        if algorithm.random_state is not None:
            random["seed"] = algorithm.random_state
        return {"random": random}
    if isinstance(algorithm, GridSearch):
        return {"grid": {}}

    raise ValueError(
        f"The trainer-native OptimizationJob doesn't support {type(algorithm).__name__} algorithm"
    )


def get_algorithm_from_crd(search_algorithm: dict[str, Any]) -> GridSearch | RandomSearch:
    """Convert OptimizationJob `spec.searchAlgorithm` into an SDK algorithm."""
    if "grid" in search_algorithm:
        return GridSearch()
    # The CRD defaults to the random search algorithm.
    random = search_algorithm.get("random") or {}
    return RandomSearch(random_state=random.get("seed"))
