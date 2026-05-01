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

from dataclasses import fields
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from kubeflow_katib_api import models

from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import (
    ALGORITHM_REGISTRY,
    GridSearch,
    RandomSearch,
)
from kubeflow.optimizer.types.optimization_types import Direction, Objective
from kubeflow.optimizer.types.search_types import (
    CategoricalSearchSpace,
    ContinuousSearchSpace,
    Distribution,
)


def convert_value(raw_value: str, target_type: Any):
    """Convert a string value to the target type, handling optional types.

    Args:
        raw_value: String value to convert.
        target_type: Target type. `Optional[T]` and `T | None` will be treated as `T`

    Returns:
        Converted value in the target type.
    """
    origin = get_origin(target_type)

    # `T | None` produces UnionType instead of Union before Python 3.14.
    # `Optional[T]` always produces Union
    if origin is Union or origin is UnionType:
        args = get_args(target_type)
        non_none_types = [arg for arg in args if arg is not NoneType]
        if len(non_none_types) == 1:
            target_type = non_none_types[0]

    if target_type is int:
        return int(raw_value)
    elif target_type is float:
        return float(raw_value)
    elif target_type is bool:
        return raw_value.lower() in ("true", "1")
    return raw_value


def get_algorithm_from_katib_spec(
    algorithm: models.V1beta1AlgorithmSpec,
) -> GridSearch | RandomSearch:
    alg_cls = ALGORITHM_REGISTRY.get(algorithm.algorithm_name or "")

    if alg_cls is None:
        raise ValueError(f"Kubeflow SDK doesn't support {algorithm.algorithm_name} algorithm.")

    kwargs = {}
    settings = {s.name: s.value for s in algorithm.algorithm_settings or []}

    for f in fields(alg_cls):
        raw_value = settings.get(f.name)
        if raw_value is None:
            continue

        if f.name in settings:
            kwargs[f.name] = convert_value(raw_value, f.type)

    return alg_cls(**kwargs)


def get_objectives_from_katib_spec(objective: models.V1beta1ObjectiveSpec) -> list[Objective]:
    if objective.objective_metric_name is None:
        raise ValueError("Objective metric name cannot be empty")

    # TODO (andreyvelich): Katib doesn't support multi-objective optimization.
    # Currently, the first metric is objective, and the rest is additional metrics.
    direction = Direction(objective.type)
    metrics = [objective.objective_metric_name] + (objective.additional_metric_names or [])

    return [Objective(metric=m, direction=direction) for m in metrics]


def get_search_space_from_katib_spec(
    parameters: list[models.V1beta1ParameterSpec],
) -> dict[str, ContinuousSearchSpace | CategoricalSearchSpace]:
    search_space = {}

    for p in parameters:
        if p.parameter_type == constants.CATEGORICAL_PARAMETERS:
            if not (p.feasible_space and p.feasible_space.list):
                raise ValueError(f"Katib categorical parameters are invalid: {parameters}")

            search_space[p.name] = CategoricalSearchSpace(
                choices=[str(v) for v in p.feasible_space.list]
            )
        else:
            if not (
                p.feasible_space
                and p.feasible_space.min
                and p.feasible_space.max
                and p.feasible_space.distribution
            ):
                raise ValueError(f"Katib continuous parameters are invalid: {parameters}")

            search_space[p.name] = ContinuousSearchSpace(
                min=float(p.feasible_space.min),
                max=float(p.feasible_space.max),
                distribution=Distribution(p.feasible_space.distribution),
            )

    return search_space
