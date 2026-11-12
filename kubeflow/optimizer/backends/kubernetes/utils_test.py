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

from kubeflow_katib_api import models
import pytest

from kubeflow.optimizer.backends.kubernetes.utils import (
    convert_value,
    get_search_space_from_katib_spec,
)
from kubeflow.optimizer.types.search_types import (
    ContinuousSearchSpace,
    Distribution,
    IntegerSearchSpace,
)
from kubeflow.test.common import FAILED, SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="int", config={"args": ("42", int)}, expected_output=42),
        TestCase(name="float", config={"args": ("3.14", float)}, expected_output=3.14),
        TestCase(name="bool-1", config={"args": ("1", bool)}, expected_output=True),
        TestCase(name="bool-0", config={"args": ("0", bool)}, expected_output=False),
        TestCase(name="bool-true", config={"args": ("true", bool)}, expected_output=True),
        TestCase(name="bool-false", config={"args": ("false", bool)}, expected_output=False),
        TestCase(name="unhandled-type", config={"args": ("hello", list)}, expected_output="hello"),
        TestCase(name="single-type-union", config={"args": ("42", int | None)}, expected_output=42),
        TestCase(
            name="multi-type-union", config={"args": ("42", int | str | None)}, expected_output="42"
        ),
    ],
)
def test_convert_value(test_case: TestCase):
    """Test convert_value handles both basic types and T | None syntax."""
    print("Executing test:", test_case.name)
    try:
        result = convert_value(*test_case.config["args"])
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert isinstance(e, test_case.expected_error)
    else:
        assert test_case.expected_status == SUCCESS
        assert result == test_case.expected_output
        assert isinstance(result, type(test_case.expected_output))
    print("test execution complete")


@pytest.mark.parametrize("step", [None, "16"])
def test_get_search_space_preserves_integer_bounds_and_step(step: str | None) -> None:
    """Integer Katib parameters should retain their discrete search space."""
    parameter = models.V1beta1ParameterSpec(
        name="batch_size",
        parameterType="int",
        feasibleSpace=models.V1beta1FeasibleSpace(
            min="16", max="128", step=step, distribution="uniform"
        ),
    )

    search_space = get_search_space_from_katib_spec([parameter])["batch_size"]

    assert isinstance(search_space, IntegerSearchSpace)
    assert type(search_space.min) is int
    assert type(search_space.max) is int
    assert search_space.min == 16
    assert search_space.max == 128
    assert search_space.step == (int(step) if step else None)
    assert search_space.distribution is Distribution.UNIFORM


def test_get_search_space_keeps_double_parameters_continuous() -> None:
    """Double Katib parameters continue to return floating bounds."""
    parameter = models.V1beta1ParameterSpec(
        name="learning_rate",
        parameterType="double",
        feasibleSpace=models.V1beta1FeasibleSpace(
            min="0.001", max="0.1", distribution="logUniform"
        ),
    )

    search_space = get_search_space_from_katib_spec([parameter])["learning_rate"]

    assert isinstance(search_space, ContinuousSearchSpace)
    assert type(search_space.min) is float
    assert type(search_space.max) is float
    assert search_space.distribution is Distribution.LOG_UNIFORM
