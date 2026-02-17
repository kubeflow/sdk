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

import pytest

from kubeflow.optimizer.backends.kubernetes.utils import convert_value
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


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
