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
import enum
import math
from pathlib import Path

import pytest

from kubeflow.common import utils
from kubeflow.test.common import SUCCESS, TestCase

# --------------------------
# Test Helpers
# --------------------------


def sample_function() -> None:
    """Sample function for testing."""
    print("Hello World")


async def sample_async_function() -> None:
    """Sample async function for testing."""
    pass


def sample_decorator(func):
    """Pass-through decorator for testing."""
    return func


@sample_decorator
def sample_decorated_function() -> None:
    """Sample decorated function."""
    pass


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid polling_interval and timeout",
            expected_status=SUCCESS,
            config={"polling_interval": 2, "timeout": 600},
        ),
        TestCase(
            name="timeout is zero",
            expected_status=SUCCESS,
            config={"polling_interval": 2, "timeout": 0},
            expected_error=ValueError,
            expected_output="Timeout must be a positive number",
        ),
        TestCase(
            name="polling_interval is zero",
            expected_status=SUCCESS,
            config={"polling_interval": 0, "timeout": 600},
            expected_error=ValueError,
            expected_output="Polling interval must be a positive number",
        ),
        TestCase(
            name="polling_interval is negative",
            expected_status=SUCCESS,
            config={"polling_interval": -5, "timeout": 600},
            expected_error=ValueError,
            expected_output="Polling interval must be a positive number",
        ),
        TestCase(
            name="polling_interval equals timeout",
            expected_status=SUCCESS,
            config={"polling_interval": 10, "timeout": 10},
            expected_error=ValueError,
            expected_output="Polling interval must be strictly less than timeout",
        ),
    ],
)
def test_validate_wait_for_job_status(test_case):
    """Test validate_wait_for_job_status across valid and invalid inputs."""
    print("Executing test:", test_case.name)

    polling_interval = test_case.config["polling_interval"]
    timeout = test_case.config["timeout"]

    if test_case.expected_error:
        with pytest.raises(test_case.expected_error, match=test_case.expected_output):
            utils.validate_wait_for_job_status(polling_interval, timeout)
    else:
        utils.validate_wait_for_job_status(polling_interval, timeout)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid function",
            expected_status=SUCCESS,
            config={"func": sample_function},
        ),
        TestCase(
            name="not a function",
            expected_status=SUCCESS,
            config={"func": "not a function"},
            expected_error=ValueError,
            expected_output="Function must be a Python function.",
        ),
        TestCase(
            name="lambda function",
            expected_status=SUCCESS,
            config={"func": lambda: None},
            expected_error=ValueError,
            expected_output="Lambda functions are not supported.",
        ),
        TestCase(
            name="async function",
            expected_status=SUCCESS,
            config={"func": sample_async_function},
            expected_error=ValueError,
            expected_output="Async functions are not supported.",
        ),
        TestCase(
            name="decorated function",
            expected_status=SUCCESS,
            config={"func": sample_decorated_function},
            expected_error=ValueError,
            expected_output="Decorated functions are not supported.",
        ),
    ],
)
def test_validate_python_function(test_case):
    """Test validate_python_function with valid and invalid functions."""
    print("Executing test:", test_case.name)

    func = test_case.config["func"]

    if test_case.expected_error:
        with pytest.raises(test_case.expected_error, match=test_case.expected_output):
            utils.validate_python_function(func)
    else:
        utils.validate_python_function(func)


class _Color(enum.IntEnum):
    RED = 1


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="None", expected_status=SUCCESS, config={"func_args": None}),
        TestCase(
            name="literal values",
            expected_status=SUCCESS,
            config={
                "func_args": {
                    "lr": 0.01,
                    "epochs": 3,
                    "shuffle": True,
                    "data_dir": "/data",
                    "seed": None,
                    "layers": [64, 32],
                    "betas": (0.9, 0.999),
                    "extra": {"dropout": 0.1},
                }
            },
        ),
        TestCase(
            name="not a dict",
            expected_status=SUCCESS,
            config={"func_args": [0.01]},
            expected_error=ValueError,
            expected_output="func_args must be a dict",
        ),
        TestCase(
            name="non str key",
            expected_status=SUCCESS,
            config={"func_args": {1: "a"}},
            expected_error=ValueError,
            expected_output="func_args keys must be str",
        ),
        TestCase(
            name="pathlib Path",
            expected_status=SUCCESS,
            config={"func_args": {"data_dir": Path("/data")}},
            expected_error=ValueError,
            expected_output="func_args\\['data_dir'\\] has unsupported value",
        ),
        TestCase(
            name="infinite float",
            expected_status=SUCCESS,
            config={"func_args": {"max_grad_norm": math.inf}},
            expected_error=ValueError,
            expected_output="func_args\\['max_grad_norm'\\] has unsupported value",
        ),
        TestCase(
            name="nan nested in a list",
            expected_status=SUCCESS,
            config={"func_args": {"weights": [1.0, math.nan]}},
            expected_error=ValueError,
            expected_output="func_args\\['weights'\\] has unsupported value",
        ),
        TestCase(
            name="int subclass",
            expected_status=SUCCESS,
            config={"func_args": {"color": _Color.RED}},
            expected_error=ValueError,
            expected_output="func_args\\['color'\\] has unsupported value",
        ),
        TestCase(
            name="custom object",
            expected_status=SUCCESS,
            config={"func_args": {"obj": object()}},
            expected_error=ValueError,
            expected_output="func_args\\['obj'\\] has unsupported value",
        ),
    ],
)
def test_validate_func_args(test_case):
    """Test validate_func_args with values that can and cannot be rendered as literals."""
    print("Executing test:", test_case.name)

    func_args = test_case.config["func_args"]

    if test_case.expected_error:
        with pytest.raises(test_case.expected_error, match=test_case.expected_output):
            utils.validate_func_args(func_args)
    else:
        utils.validate_func_args(func_args)
        # Every accepted value must survive the repr() round trip used by the script.
        if func_args is not None:
            assert eval(repr(func_args), {"__builtins__": {}}) == func_args
