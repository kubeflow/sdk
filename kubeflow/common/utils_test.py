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

from kubeflow.common import utils
from kubeflow.trainer.test.common import SUCCESS, TestCase


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
