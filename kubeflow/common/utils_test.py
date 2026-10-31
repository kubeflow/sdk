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
import warnings

import pytest

from kubeflow.common import constants, utils
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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="component env var set takes precedence without warning",
            expected_status=SUCCESS,
            config={
                "env": {
                    constants.KUBEFLOW_SPARK_NAMESPACE_ENV_VAR: "custom-spark-ns",
                    constants.KUBEFLOW_SYSTEM_NAMESPACE_ENV_VAR: "deprecated-ns",
                },
                "kwargs": {
                    "component_env_var": constants.KUBEFLOW_SPARK_NAMESPACE_ENV_VAR,
                },
                "expected_namespace": "custom-spark-ns",
                "expect_warning": False,
            },
        ),
        TestCase(
            name="deprecated system env var fallback emits warning",
            expected_status=SUCCESS,
            config={
                "env": {
                    constants.KUBEFLOW_SYSTEM_NAMESPACE_ENV_VAR: "system-ns",
                },
                "kwargs": {
                    "component_env_var": constants.KUBEFLOW_SPARK_NAMESPACE_ENV_VAR,
                },
                "expected_namespace": "system-ns",
                "expect_warning": True,
            },
        ),
        TestCase(
            name="default namespace when no env var is set",
            expected_status=SUCCESS,
            config={
                "env": {},
                "kwargs": {
                    "component_env_var": constants.KUBEFLOW_SPARK_NAMESPACE_ENV_VAR,
                },
                "expected_namespace": "kubeflow-system",
                "expect_warning": False,
            },
        ),
        TestCase(
            name="custom default namespace override when no env var is set",
            expected_status=SUCCESS,
            config={
                "env": {},
                "kwargs": {
                    "component_env_var": constants.KUBEFLOW_SPARK_NAMESPACE_ENV_VAR,
                    "default": "custom-default",
                },
                "expected_namespace": "custom-default",
                "expect_warning": False,
            },
        ),
    ],
)
def test_get_component_namespace(test_case, monkeypatch):
    """Test get_component_namespace resolution hierarchy and deprecation warnings."""
    monkeypatch.delenv(constants.KUBEFLOW_SPARK_NAMESPACE_ENV_VAR, raising=False)
    monkeypatch.delenv(constants.KUBEFLOW_SYSTEM_NAMESPACE_ENV_VAR, raising=False)

    for k, v in test_case.config["env"].items():
        monkeypatch.setenv(k, v)

    kwargs = test_case.config["kwargs"]
    expected_ns = test_case.config["expected_namespace"]

    if test_case.config["expect_warning"]:
        with pytest.warns(UserWarning, match="KUBEFLOW_SYSTEM_NAMESPACE is deprecated"):
            ns = utils.get_component_namespace(**kwargs)
    else:
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            ns = utils.get_component_namespace(**kwargs)
            assert len(recorded_warnings) == 0

    assert ns == expected_ns
