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
from unittest.mock import mock_open, patch

from kubernetes.config import kube_config
import pytest

from kubeflow.common import utils
from kubeflow.test.common import DEFAULT_NAMESPACE, SUCCESS, TestCase, write_kubeconfig

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
            name="without config_file the default kubeconfig is read",
            expected_status=SUCCESS,
            config={},
            expected_output="default-ns",
        ),
        TestCase(
            name="context is looked up in the default kubeconfig",
            expected_status=SUCCESS,
            config={"context": "other-ctx"},
            expected_output="other-ns",
        ),
        TestCase(
            name="config_file current context is used",
            expected_status=SUCCESS,
            config={"config_file": "team"},
            expected_output="team-b",
        ),
        TestCase(
            name="context is looked up in config_file",
            expected_status=SUCCESS,
            config={"config_file": "team", "context": "team-c-ctx"},
            expected_output="team-c",
        ),
        TestCase(
            name="unreadable config_file falls back to the default namespace",
            expected_status=SUCCESS,
            config={"config_file": "missing"},
            expected_output=DEFAULT_NAMESPACE,
        ),
        TestCase(
            name="in-cluster without config_file reads the service account namespace",
            expected_status=SUCCESS,
            config={"in_cluster": True},
            expected_output="sa-ns",
        ),
        TestCase(
            name="in-cluster with config_file reads config_file",
            expected_status=SUCCESS,
            config={"in_cluster": True, "config_file": "team"},
            expected_output="team-b",
        ),
    ],
)
def test_get_default_target_namespace(test_case, tmp_path, monkeypatch):
    """Test get_default_target_namespace across kubeconfig and in-cluster sources."""
    print("Executing test:", test_case.name)

    default_kubeconfig = write_kubeconfig(
        tmp_path / "default-kubeconfig",
        {"default-ctx": "default-ns", "other-ctx": "other-ns"},
        "default-ctx",
    )
    team_kubeconfig = write_kubeconfig(
        tmp_path / "team-kubeconfig",
        {"team-b-ctx": "team-b", "team-c-ctx": "team-c"},
        "team-b-ctx",
    )
    monkeypatch.setattr(kube_config, "KUBE_CONFIG_DEFAULT_LOCATION", default_kubeconfig)
    config_file = {
        None: None,
        "team": team_kubeconfig,
        "missing": str(tmp_path / "missing-kubeconfig"),
    }[test_case.config.get("config_file")]

    with (
        patch(
            "kubeflow.common.utils.is_running_in_k8s",
            return_value=test_case.config.get("in_cluster", False),
        ),
        patch("kubeflow.common.utils.open", mock_open(read_data="sa-ns"), create=True),
    ):
        namespace = utils.get_default_target_namespace(
            test_case.config.get("context"), config_file=config_file
        )

    assert namespace == test_case.expected_output
    print("test execution complete")
