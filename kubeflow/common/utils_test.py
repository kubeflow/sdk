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


def test_get_default_target_namespace_from_current_context():
    contexts = [
        {"name": "ctx-1", "context": {"namespace": "ns-1"}},
    ]
    curr = {"name": "ctx-1", "context": {"namespace": "ns-1"}}
    with (
        pytest.MonkeyPatch().context() as mp,
    ):
        mp.setattr(utils, "is_running_in_k8s", lambda: False)
        with pytest.MonkeyPatch().context() as mp2:
            from kubernetes import config

            mp2.setattr(
                config, "list_kube_config_contexts", lambda config_file=None: (contexts, curr)
            )
            ns = utils.get_default_target_namespace()
            assert ns == "ns-1"


def test_get_default_target_namespace_from_specified_context():
    contexts = [
        {"name": "ctx-1", "context": {"namespace": "ns-1"}},
        {"name": "ctx-2", "context": {"namespace": "ns-2"}},
    ]
    curr = {"name": "ctx-1", "context": {"namespace": "ns-1"}}
    from kubernetes import config

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(utils, "is_running_in_k8s", lambda: False)
        mp.setattr(config, "list_kube_config_contexts", lambda config_file=None: (contexts, curr))
        ns = utils.get_default_target_namespace(context="ctx-2")
        assert ns == "ns-2"


def test_get_default_target_namespace_with_config_file():
    contexts = [
        {"name": "remote-ctx", "context": {"namespace": "team-b"}},
    ]
    curr = {"name": "remote-ctx", "context": {"namespace": "team-b"}}
    from kubernetes import config

    passed_config_file = []

    def mock_list(config_file=None):
        passed_config_file.append(config_file)
        return contexts, curr

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(utils, "is_running_in_k8s", lambda: False)
        mp.setattr(config, "list_kube_config_contexts", mock_list)
        ns = utils.get_default_target_namespace(config_file="/custom/kubeconfig")
        assert ns == "team-b"
        assert passed_config_file == ["/custom/kubeconfig"]


def test_get_default_target_namespace_with_config_file_and_context():
    contexts = [
        {"name": "ctx-a", "context": {"namespace": "team-a"}},
        {"name": "ctx-b", "context": {"namespace": "team-b"}},
    ]
    curr = {"name": "ctx-a", "context": {"namespace": "team-a"}}
    from kubernetes import config

    passed_config_file = []

    def mock_list(config_file=None):
        passed_config_file.append(config_file)
        return contexts, curr

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(utils, "is_running_in_k8s", lambda: False)
        mp.setattr(config, "list_kube_config_contexts", mock_list)
        ns = utils.get_default_target_namespace(context="ctx-b", config_file="/custom/kubeconfig")
        assert ns == "team-b"
        assert passed_config_file == ["/custom/kubeconfig"]


def test_get_default_target_namespace_in_k8s_without_config_file():
    from unittest.mock import mock_open, patch

    mocked_open = mock_open(read_data="in-cluster-ns\n")
    with (
        patch("kubeflow.common.utils.is_running_in_k8s", return_value=True),
        patch("builtins.open", mocked_open),
    ):
        ns = utils.get_default_target_namespace()
        assert ns == "in-cluster-ns"
        mocked_open.assert_called_once_with(
            "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        )


def test_get_default_target_namespace_without_current_context():
    import os
    import tempfile

    import yaml

    cfg_no_current = {
        "apiVersion": "v1",
        "kind": "Config",
        "contexts": [
            {
                "name": "team-b-ctx",
                "context": {"cluster": "c1", "user": "u1", "namespace": "team-b"},
            }
        ],
        "clusters": [{"name": "c1", "cluster": {"server": "https://127.0.0.1:6443"}}],
        "users": [{"name": "u1", "user": {"token": "dummy"}}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg_no_current, f)
        path = f.name

    try:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(utils, "is_running_in_k8s", lambda: False)
            ns = utils.get_default_target_namespace(context="team-b-ctx", config_file=path)
            assert ns == "team-b"
    finally:
        os.remove(path)


def test_get_default_target_namespace_in_k8s_with_explicit_config_file():
    contexts = [
        {"name": "ext-ctx", "context": {"namespace": "external-ns"}},
    ]
    curr = {"name": "ext-ctx", "context": {"namespace": "external-ns"}}
    from kubernetes import config

    passed_config_file = []

    def mock_list(config_file=None):
        passed_config_file.append(config_file)
        return contexts, curr

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(utils, "is_running_in_k8s", lambda: True)
        mp.setattr(config, "list_kube_config_contexts", mock_list)
        ns = utils.get_default_target_namespace(config_file="/custom/kubeconfig")
        assert ns == "external-ns"
        assert passed_config_file == ["/custom/kubeconfig"]


def test_get_default_target_namespace_fallback_on_exception():
    from kubernetes import config

    def mock_list(config_file=None):
        raise RuntimeError("failed to read contexts")

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(utils, "is_running_in_k8s", lambda: False)
        mp.setattr(config, "list_kube_config_contexts", mock_list)
        ns = utils.get_default_target_namespace(config_file="/nonexistent")
        assert ns == "default"
