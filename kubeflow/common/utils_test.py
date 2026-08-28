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


def test_get_default_target_namespace_in_k8s(mocker):
    """Test get_default_target_namespace strips trailing newlines when running in k8s."""
    mocker.patch("kubeflow.common.utils.is_running_in_k8s", return_value=True)
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data="my-namespace\n"))

    namespace = utils.get_default_target_namespace()

    assert namespace == "my-namespace"
    mock_open.assert_called_once_with("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def test_get_default_target_namespace_out_of_k8s_with_context(mocker):
    """Test get_default_target_namespace resolves namespace from specified or current context."""
    mocker.patch("kubeflow.common.utils.is_running_in_k8s", return_value=False)
    all_contexts = [
        {"name": "ctx1", "context": {"namespace": "ns1"}},
        {"name": "ctx2", "context": {"namespace": "ns2"}},
    ]
    current_context = {"name": "ctx1", "context": {"namespace": "ns1"}}
    mocker.patch(
        "kubernetes.config.list_kube_config_contexts",
        return_value=(all_contexts, current_context),
    )

    assert utils.get_default_target_namespace(context="ctx2") == "ns2"
    assert utils.get_default_target_namespace() == "ns1"


def test_get_default_target_namespace_out_of_k8s_fallback(mocker):
    """Test get_default_target_namespace fallback to DEFAULT_NAMESPACE on exception."""
    mocker.patch("kubeflow.common.utils.is_running_in_k8s", return_value=False)
    mocker.patch(
        "kubernetes.config.list_kube_config_contexts",
        side_effect=Exception("No kubeconfig"),
    )

    assert utils.get_default_target_namespace() == "default"
