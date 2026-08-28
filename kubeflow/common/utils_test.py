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
from unittest.mock import patch

from kubernetes import client
import pytest

from kubeflow.common import utils
from kubeflow.common.types import KubernetesBackendConfig
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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="client_configuration provided explicitly",
            config={
                "client_configuration": client.Configuration(),
                "expected_host": "https://custom-k8s:6443",
            },
        ),
        TestCase(
            name="load_kube_config with config_file and context",
            config={
                "config_file": "/path/to/kubeconfig",
                "context": "test-context",
                "namespace": "target-ns",
            },
        ),
    ],
)
def test_get_k8s_client(test_case):
    """Test get_k8s_client initialization logic."""
    print("Executing test:", test_case.name)

    if "client_configuration" in test_case.config:
        custom_config = test_case.config["client_configuration"]
        custom_config.host = test_case.config["expected_host"]
        cfg = KubernetesBackendConfig(client_configuration=custom_config)
        api_client, ns = utils.get_k8s_client(cfg)
        assert api_client.configuration.host == test_case.config["expected_host"]
    else:
        with (
            patch("kubernetes.config.load_kube_config") as mock_load_kube_config,
            patch("kubeflow.common.utils.is_running_in_k8s", return_value=False),
            patch("kubeflow.common.utils.get_default_target_namespace", return_value="target-ns"),
        ):
            cfg = KubernetesBackendConfig(
                config_file=test_case.config["config_file"],
                context=test_case.config["context"],
            )
            api_client, ns = utils.get_k8s_client(cfg)
            mock_load_kube_config.assert_called_once_with(
                config_file=test_case.config["config_file"],
                context=test_case.config["context"],
            )
            assert ns == test_case.config["namespace"]
