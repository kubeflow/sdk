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

"""
Unit tests for TrainerClient backend selection.
"""

from unittest.mock import Mock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.backends.localprocess.types import LocalProcessBackendConfig
from kubeflow.trainer.test.common import TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "default_backend_is_kubernetes",
            "backend_config": None,
            "expected_backend": "KubernetesBackend",
            "use_k8s_mocks": True,
        },
        {
            "name": "local_process_backend_selection",
            "backend_config": LocalProcessBackendConfig(),
            "expected_backend": "LocalProcessBackend",
            "use_k8s_mocks": False,
        },
        {
            "name": "kubernetes_backend_selection",
            "backend_config": KubernetesBackendConfig(),
            "expected_backend": "KubernetesBackend",
            "use_k8s_mocks": True,
        },
    ],
)
def test_backend_selection(test_case):
    """Test TrainerClient backend selection logic."""
    if test_case["use_k8s_mocks"]:
        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CustomObjectsApi") as mock_custom_api,
            patch("kubernetes.client.CoreV1Api") as mock_core_api,
        ):
            mock_custom_api.return_value = Mock()
            mock_core_api.return_value = Mock()

            if test_case["backend_config"]:
                client = TrainerClient(backend_config=test_case["backend_config"])
            else:
                client = TrainerClient()

            backend_name = client.backend.__class__.__name__
            assert backend_name == test_case["expected_backend"]
    else:
        client = TrainerClient(backend_config=test_case["backend_config"])
        backend_name = client.backend.__class__.__name__
        assert backend_name == test_case["expected_backend"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="job exists returns True",
            config={"job_name": "existing-job"},
            expected_output=True,
        ),
        TestCase(
            name="job not found raises RuntimeError returns False",
            config={"job_name": "missing-k8s-job"},
            expected_error=RuntimeError,
            expected_output=False,
        ),
        TestCase(
            name="job not found raises ValueError returns False",
            config={"job_name": "missing-local-job"},
            expected_error=ValueError,
            expected_output=False,
        ),
    ],
)
def test_job_exists(test_case: TestCase):
    """Test job_exists method handles success and expected backend exceptions."""
    client = TrainerClient(backend_config=LocalProcessBackendConfig())

    with patch.object(client, "get_job") as mock_get_job:
        if test_case.expected_error:
            mock_get_job.side_effect = test_case.expected_error("Job not found")
        else:
            mock_get_job.return_value = Mock()

        result = client.job_exists(test_case.config["job_name"])
        assert result == test_case.expected_output
        mock_get_job.assert_called_with(name=test_case.config["job_name"])
