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


@pytest.fixture
def trainer_client():
    """Create a TrainerClient with a mocked LocalProcessBackend."""
    return TrainerClient(backend_config=LocalProcessBackendConfig())


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "job_exists_returns_true",
            "get_job_side_effect": None,
            "expected_result": True,
        },
        {
            "name": "job_exists_returns_false_on_value_error",
            "get_job_side_effect": ValueError("No TrainJob with name test-job"),
            "expected_result": False,
        },
        {
            "name": "job_exists_returns_false_on_runtime_error",
            "get_job_side_effect": RuntimeError("Failed to get TrainJob"),
            "expected_result": False,
        },
    ],
)
def test_job_exists(trainer_client, test_case):
    """Test TrainerClient.job_exists() across found, not-found, and error cases."""
    with patch.object(trainer_client.backend, "get_job") as mock_get_job:
        if test_case["get_job_side_effect"] is not None:
            mock_get_job.side_effect = test_case["get_job_side_effect"]
        else:
            mock_get_job.return_value = Mock()

        assert trainer_client.job_exists("test-job") == test_case["expected_result"]


def test_job_exists_propagates_timeout_error(trainer_client):
    """Test TrainerClient.job_exists() lets a TimeoutError propagate instead of masking it."""
    with patch.object(trainer_client.backend, "get_job") as mock_get_job:
        mock_get_job.side_effect = TimeoutError("Timeout to get TrainJob")

        with pytest.raises(TimeoutError):
            trainer_client.job_exists("test-job")
