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


def test_get_job_progress():
    """Test TrainerClient.get_job_progress method."""
    from datetime import datetime

    from kubeflow.trainer.types import types

    # Mock the backend.get_job method
    mock_backend = Mock()

    # Create a sample TrainJob
    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch:latest",
        ),
    )

    job = types.TrainJob(
        name="test-job",
        runtime=runtime,
        steps=[
            types.Step(
                name="step-1",
                status="Succeeded",
                pod_name="pod-1",
            ),
            types.Step(
                name="step-2",
                status="Running",
                pod_name="pod-2",
            ),
        ],
        num_nodes=2,
        creation_timestamp=datetime.now(),
        status="Running",
    )

    mock_backend.get_job.return_value = job

    client = TrainerClient(backend_config=LocalProcessBackendConfig())
    client.backend = mock_backend

    # Get progress
    progress = client.get_job_progress("test-job")

    # Verify the progress object
    assert progress.job_name == "test-job"
    assert progress.overall_status == "Running"
    assert progress.total_steps == 2
    assert progress.completed_steps == 1
    assert progress.running_steps == ["step-2"]
    assert progress.failed_steps == []
    assert progress.healthy_pods == 2
    assert progress.total_pods == 2
    assert progress.completion_percentage == 50.0

    # Verify backend.get_job was called with the correct name
    mock_backend.get_job.assert_called_once_with(name="test-job")


def test_get_job_progress_string_output():
    """Test TrainerClient.get_job_progress returns readable string."""
    from datetime import datetime

    from kubeflow.trainer.types import types

    # Mock the backend.get_job method
    mock_backend = Mock()

    # Create a sample TrainJob
    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch:latest",
        ),
    )

    job = types.TrainJob(
        name="my-training-job",
        runtime=runtime,
        steps=[
            types.Step(
                name="initialization",
                status="Succeeded",
                pod_name="pod-1",
            ),
            types.Step(
                name="training",
                status="Running",
                pod_name="pod-2",
            ),
        ],
        num_nodes=2,
        creation_timestamp=datetime.now(),
        status="Running",
    )

    mock_backend.get_job.return_value = job

    client = TrainerClient(backend_config=LocalProcessBackendConfig())
    client.backend = mock_backend

    # Get progress and convert to string
    progress = client.get_job_progress("my-training-job")
    progress_str = str(progress)

    # Verify the string representation contains expected information
    assert "my-training-job" in progress_str
    assert "Running" in progress_str
    assert "50.0%" in progress_str
