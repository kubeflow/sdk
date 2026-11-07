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

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.backends.localprocess.types import LocalProcessBackendConfig
from kubeflow.trainer.types import types


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


def _train_fn(data_dir):
    print(data_dir)


def test_train_rejects_unsupported_func_args():
    """TrainerClient.train fails before submission when func_args cannot be rendered."""
    client = TrainerClient(backend_config=LocalProcessBackendConfig())
    client.backend = Mock()

    with pytest.raises(ValueError, match="func_args\\['data_dir'\\] has unsupported value"):
        client.train(
            trainer=types.CustomTrainer(func=_train_fn, func_args={"data_dir": Path("/data")})
        )

    client.backend.train.assert_not_called()


def test_train_passes_literal_func_args_to_backend():
    """TrainerClient.train forwards a CustomTrainer with literal func_args unchanged."""
    client = TrainerClient(backend_config=LocalProcessBackendConfig())
    client.backend = Mock()
    trainer = types.CustomTrainer(func=_train_fn, func_args={"data_dir": "/data"})

    client.train(trainer=trainer)

    client.backend.train.assert_called_once()
    assert client.backend.train.call_args.kwargs["trainer"] is trainer
