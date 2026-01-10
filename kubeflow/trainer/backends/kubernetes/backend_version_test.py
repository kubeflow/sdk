
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
from unittest.mock import MagicMock, patch
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.constants import constants
import kubeflow_trainer_api
from kubernetes import client
import logging

@pytest.fixture
def backend():
    # Mock config loading and client creation
    with patch("kubernetes.config.load_kube_config"), \
         patch("kubernetes.config.load_incluster_config"), \
         patch("kubeflow.common.utils.is_running_in_k8s", return_value=False), \
         patch("kubernetes.client.ApiClient"), \
         patch("kubernetes.client.CustomObjectsApi"), \
         patch("kubernetes.client.CoreV1Api") as mock_core_cls:
        
        cfg = KubernetesBackendConfig()
        bk = KubernetesBackend(cfg)
        # Verify that bk.core_api is the return value of mock_core_cls()
        # The constructor calls client.CoreV1Api(k8s_client), so the instance is mock_core_cls.return_value
        yield bk

def test_verify_backend_match(backend, caplog):
    # Setup mock
    mock_config_map = MagicMock()
    mock_config_map.data = {"version": kubeflow_trainer_api.__version__}
    backend.core_api.read_namespaced_config_map.return_value = mock_config_map

    with caplog.at_level(logging.WARNING):
        backend.verify_backend()
    
    # Assert no warning
    assert "Kubeflow Trainer version mismatch" not in caplog.text
    
    # Verify call arguments
    backend.core_api.read_namespaced_config_map.assert_called_with(
        name=constants.TRAINER_VERSION_CONFIG_MAP,
        namespace=constants.KUBEFLOW_NAMESPACE
    )

def test_verify_backend_mismatch(backend, caplog):
    # Setup mock
    mock_config_map = MagicMock()
    mock_config_map.data = {"version": "0.0.0"} # Mismatch
    backend.core_api.read_namespaced_config_map.return_value = mock_config_map

    with caplog.at_level(logging.WARNING):
        backend.verify_backend()

    # Assert warning
    assert "Kubeflow Trainer version mismatch" in caplog.text
    assert "Server version: 0.0.0" in caplog.text

def test_verify_backend_not_found(backend, caplog):
    # Setup mock to raise 404
    error = client.ApiException(status=404)
    backend.core_api.read_namespaced_config_map.side_effect = error
    
    with caplog.at_level(logging.WARNING):
        backend.verify_backend()
    
    # Assert warning about not found
    assert f"ConfigMap '{constants.TRAINER_VERSION_CONFIG_MAP}' not found" in caplog.text

def test_verify_backend_other_error(backend):
    # Setup mock to raise 500
    error = client.ApiException(status=500)
    backend.core_api.read_namespaced_config_map.side_effect = error
    
    with pytest.raises(client.ApiException):
        backend.verify_backend()
