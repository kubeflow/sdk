# Copyright 2026 The Kubeflow Authors.
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

"""Unit tests for KubernetesBackend initialization."""

from unittest.mock import Mock, patch

from kubernetes import config
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend


@pytest.mark.parametrize(
    "backend_config, in_cluster_error, expected_loader, expected_kwargs",
    [
        pytest.param(
            KubernetesBackendConfig(namespace="spark", config_file="/tmp/spark-kubeconfig"),
            None,
            "kubeconfig",
            {"config_file": "/tmp/spark-kubeconfig"},
            id="config-file",
        ),
        pytest.param(
            KubernetesBackendConfig(
                namespace="spark",
                config_file="/tmp/spark-kubeconfig",
                context="ignored-context",
            ),
            None,
            "kubeconfig",
            {"config_file": "/tmp/spark-kubeconfig"},
            id="config-file-precedes-context",
        ),
        pytest.param(
            KubernetesBackendConfig(namespace="spark", context="spark-context"),
            None,
            "kubeconfig",
            {"context": "spark-context"},
            id="context",
        ),
        pytest.param(
            KubernetesBackendConfig(namespace="spark"),
            None,
            "in-cluster",
            {},
            id="in-cluster",
        ),
        pytest.param(
            KubernetesBackendConfig(),
            config.ConfigException("not running in a cluster"),
            "kubeconfig",
            {},
            id="kubeconfig-fallback",
        ),
    ],
)
def test_backend_configuration_loading(
    backend_config: KubernetesBackendConfig,
    in_cluster_error: config.ConfigException | None,
    expected_loader: str,
    expected_kwargs: dict[str, str],
) -> None:
    """Load Kubernetes configuration from the selected source."""
    with (
        patch(
            "kubeflow.spark.backends.kubernetes.backend.config.load_incluster_config",
            side_effect=in_cluster_error,
        ) as load_incluster_config,
        patch(
            "kubeflow.spark.backends.kubernetes.backend.config.load_kube_config"
        ) as load_kube_config,
        patch(
            "kubeflow.spark.backends.kubernetes.backend.client.CustomObjectsApi",
            return_value=Mock(),
        ) as custom_objects_api,
        patch(
            "kubeflow.spark.backends.kubernetes.backend.client.CoreV1Api",
            return_value=Mock(),
        ) as core_api,
    ):
        backend = KubernetesBackend(backend_config)

    assert backend.namespace == (backend_config.namespace or "default")
    custom_objects_api.assert_called_once_with()
    core_api.assert_called_once_with()

    if expected_loader == "in-cluster":
        load_incluster_config.assert_called_once_with()
        load_kube_config.assert_not_called()
    else:
        load_kube_config.assert_called_once_with(**expected_kwargs)
        if backend_config.config_file or backend_config.context:
            load_incluster_config.assert_not_called()
        else:
            load_incluster_config.assert_called_once_with()
