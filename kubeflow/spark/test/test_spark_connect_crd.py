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

"""Unit tests for SparkConnect CRD support."""

from unittest.mock import Mock, patch

import pytest

from kubeflow.spark.backends.operator import OperatorBackend, OperatorBackendConfig
from kubeflow.spark.models import (
    DynamicAllocation,
    SparkConnectServerConfig,
    SparkConnectServerStatus,
    SparkConnectState,
)


# --------------------------
# Sample CRD Responses
# --------------------------

SAMPLE_SPARK_CONNECT_CRD = {
    "apiVersion": "sparkoperator.k8s.io/v1alpha1",
    "kind": "SparkConnect",
    "metadata": {
        "name": "test-sparkconnect",
        "namespace": "spark-jobs",
        "creationTimestamp": "2024-01-15T10:00:00Z",
    },
    "spec": {
        "sparkVersion": "3.5.0",
        "image": "apache/spark:3.5.0",
        "server": {"cores": 2, "memory": "4g"},
        "executor": {"cores": 4, "memory": "8g", "instances": 5},
    },
    "status": {
        "state": "Ready",
        "server": {
            "podName": "test-sparkconnect-server",
            "podIp": "10.0.0.1",
            "serviceName": "test-sparkconnect-svc",
        },
    },
}

SAMPLE_SPARK_CONNECT_CRD_PROVISIONING = {
    "apiVersion": "sparkoperator.k8s.io/v1alpha1",
    "kind": "SparkConnect",
    "metadata": {
        "name": "test-sparkconnect",
        "namespace": "spark-jobs",
    },
    "status": {
        "state": "Provisioning",
    },
}


# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def operator_backend():
    """Provide an OperatorBackend with mocked Kubernetes APIs."""
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(
                    return_value=Mock(get=Mock(return_value=SAMPLE_SPARK_CONNECT_CRD))
                ),
                get_namespaced_custom_object=Mock(
                    return_value=Mock(get=Mock(return_value=SAMPLE_SPARK_CONNECT_CRD))
                ),
                list_namespaced_custom_object=Mock(
                    return_value=Mock(get=Mock(return_value={"items": [SAMPLE_SPARK_CONNECT_CRD]}))
                ),
                delete_namespaced_custom_object=Mock(
                    return_value=Mock(get=Mock(return_value={}))
                ),
            ),
        ),
        patch("kubernetes.client.CoreV1Api"),
    ):
        backend = OperatorBackend(OperatorBackendConfig(namespace="spark-jobs"))
        yield backend


# --------------------------
# Model Tests
# --------------------------


class TestSparkConnectServerConfig:
    """Tests for SparkConnectServerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SparkConnectServerConfig()
        assert config.name is None
        assert config.namespace == "default"
        assert config.spark_version == "3.5.0"
        assert config.server_cores == 1
        assert config.server_memory == "1g"
        assert config.executor_cores == 1
        assert config.executor_memory == "1g"
        assert config.num_executors == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SparkConnectServerConfig(
            name="my-spark-server",
            namespace="spark-jobs",
            spark_version="3.4.0",
            server_cores=4,
            server_memory="8g",
            executor_cores=8,
            executor_memory="16g",
            num_executors=20,
            spark_conf={"spark.sql.shuffle.partitions": "400"},
        )
        assert config.name == "my-spark-server"
        assert config.namespace == "spark-jobs"
        assert config.spark_version == "3.4.0"
        assert config.server_cores == 4
        assert config.num_executors == 20
        assert config.spark_conf["spark.sql.shuffle.partitions"] == "400"

    def test_with_dynamic_allocation(self):
        """Test configuration with dynamic allocation."""
        da = DynamicAllocation(
            enabled=True,
            min_executors=2,
            max_executors=50,
            initial_executors=5,
        )
        config = SparkConnectServerConfig(dynamic_allocation=da)
        assert config.dynamic_allocation.enabled is True
        assert config.dynamic_allocation.max_executors == 50


class TestSparkConnectServerStatus:
    """Tests for SparkConnectServerStatus."""

    def test_from_crd_ready(self):
        """Test parsing status from CRD when server is ready."""
        status = SparkConnectServerStatus.from_crd(SAMPLE_SPARK_CONNECT_CRD)

        assert status.name == "test-sparkconnect"
        assert status.namespace == "spark-jobs"
        assert status.state == SparkConnectState.READY
        assert status.connect_url == "sc://test-sparkconnect-svc.spark-jobs.svc:15002"
        assert status.service_name == "test-sparkconnect-svc"
        assert status.pod_name == "test-sparkconnect-server"

    def test_from_crd_provisioning(self):
        """Test parsing status from CRD when server is provisioning."""
        status = SparkConnectServerStatus.from_crd(SAMPLE_SPARK_CONNECT_CRD_PROVISIONING)

        assert status.name == "test-sparkconnect"
        assert status.state == SparkConnectState.PROVISIONING
        assert status.connect_url is None  # Not ready yet

    def test_from_crd_unknown_state(self):
        """Test parsing status with unknown state."""
        crd = {
            "metadata": {"name": "test", "namespace": "default"},
            "status": {"state": "SomeUnknownState"},
        }
        status = SparkConnectServerStatus.from_crd(crd)
        assert status.state == SparkConnectState.UNKNOWN


class TestSparkConnectState:
    """Tests for SparkConnectState enum."""

    def test_all_states_defined(self):
        """Test all expected states are defined."""
        expected = {"PROVISIONING", "READY", "NOT_READY", "FAILED", "UNKNOWN"}
        actual = {s.name for s in SparkConnectState}
        assert expected == actual


# --------------------------
# Backend Tests
# --------------------------


class TestOperatorBackendSparkConnect:
    """Tests for OperatorBackend SparkConnect methods."""

    def test_create_connect_server(self, operator_backend):
        """Test creating a SparkConnect server."""
        config = SparkConnectServerConfig(
            name="test-server",
            server_cores=2,
            server_memory="4g",
            num_executors=5,
        )

        status = operator_backend.create_connect_server(config)

        assert status is not None
        assert status.name == "test-server"
        assert status.state == SparkConnectState.PROVISIONING
        operator_backend.custom_api.create_namespaced_custom_object.assert_called_once()

    def test_create_connect_server_auto_name(self, operator_backend):
        """Test creating a SparkConnect server with auto-generated name."""
        config = SparkConnectServerConfig()  # No name provided

        status = operator_backend.create_connect_server(config)

        assert status is not None
        assert status.name.startswith("sparkconnect-")
        assert len(status.name) == len("sparkconnect-") + 8

    def test_get_connect_server_status(self, operator_backend):
        """Test getting SparkConnect server status."""
        status = operator_backend.get_connect_server_status("test-sparkconnect")

        assert status is not None
        assert status.state == SparkConnectState.READY

    def test_list_connect_servers(self, operator_backend):
        """Test listing SparkConnect servers."""
        servers = operator_backend.list_connect_servers()

        assert len(servers) == 1
        assert servers[0].name == "test-sparkconnect"

    def test_delete_connect_server(self, operator_backend):
        """Test deleting a SparkConnect server."""
        result = operator_backend.delete_connect_server("test-sparkconnect")

        assert result["status"] == "deleted"
        operator_backend.custom_api.delete_namespaced_custom_object.assert_called_once()

    def test_build_spark_connect_crd(self, operator_backend):
        """Test building SparkConnect CRD specification."""
        config = SparkConnectServerConfig(
            name="my-server",
            spark_version="3.5.0",
            server_cores=2,
            server_memory="4g",
            executor_cores=4,
            executor_memory="8g",
            num_executors=10,
            spark_conf={"spark.sql.shuffle.partitions": "200"},
            labels={"team": "ml"},
        )

        crd = operator_backend._build_spark_connect_crd(config, "spark-jobs")

        assert crd["apiVersion"] == "sparkoperator.k8s.io/v1alpha1"
        assert crd["kind"] == "SparkConnect"
        assert crd["metadata"]["name"] == "my-server"
        assert crd["spec"]["sparkVersion"] == "3.5.0"
        assert crd["spec"]["server"]["cores"] == 2
        assert crd["spec"]["server"]["memory"] == "4g"
        assert crd["spec"]["executor"]["instances"] == 10
        assert crd["spec"]["sparkConf"]["spark.sql.shuffle.partitions"] == "200"
        assert crd["metadata"]["labels"]["team"] == "ml"

    def test_build_spark_connect_crd_with_dynamic_allocation(self, operator_backend):
        """Test building SparkConnect CRD with dynamic allocation."""
        config = SparkConnectServerConfig(
            name="dynamic-server",
            dynamic_allocation=DynamicAllocation(
                enabled=True,
                min_executors=2,
                max_executors=20,
                initial_executors=5,
            ),
        )

        crd = operator_backend._build_spark_connect_crd(config, "default")

        assert "dynamicAllocation" in crd["spec"]
        assert crd["spec"]["dynamicAllocation"]["enabled"] is True
        assert crd["spec"]["dynamicAllocation"]["minExecutors"] == 2
        assert crd["spec"]["dynamicAllocation"]["maxExecutors"] == 20

