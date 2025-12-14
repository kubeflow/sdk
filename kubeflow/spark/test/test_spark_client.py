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

"""Unit tests for unified SparkClient."""

from unittest.mock import Mock, patch

import pytest

from kubeflow.spark.models import DynamicAllocation
from kubeflow.spark.spark_client import SparkClient, SparkClientBuilder


# --------------------------
# Builder Tests
# --------------------------


class TestSparkClientBuilder:
    """Tests for SparkClientBuilder."""

    def test_builder_creation(self):
        """Test builder can be created."""
        builder = SparkClient.builder()
        assert isinstance(builder, SparkClientBuilder)

    def test_namespace_setting(self):
        """Test namespace configuration."""
        builder = SparkClient.builder().namespace("spark-jobs")
        assert builder._namespace == "spark-jobs"

    def test_server_configuration(self):
        """Test server resource configuration."""
        builder = SparkClient.builder().server(cores=4, memory="8g")
        assert builder._server_cores == 4
        assert builder._server_memory == "8g"

    def test_executor_configuration(self):
        """Test executor resource configuration."""
        builder = SparkClient.builder().executor(cores=8, memory="16g")
        assert builder._executor_cores == 8
        assert builder._executor_memory == "16g"

    def test_num_executors(self):
        """Test number of executors configuration."""
        builder = SparkClient.builder().num_executors(20)
        assert builder._num_executors == 20

    def test_image_configuration(self):
        """Test custom image configuration."""
        builder = SparkClient.builder().image("my-spark:3.5.0")
        assert builder._image == "my-spark:3.5.0"

    def test_spark_version(self):
        """Test Spark version configuration."""
        builder = SparkClient.builder().spark_version("3.4.0")
        assert builder._spark_version == "3.4.0"

    def test_spark_conf(self):
        """Test single Spark configuration property."""
        builder = SparkClient.builder().spark_conf("spark.sql.shuffle.partitions", "200")
        assert builder._spark_conf["spark.sql.shuffle.partitions"] == "200"

    def test_spark_confs(self):
        """Test multiple Spark configuration properties."""
        conf = {
            "spark.sql.adaptive.enabled": "true",
            "spark.driver.maxResultSize": "2g",
        }
        builder = SparkClient.builder().spark_confs(conf)
        assert builder._spark_conf["spark.sql.adaptive.enabled"] == "true"
        assert builder._spark_conf["spark.driver.maxResultSize"] == "2g"

    def test_hadoop_conf(self):
        """Test Hadoop configuration property."""
        builder = SparkClient.builder().hadoop_conf("fs.s3a.endpoint", "s3.amazonaws.com")
        assert builder._hadoop_conf["fs.s3a.endpoint"] == "s3.amazonaws.com"

    def test_connect_url(self):
        """Test connect URL configuration."""
        builder = SparkClient.builder().connect_url("sc://spark-cluster:15002")
        assert builder._connect_url == "sc://spark-cluster:15002"

    def test_cleanup_on_exit(self):
        """Test cleanup configuration."""
        builder = SparkClient.builder().cleanup_on_exit(False)
        assert builder._cleanup_on_exit is False

    def test_timeout(self):
        """Test timeout configuration."""
        builder = SparkClient.builder().timeout(600)
        assert builder._timeout == 600

    def test_kube_config(self):
        """Test Kubernetes configuration."""
        builder = SparkClient.builder().kube_config(
            config_file="/home/user/.kube/config",
            context="my-cluster",
        )
        assert builder._config_file == "/home/user/.kube/config"
        assert builder._context == "my-cluster"

    def test_dynamic_allocation(self):
        """Test dynamic allocation configuration."""
        builder = SparkClient.builder().dynamic_allocation(
            enabled=True, min_executors=2, max_executors=20
        )
        assert builder._dynamic_allocation.enabled is True
        assert builder._dynamic_allocation.min_executors == 2
        assert builder._dynamic_allocation.max_executors == 20

    def test_method_chaining(self):
        """Test fluent method chaining."""
        builder = (
            SparkClient.builder()
            .namespace("spark-jobs")
            .server(cores=2, memory="4g")
            .executor(cores=4, memory="8g")
            .num_executors(10)
            .spark_version("3.5.0")
            .spark_conf("spark.sql.shuffle.partitions", "200")
        )

        assert builder._namespace == "spark-jobs"
        assert builder._server_cores == 2
        assert builder._executor_cores == 4
        assert builder._num_executors == 10


# --------------------------
# Client Tests
# --------------------------


class TestSparkClient:
    """Tests for SparkClient."""

    @pytest.fixture
    def mock_operator_backend(self):
        """Create a mock operator backend."""
        with (
            patch("kubernetes.config.load_kube_config", return_value=None),
            patch("kubernetes.client.CustomObjectsApi"),
            patch("kubernetes.client.CoreV1Api"),
        ):
            yield

    def test_build_creates_client(self, mock_operator_backend):
        """Test that build() creates a SparkClient."""
        client = SparkClient.builder().namespace("test").build()
        assert isinstance(client, SparkClient)
        client.stop()

    def test_connect_class_method(self, mock_operator_backend):
        """Test SparkClient.connect() class method."""
        client = SparkClient.connect("sc://spark-cluster:15002")
        assert isinstance(client, SparkClient)
        assert client._builder._connect_url == "sc://spark-cluster:15002"
        client.stop()

    def test_context_manager(self, mock_operator_backend):
        """Test using SparkClient as context manager."""
        with SparkClient.builder().build() as client:
            assert isinstance(client, SparkClient)
        # stop() should have been called

    def test_stop_cleanup(self, mock_operator_backend):
        """Test that stop() cleans up resources."""
        client = SparkClient.builder().build()
        client.stop()
        assert client._operator_backend is None


class TestSparkClientAutoProvisioning:
    """Tests for SparkClient auto-provisioning feature."""

    @pytest.fixture
    def mock_backends(self):
        """Create mock backends for auto-provisioning tests."""
        mock_operator = Mock()
        mock_operator.create_connect_server.return_value = Mock(
            name="test-server",
            namespace="default",
            state="Provisioning",
        )
        mock_operator.wait_for_connect_server_ready.return_value = Mock(
            name="test-server",
            namespace="default",
            connect_url="sc://test-server-svc.default.svc:15002",
        )
        mock_operator.delete_connect_server.return_value = {"status": "deleted"}
        mock_operator.close.return_value = None

        with (
            patch("kubernetes.config.load_kube_config", return_value=None),
            patch("kubernetes.client.CustomObjectsApi"),
            patch("kubernetes.client.CoreV1Api"),
            patch(
                "kubeflow.spark.spark_client.OperatorBackend",
                return_value=mock_operator,
            ),
        ):
            yield mock_operator

    def test_provision_connect_server_called(self, mock_backends):
        """Test that auto-provisioning is triggered when no connect_url."""
        # This test verifies the provisioning flow is set up correctly
        client = SparkClient.builder().num_executors(5).build()

        # Manually trigger provisioning
        client._provision_connect_server()

        # Verify methods were called
        mock_backends.create_connect_server.assert_called_once()
        mock_backends.wait_for_connect_server_ready.assert_called_once()

        client.stop()

    def test_no_provisioning_with_connect_url(self, mock_backends):
        """Test that no provisioning happens when connect_url is provided."""
        client = SparkClient.builder().connect_url("sc://existing:15002").build()

        # _provision_connect_server should not be called
        assert client._provisioned_server is False

        client.stop()

    def test_cleanup_deletes_provisioned_server(self, mock_backends):
        """Test that cleanup deletes auto-provisioned server."""
        client = SparkClient.builder().build()

        # Simulate provisioning
        client._provision_connect_server()
        client._provisioned_server = True

        # Stop should delete the server
        client.stop()

        mock_backends.delete_connect_server.assert_called_once()

    def test_no_cleanup_when_disabled(self, mock_backends):
        """Test that cleanup is skipped when cleanup_on_exit is False."""
        client = SparkClient.builder().cleanup_on_exit(False).build()

        # Simulate provisioning
        client._provision_connect_server()
        client._provisioned_server = True

        # Stop should NOT delete the server
        client.stop()

        mock_backends.delete_connect_server.assert_not_called()

