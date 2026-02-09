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

"""Unit tests for Kubernetes Spark backend utilities."""

import pytest

from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.backends.kubernetes.utils import (
    _memory_kubernetes_to_spark,
    build_service_url,
    build_spark_connect_crd,
    generate_session_name,
    parse_spark_connect_status,
    validate_spark_connect_url,
)
from kubeflow.spark.types.types import Driver, Executor, SparkConnectInfo, SparkConnectState


class TestMemoryKubernetesToSpark:
    """Tests for _memory_kubernetes_to_spark."""

    @pytest.mark.parametrize(
        "k8s_memory,expected_spark",
        [
            ("4Gi", "4g"),
            ("512Mi", "512m"),
            ("8Gi", "8g"),
            ("1Ti", "1t"),
            ("4g", "4g"),
            ("512m", "512m"),
            ("2G", "2g"),
        ],
    )
    def test_conversion(self, k8s_memory: str, expected_spark: str) -> None:
        assert _memory_kubernetes_to_spark(k8s_memory) == expected_spark


class TestGenerateSessionName:
    """Tests for generate_session_name function."""

    def test_generates_unique_name(self):
        """U11: Generate unique session name with prefix."""
        name = generate_session_name()
        assert name.startswith("spark-connect-")
        assert len(name) > len("spark-connect-")

    def test_generates_different_names(self):
        """Generated names should be unique."""
        names = {generate_session_name() for _ in range(10)}
        assert len(names) == 10


class TestValidateSparkConnectUrl:
    """Tests for validate_spark_connect_url function."""

    def test_valid_url(self):
        """U12: Valid Spark Connect URL passes."""
        assert validate_spark_connect_url("sc://localhost:15002") is True
        assert validate_spark_connect_url("sc://spark-server:15002") is True

    def test_invalid_scheme(self):
        """U13: Invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            validate_spark_connect_url("http://localhost:15002")

    def test_missing_port(self):
        """U14: Missing port raises ValueError."""
        with pytest.raises(ValueError, match="Port is required"):
            validate_spark_connect_url("sc://localhost")


class TestBuildServiceUrl:
    """Tests for build_service_url function."""

    def test_build_from_session_info(self):
        """U15: Build service URL from SparkConnectInfo."""
        info = SparkConnectInfo(
            name="my-session",
            namespace="spark",
            state=SparkConnectState.READY,
            service_name="my-session-svc",
        )
        url = build_service_url(info)
        assert url == "sc://my-session-svc.spark.svc.cluster.local:15002"

    def test_build_without_service_name(self):
        """Build URL when service_name is None."""
        info = SparkConnectInfo(
            name="my-session",
            namespace="default",
            state=SparkConnectState.READY,
        )
        url = build_service_url(info)
        assert "my-session-svc" in url


class TestBuildSparkConnectCrd:
    """Tests for build_spark_connect_crd function."""

    def test_minimal_crd(self):
        """U01: Build SparkConnect CRD with minimal config."""
        crd = build_spark_connect_crd(name="test-session", namespace="default")

        assert (
            crd["apiVersion"]
            == f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}"
        )
        assert crd["kind"] == constants.SPARK_CONNECT_KIND
        assert crd["metadata"]["name"] == "test-session"
        assert crd["metadata"]["namespace"] == "default"
        assert crd["spec"]["sparkVersion"] == constants.DEFAULT_SPARK_VERSION
        assert crd["spec"]["executor"]["instances"] == constants.DEFAULT_NUM_EXECUTORS
        assert crd["spec"]["executor"]["cores"] == constants.DEFAULT_EXECUTOR_CPU
        assert crd["spec"]["executor"]["memory"] == "512m"
        assert crd["spec"]["server"]["cores"] == constants.DEFAULT_DRIVER_CPU
        assert crd["spec"]["server"]["memory"] == "512m"
        assert crd["spec"]["sparkConf"]["spark.connect.grpc.binding.address"] == "0.0.0.0"

    def test_with_num_executors(self):
        """U02: Build CRD with num_executors."""
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            num_executors=3,
        )
        assert crd["spec"]["executor"]["instances"] == 3

    def test_with_resources(self):
        """U03: Build CRD with resources_per_executor."""
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            resources_per_executor={"cpu": "2", "memory": "4Gi"},
        )
        assert crd["spec"]["executor"]["cores"] == 2
        assert crd["spec"]["executor"]["memory"] == "4g"

    def test_with_spark_conf(self):
        """U04: Build CRD with spark_conf."""
        spark_conf = {"spark.sql.adaptive.enabled": "true"}
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            spark_conf=spark_conf,
        )
        assert crd["spec"]["sparkConf"]["spark.jars"].endswith(
            f"spark-connect_2.12-{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert crd["spec"]["sparkConf"]["spark.sql.adaptive.enabled"] == "true"

    def test_spark_conf_overrides_binding_address(self):
        """User spark_conf can override default grpc binding address."""
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            spark_conf={"spark.connect.grpc.binding.address": "127.0.0.1"},
        )
        assert crd["spec"]["sparkConf"]["spark.connect.grpc.binding.address"] == "127.0.0.1"

    def test_with_driver_image(self):
        """U05: Build CRD with custom image via Driver."""
        driver = Driver(image="custom-spark:v1")
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert crd["spec"]["image"] == "custom-spark:v1"

    def test_with_driver_config(self):
        """U06: Build CRD with Driver config (KEP-107 resources dict)."""
        driver = Driver(resources={"cpu": "2", "memory": "2Gi"})
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert crd["spec"]["server"]["cores"] == 2
        assert crd["spec"]["server"]["memory"] == "2g"

    def test_with_service_account(self):
        """U07: Build CRD with service account."""
        driver = Driver(service_account="spark-sa")
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert crd["spec"]["server"]["template"]["spec"]["serviceAccountName"] == "spark-sa"

    def test_with_executor_config(self):
        """Build CRD with Executor config (KEP-107 resources_per_executor)."""
        executor = Executor(
            num_instances=5,
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
        )
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            executor=executor,
        )
        assert crd["spec"]["executor"]["instances"] == 5
        assert crd["spec"]["executor"]["cores"] == 4
        assert crd["spec"]["executor"]["memory"] == "8g"

    def test_app_name(self):
        """Build CRD with spark.app.name via spark_conf."""
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            spark_conf={"spark.app.name": "my-spark-app"},
        )
        assert crd["spec"]["sparkConf"]["spark.jars"].endswith(
            f"spark-connect_2.12-{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert crd["spec"]["sparkConf"]["spark.app.name"] == "my-spark-app"

    def test_precedence_executor_instances(self):
        """Test precedence: executor.num_instances > num_executors."""
        executor = Executor(num_instances=10)
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            num_executors=5,
            executor=executor,
        )
        # Executor object should override simple parameter
        assert crd["spec"]["executor"]["instances"] == 10

    def test_precedence_executor_resources(self):
        """Test precedence: executor.resources_per_executor > resources_per_executor."""
        executor = Executor(
            resources_per_executor={"cpu": "8", "memory": "16Gi"},
        )
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
            executor=executor,
        )
        # Executor object should override simple parameter
        assert crd["spec"]["executor"]["cores"] == 8
        assert crd["spec"]["executor"]["memory"] == "16g"

    def test_kep107_level2_simple(self):
        """Test KEP-107 Level 2 (simple mode) example."""
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            num_executors=5,
            resources_per_executor={"cpu": "5", "memory": "10Gi"},
        )
        assert crd["spec"]["executor"]["instances"] == 5
        assert crd["spec"]["executor"]["cores"] == 5
        assert crd["spec"]["executor"]["memory"] == "10g"

    def test_kep107_level3_advanced(self):
        """Test KEP-107 Level 3 (advanced mode) example."""
        driver = Driver(
            resources={"cpu": "4", "memory": "8Gi"},
            service_account="spark-driver-prod",
        )
        executor = Executor(
            num_instances=20,
            resources_per_executor={"cpu": "8", "memory": "32Gi"},
        )
        crd = build_spark_connect_crd(
            name="test-session",
            namespace="default",
            driver=driver,
            executor=executor,
        )
        assert crd["spec"]["server"]["cores"] == 4
        assert crd["spec"]["server"]["memory"] == "8g"
        assert (
            crd["spec"]["server"]["template"]["spec"]["serviceAccountName"] == "spark-driver-prod"
        )
        assert crd["spec"]["executor"]["instances"] == 20
        assert crd["spec"]["executor"]["cores"] == 8
        assert crd["spec"]["executor"]["memory"] == "32g"


class TestParseSparkConnectStatus:
    """Tests for parse_spark_connect_status function."""

    def test_parse_ready_status(self):
        """U08: Parse CRD with Ready state."""
        crd_response = {
            "metadata": {
                "name": "my-session",
                "namespace": "default",
                "creationTimestamp": "2025-01-12T10:30:00Z",
            },
            "status": {
                "state": "Ready",
                "server": {
                    "podName": "my-session-server-0",
                    "podIp": "10.0.0.5",
                    "serviceName": "my-session-svc",
                },
            },
        }
        info = parse_spark_connect_status(crd_response)

        assert info.name == "my-session"
        assert info.namespace == "default"
        assert info.state == SparkConnectState.READY
        assert info.pod_name == "my-session-server-0"
        assert info.pod_ip == "10.0.0.5"
        assert info.service_name == "my-session-svc"
        assert info.creation_timestamp is not None

    def test_parse_provisioning_status(self):
        """U09: Parse CRD with Provisioning state."""
        crd_response = {
            "metadata": {"name": "new-session", "namespace": "spark"},
            "status": {"state": "Provisioning"},
        }
        info = parse_spark_connect_status(crd_response)

        assert info.name == "new-session"
        assert info.namespace == "spark"
        assert info.state == SparkConnectState.PROVISIONING

    def test_parse_failed_status(self):
        """U10: Parse CRD with Failed state."""
        crd_response = {
            "metadata": {"name": "failed-session", "namespace": "default"},
            "status": {"state": "Failed"},
        }
        info = parse_spark_connect_status(crd_response)

        assert info.state == SparkConnectState.FAILED

    def test_parse_running_status(self):
        """Parse CRD with Running state (operator may set this when server is up)."""
        crd_response = {
            "metadata": {"name": "run-session", "namespace": "default"},
            "status": {
                "state": "Running",
                "server": {"podName": "run-session-server", "serviceName": "run-session-svc"},
            },
        }
        info = parse_spark_connect_status(crd_response)
        assert info.state == SparkConnectState.RUNNING
        assert info.service_name == "run-session-svc"

    def test_parse_empty_status(self):
        """Parse CRD with empty status."""
        crd_response = {
            "metadata": {"name": "new-session", "namespace": "default"},
        }
        info = parse_spark_connect_status(crd_response)

        assert info.state == SparkConnectState.PROVISIONING
        assert info.pod_name is None
