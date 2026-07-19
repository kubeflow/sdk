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

from kubeflow_spark_api import models
import pytest

from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.backends.kubernetes.utils import (
    _memory_kubernetes_to_spark,
    build_service_url,
    build_spark_connect_cr,
    generate_session_name,
    get_executor_spec_from_executor,
    get_server_spec_from_driver,
    get_spark_connect_info_from_cr,
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


class TestBuildSparkConnectCr:
    """Tests for build_spark_connect_cr function."""

    def test_minimal_cr(self):
        """U01: Build SparkConnect CR with minimal config."""
        spark_connect = build_spark_connect_cr(name="test-session", namespace="default")

        assert (
            spark_connect.api_version
            == f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}"
        )
        assert spark_connect.kind == constants.SPARK_CONNECT_KIND
        assert spark_connect.metadata.name == "test-session"
        assert spark_connect.metadata.namespace == "default"
        assert spark_connect.spec.spark_version == constants.DEFAULT_SPARK_VERSION
        assert spark_connect.spec.executor.instances == constants.DEFAULT_NUM_EXECUTORS
        assert spark_connect.spec.executor.cores == constants.DEFAULT_EXECUTOR_CPU
        assert spark_connect.spec.executor.memory == "512m"
        assert spark_connect.spec.server.cores == constants.DEFAULT_DRIVER_CPU
        assert spark_connect.spec.server.memory == "512m"
        assert spark_connect.spec.spark_conf["spark.connect.grpc.binding.address"] == "0.0.0.0"

    def test_with_num_executors(self):
        """U02: Build CR with num_executors."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            num_executors=3,
        )
        assert spark_connect.spec.executor.instances == 3

    def test_with_resources(self):
        """U03: Build CR with resources_per_executor."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            resources_per_executor={"cpu": "2", "memory": "4Gi"},
        )
        assert spark_connect.spec.executor.cores == 2
        assert spark_connect.spec.executor.memory == "4g"

    def test_with_spark_conf(self):
        """U04: Build CR with spark_conf."""
        spark_conf = {"spark.sql.adaptive.enabled": "true"}
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            spark_conf=spark_conf,
        )
        assert spark_connect.spec.spark_conf["spark.jars"].endswith(
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert spark_connect.spec.spark_conf["spark.sql.adaptive.enabled"] == "true"

    def test_spark_conf_overrides_binding_address(self):
        """User spark_conf can override default grpc binding address."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            spark_conf={"spark.connect.grpc.binding.address": "127.0.0.1"},
        )
        assert spark_connect.spec.spark_conf["spark.connect.grpc.binding.address"] == "127.0.0.1"

    def test_with_driver_image(self):
        """U05: Build CR with custom image via Driver."""
        driver = Driver(image="custom-spark:v1")
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert spark_connect.spec.image == "custom-spark:v1"

    def test_with_driver_config(self):
        """U06: Build CR with Driver config (KEP-107 resources dict)."""
        driver = Driver(resources={"cpu": "2", "memory": "2Gi"})
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert spark_connect.spec.server.cores == 2
        assert spark_connect.spec.server.memory == "2g"

    def test_with_service_account(self):
        """U07: Build CR with service account."""
        driver = Driver(service_account="spark-sa")
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert spark_connect.spec.server.template.spec.service_account_name == "spark-sa"

    def test_with_executor_config(self):
        """Build CR with Executor config (KEP-107 resources_per_executor)."""
        executor = Executor(
            num_instances=5,
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
        )
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            executor=executor,
        )
        assert spark_connect.spec.executor.instances == 5
        assert spark_connect.spec.executor.cores == 4
        assert spark_connect.spec.executor.memory == "8g"

    def test_app_name(self):
        """Build CR with spark.app.name via spark_conf."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            spark_conf={"spark.app.name": "my-spark-app"},
        )
        assert spark_connect.spec.spark_conf["spark.jars"].endswith(
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert spark_connect.spec.spark_conf["spark.app.name"] == "my-spark-app"

    def test_precedence_executor_instances(self):
        """Test precedence: executor.num_instances > num_executors."""
        executor = Executor(num_instances=10)
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            num_executors=5,
            executor=executor,
        )
        # Executor object should override simple parameter
        assert spark_connect.spec.executor.instances == 10

    def test_precedence_executor_resources(self):
        """Test precedence: executor.resources_per_executor > resources_per_executor."""
        executor = Executor(
            resources_per_executor={"cpu": "8", "memory": "16Gi"},
        )
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
            executor=executor,
        )
        # Executor object should override simple parameter
        assert spark_connect.spec.executor.cores == 8
        assert spark_connect.spec.executor.memory == "16g"

    def test_kep107_level2_simple(self):
        """Test KEP-107 Level 2 (simple mode) example."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            num_executors=5,
            resources_per_executor={"cpu": "5", "memory": "10Gi"},
        )
        assert spark_connect.spec.executor.instances == 5
        assert spark_connect.spec.executor.cores == 5
        assert spark_connect.spec.executor.memory == "10g"

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
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
            executor=executor,
        )
        assert spark_connect.spec.server.cores == 4
        assert spark_connect.spec.server.memory == "8g"
        assert spark_connect.spec.server.template.spec.service_account_name == "spark-driver-prod"
        assert spark_connect.spec.executor.instances == 20
        assert spark_connect.spec.executor.cores == 8
        assert spark_connect.spec.executor.memory == "32g"


class TestGetSparkConnectInfoFromCr:
    """Tests for get_spark_connect_info_from_cr function."""

    @pytest.fixture
    def minimal_spec(self):
        """Create minimal spec required for SparkConnect model."""
        return models.SparkV1alpha1SparkConnectSpec(
            sparkVersion=constants.DEFAULT_SPARK_VERSION,
            server=models.SparkV1alpha1ServerSpec(),
            executor=models.SparkV1alpha1ExecutorSpec(),
        )

    def test_parse_ready_status(self, minimal_spec):
        """U08: Parse CR with Ready state."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="my-session",
                namespace="default",
                creationTimestamp="2025-01-12T10:30:00Z",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(
                state="Ready",
                server=models.SparkV1alpha1SparkConnectServerStatus(
                    podName="my-session-server-0",
                    podIp="10.0.0.5",
                    serviceName="my-session-svc",
                ),
            ),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.name == "my-session"
        assert info.namespace == "default"
        assert info.state == SparkConnectState.READY
        assert info.driver_pod_name == "my-session-server-0"
        assert info.pod_ip == "10.0.0.5"
        assert info.service_name == "my-session-svc"
        assert info.creation_timestamp is not None

    def test_parse_provisioning_status(self, minimal_spec):
        """U09: Parse CR with Provisioning state."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="new-session",
                namespace="spark",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(state="Provisioning"),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.name == "new-session"
        assert info.namespace == "spark"
        assert info.state == SparkConnectState.PROVISIONING

    def test_parse_failed_status(self, minimal_spec):
        """U10: Parse CR with Failed state."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="failed-session",
                namespace="default",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(state="Failed"),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.state == SparkConnectState.FAILED

    def test_parse_running_status(self, minimal_spec):
        """Parse CR with Running state (operator may set this when server is up)."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="run-session",
                namespace="default",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(
                state="Running",
                server=models.SparkV1alpha1SparkConnectServerStatus(
                    podName="run-session-server",
                    serviceName="run-session-svc",
                ),
            ),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)
        assert info.state == SparkConnectState.RUNNING
        assert info.service_name == "run-session-svc"

    def test_parse_empty_status(self, minimal_spec):
        """Parse CR with empty status."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="new-session",
                namespace="default",
            ),
            spec=minimal_spec,
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.state == SparkConnectState.PROVISIONING
        assert info.driver_pod_name is None

    def test_invalid_cr_missing_name_raises_error(self, minimal_spec):
        """Test that CR without name in metadata raises ValueError."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                namespace="default",
            ),
            spec=minimal_spec,
        )
        with pytest.raises(ValueError, match="SparkConnect CR is invalid"):
            get_spark_connect_info_from_cr(spark_connect_cr)


class TestGetServerSpecFromDriver:
    """Direct unit tests for get_server_spec_from_driver."""

    def test_defaults_when_no_driver(self):
        """Default CPU, memory, and no template when driver is None."""
        spec = get_server_spec_from_driver(None)
        assert spec.cores == constants.DEFAULT_DRIVER_CPU
        assert spec.memory == _memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY)
        assert spec.template is None

    def test_defaults_when_driver_has_no_resources(self):
        """Defaults hold when a Driver object is provided but resources are empty."""
        spec = get_server_spec_from_driver(Driver())
        assert spec.cores == constants.DEFAULT_DRIVER_CPU
        assert spec.memory == _memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY)
        assert spec.template is None

    @pytest.mark.parametrize(
        "cpu,memory,expected_cores,expected_memory",
        [
            ("2", "4Gi", 2, "4g"),
            ("4", "8Gi", 4, "8g"),
            ("1", "512Mi", 1, "512m"),
        ],
    )
    def test_custom_resources(
        self, cpu: str, memory: str, expected_cores: int, expected_memory: str
    ) -> None:
        """Custom CPU and memory resources are converted correctly."""
        driver = Driver(resources={"cpu": cpu, "memory": memory})
        spec = get_server_spec_from_driver(driver)
        assert spec.cores == expected_cores
        assert spec.memory == expected_memory
        assert spec.template is None

    def test_service_account_sets_pod_template(self):
        """Service account creates a PodTemplateSpec with the correct SA name."""
        driver = Driver(service_account="spark-sa")
        spec = get_server_spec_from_driver(driver)
        assert spec.template is not None
        assert spec.template.spec.service_account_name == "spark-sa"

    def test_service_account_with_resources(self):
        """Service account and custom resources are applied together."""
        driver = Driver(resources={"cpu": "3", "memory": "6Gi"}, service_account="my-sa")
        spec = get_server_spec_from_driver(driver)
        assert spec.cores == 3
        assert spec.memory == "6g"
        assert spec.template.spec.service_account_name == "my-sa"

    def test_cpu_only_resource_leaves_memory_at_default(self):
        """Only CPU is overridden; memory stays at default."""
        driver = Driver(resources={"cpu": "8"})
        spec = get_server_spec_from_driver(driver)
        assert spec.cores == 8
        assert spec.memory == _memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY)


class TestGetExecutorSpecFromExecutor:
    """Direct unit tests for get_executor_spec_from_executor."""

    def test_all_defaults_when_no_args(self):
        """Default instances, cores, and memory when all arguments are None."""
        spec = get_executor_spec_from_executor(None, None, None)
        assert spec.instances == constants.DEFAULT_NUM_EXECUTORS
        assert spec.cores == constants.DEFAULT_EXECUTOR_CPU
        assert spec.memory == _memory_kubernetes_to_spark(constants.DEFAULT_EXECUTOR_MEMORY)

    def test_num_executors_simple_mode(self):
        """num_executors simple-mode parameter sets instance count."""
        spec = get_executor_spec_from_executor(None, 5, None)
        assert spec.instances == 5

    def test_resources_per_executor_simple_mode(self):
        """resources_per_executor simple-mode parameter sets cores and memory."""
        spec = get_executor_spec_from_executor(None, None, {"cpu": "4", "memory": "8Gi"})
        assert spec.cores == 4
        assert spec.memory == "8g"

    @pytest.mark.parametrize(
        "num_instances,resources,expected_instances,expected_cores,expected_memory",
        [
            (3, {"cpu": "2", "memory": "4Gi"}, 3, 2, "4g"),
            (10, {"cpu": "8", "memory": "16Gi"}, 10, 8, "16g"),
            (1, {"cpu": "1", "memory": "512Mi"}, 1, 1, "512m"),
        ],
    )
    def test_executor_advanced_mode(
        self,
        num_instances: int,
        resources: dict,
        expected_instances: int,
        expected_cores: int,
        expected_memory: str,
    ) -> None:
        """Executor advanced-mode object overrides simple params for all fields."""
        executor = Executor(num_instances=num_instances, resources_per_executor=resources)
        spec = get_executor_spec_from_executor(executor, 1, {"cpu": "1", "memory": "1Gi"})
        assert spec.instances == expected_instances
        assert spec.cores == expected_cores
        assert spec.memory == expected_memory

    def test_executor_instances_precedence_over_num_executors(self):
        """executor.num_instances overrides num_executors simple parameter."""
        executor = Executor(num_instances=7)
        spec = get_executor_spec_from_executor(executor, 3, None)
        assert spec.instances == 7

    def test_executor_resources_precedence_over_simple_resources(self):
        """executor.resources_per_executor overrides resources_per_executor simple param."""
        executor = Executor(resources_per_executor={"cpu": "6", "memory": "12Gi"})
        spec = get_executor_spec_from_executor(executor, None, {"cpu": "2", "memory": "4Gi"})
        assert spec.cores == 6
        assert spec.memory == "12g"

    def test_memory_only_resource_leaves_cores_at_default(self):
        """Only memory is overridden via simple params; cores stay at default."""
        spec = get_executor_spec_from_executor(None, None, {"memory": "16Gi"})
        assert spec.cores == constants.DEFAULT_EXECUTOR_CPU
        assert spec.memory == "16g"
