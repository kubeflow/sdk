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

from datetime import datetime
import multiprocessing
from unittest.mock import Mock, patch

from kubeflow_spark_api import models
import pytest

from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.backends.kubernetes.utils import (
    _memory_kubernetes_to_spark,
    _resolve_driver_resources,
    _resolve_executor_resources,
    _validate_cpu_value,
    build_service_url,
    build_spark_application_cr,
    build_spark_connect_cr,
    generate_job_name,
    generate_session_name,
    get_spark_application_info_from_cr,
    get_spark_connect_info_from_cr,
    get_spark_job_driver_spec,
    get_spark_job_executor_spec,
    read_pod_logs,
    validate_spark_connect_url,
)
from kubeflow.spark.types.types import (
    Driver,
    Executor,
    SparkConnectInfo,
    SparkConnectState,
    SparkJobStatus,
)


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
            ("1.5Gi", "1536m"),
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


class TestGenerateJobName:
    """Tests for generate_job_name function."""

    def test_generates_unique_name(self):
        name = generate_job_name()

        assert name.startswith("spark-job-")
        assert len(name) > len("spark-job-")

    def test_generates_different_names(self):
        names = {generate_job_name() for _ in range(10)}

        assert len(names) == 10


class TestValidateCpuValue:
    """Tests for _validate_cpu_value."""

    @pytest.mark.parametrize(
        "cpu,expected",
        [
            ("1", 1),
            ("4", 4),
            ("1.5", 2),
            ("500m", 1),
            ("1500m", 2),
            ("2500m", 3),
            (" 1500m ", 2),
            (2, 2),
            (16, 16),
        ],
    )
    def test_valid_cpu_values(self, cpu, expected):
        """Valid CPU values are converted to Spark cores."""
        assert _validate_cpu_value(cpu) == expected

    @pytest.mark.parametrize(
        "cpu",
        [
            None,
            "",
            " ",
            "abc",
            "50O0m",
            "0",
            "-1",
            "-500m",
            "1.5m",
            "nan",
            "inf",
            0,
            -1,
            2048,
        ],
    )
    def test_invalid_cpu_values(self, cpu):
        """Invalid CPU values raise ValueError."""
        with pytest.raises(ValueError):
            _validate_cpu_value(cpu)


class TestResolveDriverResources:
    """Tests for _resolve_driver_resources."""

    def test_defaults(self):
        """Default driver resources are returned when no Driver is provided."""

        cores, memory = _resolve_driver_resources()

        assert cores == constants.DEFAULT_DRIVER_CPU
        assert memory == _memory_kubernetes_to_spark(
            constants.DEFAULT_DRIVER_MEMORY,
        )

    def test_driver_resources(self):
        """Driver resources override defaults."""

        driver = Driver(
            resources={
                "cpu": "2",
                "memory": "4Gi",
            },
        )

        cores, memory = _resolve_driver_resources(driver)

        assert cores == 2
        assert memory == "4g"

    def test_driver_fractional_memory(self):
        """Fractional Kubernetes memory is converted to MiB."""

        driver = Driver(
            resources={
                "cpu": "2",
                "memory": "1.5Gi",
            },
        )

        cores, memory = _resolve_driver_resources(driver)

        assert cores == 2
        assert memory == "1536m"


class TestResolveExecutorResources:
    """Tests for _resolve_executor_resources."""

    def test_defaults(self):
        """Default executor resources are returned."""

        instances, cores, memory = _resolve_executor_resources()

        assert instances == constants.DEFAULT_NUM_EXECUTORS
        assert cores == constants.DEFAULT_EXECUTOR_CPU
        assert memory == _memory_kubernetes_to_spark(
            constants.DEFAULT_EXECUTOR_MEMORY,
        )

    def test_simple_parameters(self):
        """Simple executor parameters override defaults."""

        instances, cores, memory = _resolve_executor_resources(
            num_executors=3,
            resources_per_executor={
                "cpu": "2",
                "memory": "4Gi",
            },
        )

        assert instances == 3
        assert cores == 2
        assert memory == "4g"

    def test_executor_precedence(self):
        """Executor configuration takes precedence over simple parameters."""

        executor = Executor(
            num_instances=5,
            resources_per_executor={
                "cpu": "8",
                "memory": "16Gi",
            },
        )

        instances, cores, memory = _resolve_executor_resources(
            executor=executor,
            num_executors=2,
            resources_per_executor={
                "cpu": "4",
                "memory": "8Gi",
            },
        )

        assert instances == 5
        assert cores == 8
        assert memory == "16g"

    def test_executor_fractional_memory(self):
        """Fractional executor memory is converted to MiB."""

        instances, cores, memory = _resolve_executor_resources(
            resources_per_executor={
                "cpu": "2",
                "memory": "1.5Gi",
            },
        )

        assert instances == constants.DEFAULT_NUM_EXECUTORS
        assert cores == 2
        assert memory == "1536m"


class TestReadPodLogs:
    """Tests for read_pod_logs."""

    def test_read_logs(self):
        """Read pod logs without following."""

        core_api = Mock()

        thread = Mock()
        thread.get.return_value = "log line 1\nlog line 2"

        core_api.read_namespaced_pod_log.return_value = thread

        logs = list(
            read_pod_logs(
                core_api=core_api,
                namespace="default",
                pod_name="driver-pod",
            )
        )

        assert logs == [
            "log line 1",
            "log line 2",
        ]

        core_api.read_namespaced_pod_log.assert_called_once_with(
            name="driver-pod",
            namespace="default",
            async_req=True,
        )

    def test_follow_logs(self):
        """Stream pod logs."""

        core_api = Mock()

        stream = Mock()
        stream.stream.return_value = iter(
            [
                b"log line 1\n",
                b"log line 2\n",
            ]
        )

        thread = Mock()
        thread.get.return_value = stream

        core_api.read_namespaced_pod_log.return_value = thread

        logs = list(
            read_pod_logs(
                core_api=core_api,
                namespace="default",
                pod_name="driver-pod",
                follow=True,
            )
        )

        assert logs == [
            "log line 1",
            "log line 2",
        ]

        core_api.read_namespaced_pod_log.assert_called_once_with(
            name="driver-pod",
            namespace="default",
            follow=True,
            _preload_content=False,
            async_req=True,
        )

    def test_timeout(self):
        """Timeout while reading pod logs."""

        core_api = Mock()

        thread = Mock()
        thread.get.side_effect = multiprocessing.TimeoutError()

        core_api.read_namespaced_pod_log.return_value = thread

        with pytest.raises(TimeoutError):
            list(
                read_pod_logs(
                    core_api=core_api,
                    namespace="default",
                    pod_name="driver-pod",
                )
            )

    def test_runtime_error(self):
        """Runtime error while reading pod logs."""

        core_api = Mock()

        thread = Mock()
        thread.get.side_effect = RuntimeError()

        core_api.read_namespaced_pod_log.return_value = thread

        with pytest.raises(RuntimeError):
            list(
                read_pod_logs(
                    core_api=core_api,
                    namespace="default",
                    pod_name="driver-pod",
                )
            )


class TestGetSparkJobDriverSpec:
    """Tests for get_spark_job_driver_spec."""

    def test_defaults(self):
        """Default SparkApplication driver spec."""
        spec = get_spark_job_driver_spec()

        assert spec.cores == constants.DEFAULT_DRIVER_CPU
        assert spec.memory == _memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY)
        assert spec.service_account == constants.DEFAULT_SERVICE_ACCOUNT


class TestGetSparkJobExecutorSpec:
    """Tests for get_spark_job_executor_spec."""

    def test_defaults(self):
        """Default SparkApplication executor spec."""
        spec = get_spark_job_executor_spec()

        assert spec.cores == constants.DEFAULT_EXECUTOR_CPU
        assert spec.memory == _memory_kubernetes_to_spark(constants.DEFAULT_EXECUTOR_MEMORY)
        assert spec.instances == constants.DEFAULT_NUM_EXECUTORS


class TestBuildSparkApplicationCr:
    """Tests for build_spark_application_cr."""

    def test_remote_uri_job(self):
        app = build_spark_application_cr(
            name="test-job",
            namespace="default",
            main_file="s3://bucket/job.py",
            arguments=["--date", "2026-06-30"],
            num_executors=3,
            resources_per_executor={
                "cpu": "2",
                "memory": "4Gi",
            },
        )

        assert app.metadata.name == "test-job"
        assert app.metadata.namespace == "default"

        assert app.spec.main_application_file == "s3://bucket/job.py"
        assert app.spec.arguments == ["--date", "2026-06-30"]

        assert app.spec.driver.cores == 1
        assert app.spec.driver.memory == _memory_kubernetes_to_spark(
            constants.DEFAULT_DRIVER_MEMORY
        )
        assert app.spec.driver.service_account == constants.DEFAULT_SERVICE_ACCOUNT

        assert app.spec.executor.instances == 3
        assert app.spec.executor.cores == 2
        assert app.spec.executor.memory == _memory_kubernetes_to_spark("4Gi")


class TestGetSparkApplicationInfoFromCr:
    """Tests for get_spark_application_info_from_cr."""

    @pytest.fixture
    def minimal_spec(self):
        """Create minimal SparkApplication spec."""
        return models.SparkV1beta2SparkApplicationSpec(
            spark_version=constants.DEFAULT_SPARK_VERSION,
            type="Python",
            mode="cluster",
            image=constants.DEFAULT_SPARK_IMAGE,
            main_application_file="s3://bucket/job.py",
            driver=models.SparkV1beta2DriverSpec(
                cores=1,
                memory="1g",
            ),
            executor=models.SparkV1beta2ExecutorSpec(
                cores=2,
                memory="2g",
                instances=5,
            ),
        )

    @pytest.mark.parametrize(
        "spark_state,expected_status",
        [
            ("SUBMITTED", SparkJobStatus.CREATED),
            ("RUNNING", SparkJobStatus.RUNNING),
            ("SUCCEEDING", SparkJobStatus.RUNNING),
            ("SUSPENDING", SparkJobStatus.RUNNING),
            ("SUSPENDED", SparkJobStatus.RUNNING),
            ("RESUMING", SparkJobStatus.RUNNING),
            ("COMPLETED", SparkJobStatus.COMPLETED),
            ("FAILED", SparkJobStatus.FAILED),
            ("SUBMISSION_FAILED", SparkJobStatus.FAILED),
            ("FAILING", SparkJobStatus.FAILED),
            ("PENDING_RERUN", SparkJobStatus.FAILED),
            ("INVALIDATING", SparkJobStatus.FAILED),
        ],
    )
    def test_status_mapping(
        self,
        minimal_spec,
        spark_state,
        expected_status,
    ):
        creation_timestamp = datetime.now()

        spark_app = models.SparkV1beta2SparkApplication(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="test-job",
                namespace="default",
                creation_timestamp=creation_timestamp,
            ),
            spec=minimal_spec,
            status=models.SparkV1beta2SparkApplicationStatus(
                application_state=models.SparkV1beta2ApplicationState(
                    state=spark_state,
                ),
                driver_info=models.SparkV1beta2DriverInfo(
                    pod_name="test-driver",
                ),
            ),
        )

        job = get_spark_application_info_from_cr(
            spark_app,
        )

        assert job.name == "test-job"
        assert job.namespace == "default"
        assert job.status == expected_status
        assert job.driver_pod_name == "test-driver"
        assert job.creation_timestamp == creation_timestamp
        assert job.num_executors == 5

    def test_without_status(self, minimal_spec):
        creation_timestamp = datetime.now()

        spark_app = models.SparkV1beta2SparkApplication(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="new-job",
                namespace="default",
                creation_timestamp=creation_timestamp,
            ),
            spec=minimal_spec,
        )

        job = get_spark_application_info_from_cr(
            spark_app,
        )

        assert job.name == "new-job"
        assert job.namespace == "default"
        assert job.status == SparkJobStatus.CREATED
        assert job.driver_pod_name is None
        assert job.creation_timestamp == creation_timestamp
        assert job.num_executors == 5

    def test_uses_from_operator_state(self, minimal_spec):
        """Verify SparkApplication status is mapped to SparkJobStatus."""

        creation_timestamp = datetime.now()

        spark_app = models.SparkV1beta2SparkApplication(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="test-job",
                namespace="default",
                creation_timestamp=creation_timestamp,
            ),
            spec=minimal_spec,
            status=models.SparkV1beta2SparkApplicationStatus(
                application_state=models.SparkV1beta2ApplicationState(
                    state="RUNNING",
                ),
                driver_info=models.SparkV1beta2DriverInfo(
                    pod_name="test-driver",
                ),
            ),
        )

        with patch.object(
            SparkJobStatus,
            "from_operator_state",
            return_value=SparkJobStatus.RUNNING,
        ) as mock_from_operator_state:
            job = get_spark_application_info_from_cr(spark_app)

        mock_from_operator_state.assert_called_once_with("RUNNING")

        assert job.name == "test-job"
        assert job.namespace == "default"
        assert job.status == SparkJobStatus.RUNNING
        assert job.driver_pod_name == "test-driver"
        assert job.creation_timestamp == creation_timestamp
        assert job.num_executors == 5

    def test_invalid_metadata(self, minimal_spec):
        """Test invalid SparkApplication CR raises ValueError."""

        spark_app = models.SparkV1beta2SparkApplication.model_construct(
            metadata=None,
            spec=minimal_spec,
        )

        with pytest.raises(
            ValueError,
            match="SparkApplication CR is invalid",
        ):
            get_spark_application_info_from_cr(spark_app)
