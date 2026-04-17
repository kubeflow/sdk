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

"""Unit tests for Kubeflow Spark types."""

from datetime import datetime
from unittest.mock import patch

import pytest

from kubeflow.spark.types.types import (
    Driver,
    Executor,
    FileJob,
    FuncJob,
    SparkConnectInfo,
    SparkConnectState,
    SparkJob,
    SparkJobStatus,
)


class TestSparkConnectState:
    """Tests for SparkConnectState enum."""

    def test_enum_values(self):
        """T01: Verify SparkConnectState enum has expected values."""
        assert SparkConnectState.PROVISIONING == "Provisioning"
        assert SparkConnectState.READY == "Ready"
        assert SparkConnectState.RUNNING == "Running"
        assert SparkConnectState.NOT_READY == "NotReady"
        assert SparkConnectState.FAILED == "Failed"

    def test_enum_is_string(self):
        """Verify SparkConnectState inherits from str."""
        assert isinstance(SparkConnectState.READY, str)
        assert SparkConnectState.READY == "Ready"


class TestSparkConnectInfo:
    """Tests for SparkConnectInfo dataclass."""

    def test_defaults(self):
        """T02: SparkConnectInfo with only required fields has None for optional."""
        info = SparkConnectInfo(
            name="test-session",
            namespace="default",
            state=SparkConnectState.READY,
        )
        assert info.name == "test-session"
        assert info.namespace == "default"
        assert info.state == SparkConnectState.READY
        assert info.driver_pod_name is None
        assert info.pod_ip is None
        assert info.service_name is None
        assert info.creation_timestamp is None

    def test_all_fields(self):
        """T03: SparkConnectInfo with all fields set."""
        created = datetime(2025, 1, 12, 10, 30, 0)
        info = SparkConnectInfo(
            name="full-session",
            namespace="spark-ns",
            state=SparkConnectState.READY,
            driver_pod_name="spark-connect-server-0",
            pod_ip="10.0.0.5",
            service_name="spark-connect-svc",
            creation_timestamp=created,
        )
        assert info.name == "full-session"
        assert info.namespace == "spark-ns"
        assert info.state == SparkConnectState.READY
        assert info.driver_pod_name == "spark-connect-server-0"
        assert info.pod_ip == "10.0.0.5"
        assert info.service_name == "spark-connect-svc"
        assert info.creation_timestamp == created


class TestDriver:
    """Tests for Driver dataclass (KEP-107 compliant)."""

    def test_defaults(self):
        """T04: Driver with no arguments has all fields None."""
        driver = Driver()
        assert driver.image is None
        assert driver.resources is None
        assert driver.java_options is None
        assert driver.service_account is None

    def test_with_resources(self):
        """T06: Driver with resources dict (KEP-107 pattern)."""
        driver = Driver(
            resources={"cpu": "2", "memory": "4Gi"},
        )
        assert driver.resources == {"cpu": "2", "memory": "4Gi"}

    def test_with_gpu_resources(self):
        """Driver with GPU resources (KEP-107 pattern)."""
        driver = Driver(
            resources={"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
        )
        assert driver.resources["cpu"] == "4"
        assert driver.resources["memory"] == "8Gi"
        assert driver.resources["nvidia.com/gpu"] == "1"

    def test_with_service_account(self):
        """Driver with service account."""
        driver = Driver(service_account="spark-sa")
        assert driver.service_account == "spark-sa"

    def test_kep107_example(self):
        """Test KEP-107 example from lines 165-170."""
        driver = Driver(
            resources={"cpu": "4", "memory": "8Gi"},
            service_account="spark-driver-prod",
        )
        assert driver.resources["cpu"] == "4"
        assert driver.resources["memory"] == "8Gi"
        assert driver.service_account == "spark-driver-prod"


class TestExecutor:
    """Tests for Executor dataclass (KEP-107 compliant)."""

    def test_defaults(self):
        """T05: Executor with no arguments has all fields None."""
        executor = Executor()
        assert executor.num_instances is None
        assert executor.resources_per_executor is None
        assert executor.java_options is None

    def test_with_num_instances(self):
        """T07: Executor with num_instances set."""
        executor = Executor(num_instances=5)
        assert executor.num_instances == 5

    def test_with_resources_per_executor(self):
        """Executor with resources_per_executor dict (KEP-107 pattern)."""
        executor = Executor(
            num_instances=3,
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
        )
        assert executor.num_instances == 3
        assert executor.resources_per_executor == {"cpu": "4", "memory": "8Gi"}

    def test_with_gpu_resources(self):
        """Executor with GPU resources (KEP-107 pattern)."""
        executor = Executor(
            num_instances=10,
            resources_per_executor={
                "cpu": "8",
                "memory": "32Gi",
                "nvidia.com/gpu": "2",
            },
        )
        assert executor.num_instances == 10
        assert executor.resources_per_executor["cpu"] == "8"
        assert executor.resources_per_executor["memory"] == "32Gi"
        assert executor.resources_per_executor["nvidia.com/gpu"] == "2"

    def test_kep107_example(self):
        """Test KEP-107 example from lines 172-177."""
        executor = Executor(
            num_instances=20,
            resources_per_executor={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "2"},
        )
        assert executor.num_instances == 20
        assert executor.resources_per_executor["cpu"] == "8"
        assert executor.resources_per_executor["memory"] == "32Gi"
        assert executor.resources_per_executor["nvidia.com/gpu"] == "2"


class TestSparkJobStatus:
    """Tests for SparkJobStatus enum."""

    def test_enum_values(self):
        """Verify SparkJobStatus enum has expected values."""
        assert SparkJobStatus.CREATED == "Created"
        assert SparkJobStatus.RUNNING == "Running"
        assert SparkJobStatus.COMPLETED == "Completed"
        assert SparkJobStatus.FAILED == "Failed"

    def test_enum_is_string(self):
        """Verify SparkJobStatus inherits from str."""
        assert isinstance(SparkJobStatus.RUNNING, str)
        assert SparkJobStatus.RUNNING == "Running"

    @pytest.mark.parametrize(
        "operator_state,expected_status",
        [
            (None, SparkJobStatus.CREATED),
            ("", SparkJobStatus.CREATED),
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
            ("UNKNOWN", SparkJobStatus.FAILED),
        ],
    )
    def test_from_operator_state(
        self,
        operator_state,
        expected_status,
    ):
        """Verify SparkApplication states map to SparkJobStatus."""
        assert SparkJobStatus.from_operator_state(operator_state) == expected_status

    def test_unknown_operator_state(self):
        """Verify unknown SparkApplication states default to FAILED."""
        with patch("kubeflow.spark.types.types.logger") as mock_logger:
            status = SparkJobStatus.from_operator_state("SOME_NEW_STATE")

        assert status == SparkJobStatus.FAILED
        mock_logger.warning.assert_called_once_with(
            "Unknown SparkApplication state '%s'. Defaulting to FAILED.",
            "SOME_NEW_STATE",
        )


class TestSparkJob:
    """Tests for SparkJob dataclass."""

    def test_defaults(self):
        """SparkJob with only required fields."""

        job = SparkJob(
            name="test-job",
            namespace="default",
        )

        assert job.name == "test-job"
        assert job.namespace == "default"
        assert job.status is None
        assert job.creation_timestamp is None
        assert job.num_executors is None
        assert job.driver_pod_name is None

    def test_all_fields(self):
        """SparkJob with all fields set."""

        created = datetime(2025, 1, 12, 10, 30, 0)

        job = SparkJob(
            name="full-job",
            namespace="spark-ns",
            status=SparkJobStatus.RUNNING,
            creation_timestamp=created,
            num_executors=10,
            driver_pod_name="driver-pod-1",
        )

        assert job.name == "full-job"
        assert job.namespace == "spark-ns"
        assert job.status == SparkJobStatus.RUNNING
        assert job.creation_timestamp == created
        assert job.num_executors == 10
        assert job.driver_pod_name == "driver-pod-1"


class TestFileJob:
    """Tests for FileJob dataclass."""

    def test_defaults(self):
        """FileJob with required fields."""

        job = FileJob(
            file_source="s3://bucket/job.py",
        )

        assert job.file_source == "s3://bucket/job.py"
        assert job.args is None
        assert job.main_class is None

    def test_all_fields(self):
        """FileJob with all fields."""

        job = FileJob(
            file_source="local:///opt/spark/app.py",
            args=["--date", "2026-06-30"],
            main_class="org.apache.spark.Main",
        )

        assert job.file_source == "local:///opt/spark/app.py"
        assert job.args == ["--date", "2026-06-30"]
        assert job.main_class == "org.apache.spark.Main"


class TestFuncJob:
    """Tests for FuncJob dataclass."""

    def test_defaults(self):
        """FuncJob with only required fields."""

        def sample():
            return "ok"

        job = FuncJob(
            func=sample,
        )

        assert job.func is sample
        assert job.func_args is None

    def test_all_fields(self):
        """FuncJob with function arguments."""

        def sample(x, y):
            return x + y

        job = FuncJob(
            func=sample,
            func_args={
                "x": 1,
                "y": 2,
            },
        )

        assert job.func is sample
        assert job.func_args == {
            "x": 1,
            "y": 2,
        }
