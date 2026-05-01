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

from kubeflow.spark.types.types import (
    Driver,
    Executor,
    SparkConnectInfo,
    SparkConnectState,
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
        assert info.pod_name is None
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
            pod_name="spark-connect-server-0",
            pod_ip="10.0.0.5",
            service_name="spark-connect-svc",
            creation_timestamp=created,
        )
        assert info.name == "full-session"
        assert info.namespace == "spark-ns"
        assert info.state == SparkConnectState.READY
        assert info.pod_name == "spark-connect-server-0"
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
