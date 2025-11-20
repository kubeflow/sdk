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

"""Unit tests for Spark Connect backend and configuration."""

import pytest

from kubeflow.spark.models import ConnectBackendConfig, SessionInfo, SessionMetrics


def _is_pyspark_available() -> bool:
    """Check if PySpark Connect is available."""
    try:
        import pyspark  # noqa: F401

        return True
    except ImportError:
        return False


class TestConnectBackendConfig:
    """Tests for ConnectBackendConfig validation and URL building."""

    def test_valid_basic_config(self):
        """Test creating a basic valid configuration."""
        config = ConnectBackendConfig(connect_url="sc://localhost:15002")

        assert config.connect_url == "sc://localhost:15002"
        assert config.use_ssl is True
        assert config.token is None
        assert config.timeout == 300

    def test_valid_config_with_authentication(self):
        """Test configuration with authentication token."""
        config = ConnectBackendConfig(
            connect_url="sc://cluster:15002",
            token="test-token",
            use_ssl=True,
        )

        assert config.token == "test-token"
        assert config.use_ssl is True

    def test_valid_config_with_all_options(self):
        """Test configuration with all optional parameters."""
        config = ConnectBackendConfig(
            connect_url="sc://cluster:15002",
            token="test-token",
            use_ssl=True,
            user_id="testuser",
            session_id="test-session-123",
            grpc_max_message_size=256 * 1024 * 1024,
            enable_auto_provision=False,
            namespace="spark-jobs",
            enable_monitoring=True,
            timeout=600,
        )

        assert config.connect_url == "sc://cluster:15002"
        assert config.token == "test-token"
        assert config.user_id == "testuser"
        assert config.session_id == "test-session-123"
        assert config.grpc_max_message_size == 256 * 1024 * 1024
        assert config.namespace == "spark-jobs"
        assert config.timeout == 600

    def test_url_with_parameters(self):
        """Test URL with embedded parameters."""
        config = ConnectBackendConfig(connect_url="sc://cluster:15002/;use_ssl=true;token=abc123")

        assert config.connect_url == "sc://cluster:15002/;use_ssl=true;token=abc123"

    def test_kubernetes_service_url(self):
        """Test Kubernetes service DNS format."""
        config = ConnectBackendConfig(
            connect_url="sc://spark-connect.spark-ns.svc.cluster.local:15002"
        )

        assert config.connect_url == "sc://spark-connect.spark-ns.svc.cluster.local:15002"


class TestConnectBackendValidation:
    """Tests for ConnectBackend URL validation."""

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_invalid_url_missing_scheme(self):
        """Test that invalid URL (missing sc://) is rejected."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="localhost:15002")

        with pytest.raises(ValueError, match="Invalid Spark Connect URL"):
            ConnectBackend(config)

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_invalid_url_missing_port(self):
        """Test that URL without port is rejected."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost")

        with pytest.raises(ValueError, match="Invalid Spark Connect URL"):
            ConnectBackend(config)

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_empty_url(self):
        """Test that empty URL is rejected."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="")

        with pytest.raises(ValueError, match="connect_url is required"):
            ConnectBackend(config)


class TestConnectBackendURLBuilding:
    """Tests for connection URL building logic."""

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_url_building_with_ssl(self):
        """Test URL building with SSL enabled."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002", use_ssl=True)
        backend = ConnectBackend(config)

        url = backend._build_connection_url()
        assert "use_ssl=true" in url

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_url_building_with_token(self):
        """Test URL building with authentication token."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002", token="test-token")
        backend = ConnectBackend(config)

        url = backend._build_connection_url()
        assert "token=test-token" in url

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_url_building_preserves_existing_params(self):
        """Test that existing URL parameters are preserved."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002/;custom_param=value")
        backend = ConnectBackend(config)

        url = backend._build_connection_url()
        assert "custom_param=value" in url

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_url_building_config_overrides_url_params(self):
        """Test that config parameters override URL parameters."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(
            connect_url="sc://localhost:15002/;token=url-token", token="config-token"
        )
        backend = ConnectBackend(config)

        url = backend._build_connection_url()
        assert "token=config-token" in url
        assert "token=url-token" not in url


class TestSessionMetrics:
    """Tests for SessionMetrics model."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = SessionMetrics(session_id="test-123")

        assert metrics.session_id == "test-123"
        assert metrics.queries_executed == 0
        assert metrics.active_queries == 0
        assert metrics.artifacts_uploaded == 0
        assert metrics.data_read_bytes == 0
        assert metrics.data_written_bytes == 0
        assert metrics.execution_time_ms == 0

    def test_metrics_with_values(self):
        """Test metrics with custom values."""
        metrics = SessionMetrics(
            session_id="test-123",
            queries_executed=10,
            active_queries=2,
            artifacts_uploaded=5,
            data_read_bytes=1024 * 1024,
            data_written_bytes=512 * 1024,
            execution_time_ms=5000,
        )

        assert metrics.queries_executed == 10
        assert metrics.active_queries == 2
        assert metrics.artifacts_uploaded == 5
        assert metrics.data_read_bytes == 1024 * 1024
        assert metrics.data_written_bytes == 512 * 1024
        assert metrics.execution_time_ms == 5000


class TestSessionInfo:
    """Tests for SessionInfo model."""

    def test_basic_session_info(self):
        """Test basic session info creation."""
        info = SessionInfo(session_id="test-123", app_name="test-app")

        assert info.session_id == "test-123"
        assert info.app_name == "test-app"
        assert info.state == "active"
        assert info.user_id is None
        assert info.metrics is None

    def test_session_info_with_metrics(self):
        """Test session info with metrics."""
        metrics = SessionMetrics(session_id="test-123", queries_executed=5)
        info = SessionInfo(
            session_id="test-123", app_name="test-app", state="active", metrics=metrics
        )

        assert info.metrics is not None
        assert info.metrics.queries_executed == 5

    def test_session_info_with_all_fields(self):
        """Test session info with all fields populated."""
        metrics = SessionMetrics(session_id="test-123")
        info = SessionInfo(
            session_id="test-123",
            app_name="test-app",
            user_id="testuser",
            created_at="2024-01-01T00:00:00",
            last_activity="2024-01-01T01:00:00",
            state="active",
            metrics=metrics,
        )

        assert info.session_id == "test-123"
        assert info.app_name == "test-app"
        assert info.user_id == "testuser"
        assert info.created_at == "2024-01-01T00:00:00"
        assert info.last_activity == "2024-01-01T01:00:00"
        assert info.state == "active"
        assert info.metrics == metrics


class TestConnectBackendBatchMethodsRaiseErrors:
    """Test that batch-oriented methods raise NotImplementedError."""

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_submit_application_raises_error(self):
        """Test that submit_application raises NotImplementedError."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002")
        backend = ConnectBackend(config)

        with pytest.raises(NotImplementedError, match="batch application submission"):
            backend.submit_application(
                app_name="test",
                main_application_file="test.py",
                spark_version="3.5.0",
                app_type="Python",
                driver_cores=1,
                driver_memory="1g",
                executor_cores=1,
                executor_memory="1g",
                num_executors=1,
                queue=None,
                arguments=None,
                python_version="3",
                spark_conf=None,
                hadoop_conf=None,
                env_vars=None,
                deps=None,
            )

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_get_status_raises_error(self):
        """Test that get_status raises NotImplementedError."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002")
        backend = ConnectBackend(config)

        with pytest.raises(NotImplementedError, match="batch application status"):
            backend.get_status("test-id")

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_delete_application_raises_error(self):
        """Test that delete_application raises NotImplementedError."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002")
        backend = ConnectBackend(config)

        with pytest.raises(NotImplementedError, match="batch application deletion"):
            backend.delete_application("test-id")

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_get_logs_raises_error(self):
        """Test that get_logs raises NotImplementedError."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002")
        backend = ConnectBackend(config)

        with pytest.raises(NotImplementedError, match="logs retrieval"):
            list(backend.get_logs("test-id"))

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_list_applications_raises_error(self):
        """Test that list_applications raises NotImplementedError."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002")
        backend = ConnectBackend(config)

        with pytest.raises(NotImplementedError, match="listing applications"):
            backend.list_applications()

    @pytest.mark.skipif(not _is_pyspark_available(), reason="PySpark Connect not installed")
    def test_wait_for_completion_raises_error(self):
        """Test that wait_for_completion raises NotImplementedError."""
        from kubeflow.spark.backends.connect import ConnectBackend

        config = ConnectBackendConfig(connect_url="sc://localhost:15002")
        backend = ConnectBackend(config)

        with pytest.raises(NotImplementedError, match="application completion"):
            backend.wait_for_completion("test-id")
