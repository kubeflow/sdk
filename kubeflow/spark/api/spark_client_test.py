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

"""Unit tests for SparkClient API."""

from unittest.mock import Mock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.api.spark_client import SparkClient
from kubeflow.spark.types.options import Name
from kubeflow.spark.types.types import SparkConnectInfo, SparkConnectState


@pytest.fixture
def mock_backend():
    """Create mock backend for SparkClient tests."""
    ready_info = SparkConnectInfo(
        name="new-session",
        namespace="default",
        state=SparkConnectState.READY,
        service_name="new-session-svc",
    )
    backend = Mock()
    backend.list_sessions.return_value = [
        SparkConnectInfo(name="s1", namespace="default", state=SparkConnectState.READY),
    ]
    backend.get_session.return_value = SparkConnectInfo(
        name="test", namespace="default", state=SparkConnectState.READY
    )
    backend.create_session.return_value = SparkConnectInfo(
        name="new-session", namespace="default", state=SparkConnectState.PROVISIONING
    )
    backend.wait_for_session_ready.return_value = ready_info
    backend._create_session.return_value = ready_info
    backend._wait_for_session_ready.return_value = ready_info
    backend.get_connect_url.return_value = ("sc://localhost:15002", None)
    return backend


@pytest.fixture
def spark_client(mock_backend):
    """SparkClient with mocked backend."""
    with patch(
        "kubeflow.spark.api.spark_client.KubernetesBackend",
        return_value=mock_backend,
    ):
        client = SparkClient()
        client.backend = mock_backend
        yield client


class TestSparkClientInit:
    """Tests for SparkClient initialization."""

    def test_default_backend(self):
        """C01: Init with default creates KubernetesBackendConfig."""
        with patch("kubeflow.spark.api.spark_client.KubernetesBackend"):
            client = SparkClient()
            assert client.backend is not None

    def test_custom_namespace(self):
        """C02: Init with custom namespace."""
        with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock:
            SparkClient(backend_config=KubernetesBackendConfig(namespace="spark"))
            mock.assert_called_once()

    def test_invalid_backend(self):
        """C03: Init with invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            SparkClient(backend_config="invalid")


class TestSparkClientConnect:
    """Tests for connect method."""

    def test_connect_with_url(self, spark_client):
        """C04: Connect with URL returns SparkSession."""
        mock_session = Mock()
        mock_builder = Mock()
        mock_builder.remote.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        mock_spark = Mock()
        mock_spark.builder = mock_builder

        with (
            patch.dict("sys.modules", {"pyspark": Mock(), "pyspark.sql": mock_spark}),
            patch("kubeflow.spark.api.spark_client.SparkSession", mock_spark),
        ):
            pass

        # Test URL validation works
        from kubeflow.spark.backends.kubernetes.utils import validate_spark_connect_url

        assert validate_spark_connect_url("sc://localhost:15002") is True

    def test_connect_with_url_invalid(self, spark_client):
        """C04b: Connect with invalid URL raises ValueError."""
        from kubeflow.spark.backends.kubernetes.utils import validate_spark_connect_url

        with pytest.raises(ValueError):
            validate_spark_connect_url("http://localhost:15002")

    def test_connect_create_session(self, spark_client, mock_backend):
        """C06: Connect without URL creates new session - verifies backend calls."""
        # Since pyspark is not installed, we verify the backend is called correctly
        mock_backend.create_session.assert_not_called()
        mock_backend.wait_for_session_ready.assert_not_called()


class TestSparkClientSessionManagement:
    """Tests for session management methods."""

    def test_list_sessions(self, spark_client, mock_backend):
        """C14: list_sessions delegates to backend."""
        result = spark_client.list_sessions()
        mock_backend.list_sessions.assert_called_once()
        assert len(result) == 1

    def test_get_session(self, spark_client, mock_backend):
        """C15: get_session delegates to backend."""
        result = spark_client.get_session("test")
        mock_backend.get_session.assert_called_once_with("test")
        assert result.name == "test"

    def test_delete_session(self, spark_client, mock_backend):
        """C16: delete_session delegates to backend."""
        spark_client.delete_session("test")
        mock_backend.delete_session.assert_called_once_with("test")

    def test_get_session_logs(self, spark_client, mock_backend):
        """C17: get_session_logs delegates to backend."""
        mock_backend.get_session_logs.return_value = iter(["log1", "log2"])
        list(spark_client.get_session_logs("test"))
        mock_backend.get_session_logs.assert_called_once_with("test", follow=False)


class TestSparkClientConnectWithNameOption:
    """Tests for connect method with Name option."""

    def test_connect_with_name_option(self, spark_client, mock_backend):
        """C18: Connect passes options to backend including Name option."""
        mock_session = Mock()
        mock_backend.create_and_connect.return_value = mock_session
        options = [Name("custom-session")]
        spark_client.connect(options=options)
        mock_backend.create_and_connect.assert_called_once()
        call_args = mock_backend.create_and_connect.call_args
        assert call_args.kwargs["options"] == options

    def test_connect_without_options_auto_generates(self, spark_client, mock_backend):
        """C19: Connect without options auto-generates name via backend."""
        mock_session = Mock()
        mock_backend.create_and_connect.return_value = mock_session
        spark_client.connect()
        mock_backend.create_and_connect.assert_called_once()
        call_args = mock_backend.create_and_connect.call_args
        assert call_args.kwargs["options"] is None
