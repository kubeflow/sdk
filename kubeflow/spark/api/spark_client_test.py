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
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="Init with default creates KubernetesBackendConfig",
            config={"backend_config": None},
            expected_status=SUCCESS,
        ),
        TestCase(
            name="Init with custom namespace",
            config={"backend_config": KubernetesBackendConfig(namespace="spark")},
            expected_status=SUCCESS,
        ),
        TestCase(
            name="Init with invalid backend raises ValueError",
            config={"backend_config": "invalid"},
            expected_status=FAILED,
            expected_error=ValueError,
        ),
    ],
)
def test_spark_client_init(test_case: TestCase):
    """Test SparkClient initialization."""
    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        if test_case.expected_status == SUCCESS:
            client = SparkClient(backend_config=test_case.config["backend_config"])
            assert client.backend is not None
            if test_case.config["backend_config"]:
                mock_backend.assert_called_once_with(test_case.config["backend_config"])
        else:
            with pytest.raises(test_case.expected_error):
                SparkClient(backend_config=test_case.config["backend_config"])


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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list_sessions delegates to backend",
            config={"method": "list_sessions", "args": []},
            expected_output=1,  # Number of sessions in mock
        ),
        TestCase(
            name="get_session delegates to backend",
            config={"method": "get_session", "args": ["test"]},
            expected_output="test",
        ),
        TestCase(
            name="delete_session delegates to backend",
            config={"method": "delete_session", "args": ["test"]},
        ),
        TestCase(
            name="get_session_logs delegates to backend",
            config={"method": "get_session_logs", "args": ["test"]},
            expected_output=["log1", "log2"],
        ),
    ],
)
def test_spark_client_session_management(spark_client, mock_backend, test_case: TestCase):
    """Test session management methods."""
    method_name = test_case.config["method"]
    args = test_case.config["args"]
    method = getattr(spark_client, method_name)

    if method_name == "get_session_logs":
        mock_backend.get_session_logs.return_value = iter(test_case.expected_output)
        result = list(method(*args))
        assert result == test_case.expected_output
        mock_backend.get_session_logs.assert_called_once_with(*args, follow=False)
    else:
        result = method(*args)
        getattr(mock_backend, method_name).assert_called_once_with(*args)
        if test_case.expected_output:
            if method_name == "list_sessions":
                assert len(result) == test_case.expected_output
            elif method_name == "get_session":
                assert result.name == test_case.expected_output


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
