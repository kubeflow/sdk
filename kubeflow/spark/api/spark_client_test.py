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

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.api.spark_client import SparkClient
from kubeflow.spark.types.options import Name
from kubeflow.spark.types.types import SparkConnectInfo, SparkConnectState


@dataclass
class TestCase:
    """Test case structure for parametrized SparkClient tests."""

    name: str
    expected_status: str
    config: dict[str, Any]
    expected_output: Optional[Any] = None
    expected_error: Optional[str] = None
    # Prevent pytest from collecting this dataclass as a test
    __test__ = False


SUCCESS = "SUCCESS"
EXCEPTION = "EXCEPTION"


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

    # Configure mock to handle both existing and non-existent sessions
    def mock_get_session(session_name):
        if session_name == "nonexistent":
            raise ValueError("Session not found")
        return SparkConnectInfo(
            name=session_name, namespace="default", state=SparkConnectState.READY
        )

    backend.get_session.side_effect = mock_get_session
    backend.create_session.return_value = SparkConnectInfo(
        name="new-session", namespace="default", state=SparkConnectState.PROVISIONING
    )
    backend.wait_for_session_ready.return_value = ready_info
    backend._create_session.return_value = ready_info
    backend._wait_for_session_ready.return_value = ready_info
    backend.get_connect_url.return_value = ("sc://localhost:15002", None)
    backend.get_session_logs.return_value = iter(["log1", "log2"])
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
            name="default backend initialization",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="custom namespace initialization",
            expected_status=SUCCESS,
            config={"namespace": "spark"},
        ),
        TestCase(
            name="invalid backend config",
            expected_status=EXCEPTION,
            config={"backend_config": "invalid"},
            expected_error="ValueError",
        ),
    ],
)
def test_spark_client_initialization(test_case: TestCase):
    """Test SparkClient initialization scenarios."""

    try:
        if "namespace" in test_case.config:
            with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock:
                SparkClient(
                    backend_config=KubernetesBackendConfig(namespace=test_case.config["namespace"])
                )
                mock.assert_called_once()
        elif "backend_config" in test_case.config:
            SparkClient(backend_config=test_case.config["backend_config"])
        else:
            with patch("kubeflow.spark.api.spark_client.KubernetesBackend"):
                client = SparkClient()
                assert client.backend is not None

        # If we reach here but expected an exception, fail
        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )
    except Exception as e:
        # If we got an exception but expected success, fail
        assert test_case.expected_status == EXCEPTION, (
            f"Unexpected exception in {test_case.name}: {e}"
        )
        # Validate the exception type/message if specified
        if test_case.expected_error:
            assert test_case.expected_error in str(e), (
                f"Expected error '{test_case.expected_error}' but got '{str(e)}'"
            )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="connect with valid URL validation",
            expected_status=SUCCESS,
            config={"url": "sc://localhost:15002"},
            expected_output=True,
        ),
        TestCase(
            name="connect with invalid URL validation",
            expected_status=EXCEPTION,
            config={"url": "http://localhost:15002"},
            expected_error="ValueError",
        ),
        TestCase(
            name="connect create session verification",
            expected_status=SUCCESS,
            config={},
        ),
    ],
)
def test_spark_client_connect(test_case: TestCase, spark_client):
    """Test SparkClient connect method scenarios."""

    try:
        if "url" in test_case.config:
            from kubeflow.spark.backends.kubernetes.utils import validate_spark_connect_url

            result = validate_spark_connect_url(test_case.config["url"])
            assert result == test_case.expected_output
        else:
            # Verify backend methods are not called initially
            spark_client.backend.create_session.assert_not_called()
            spark_client.backend.wait_for_session_ready.assert_not_called()

        # If we reach here but expected an exception, fail
        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )
    except Exception as e:
        # If we got an exception but expected success, fail
        assert test_case.expected_status == EXCEPTION, (
            f"Unexpected exception in {test_case.name}: {e}"
        )
        # Validate the exception type/message if specified
        if test_case.expected_error:
            assert test_case.expected_error in str(e), (
                f"Expected error '{test_case.expected_error}' but got '{str(e)}'"
            )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list sessions successfully", expected_status=SUCCESS, config={}, expected_output=1
        ),
        TestCase(
            name="get existing session",
            expected_status=SUCCESS,
            config={"session_name": "test"},
            expected_output="test",
        ),
        TestCase(
            name="get non-existent session",
            expected_status=EXCEPTION,
            config={"session_name": "nonexistent"},
            expected_error="Session not found",
        ),
        TestCase(
            name="delete session",
            expected_status=SUCCESS,
            config={"session_name": "test", "operation": "delete"},
        ),
        TestCase(
            name="get session logs",
            expected_status=SUCCESS,
            config={"session_name": "test", "operation": "logs"},
            expected_output=2,  # Expected number of log entries
        ),
    ],
)
def test_spark_client_session_management(test_case: TestCase, spark_client, mock_backend):
    """Test SparkClient session management operations."""

    try:
        if "operation" in test_case.config:
            if test_case.config["operation"] == "delete":
                spark_client.delete_session(test_case.config["session_name"])
                mock_backend.delete_session.assert_called_once_with(
                    test_case.config["session_name"]
                )
            elif test_case.config["operation"] == "logs":
                result = list(spark_client.get_session_logs(test_case.config["session_name"]))
                assert len(result) == test_case.expected_output
                mock_backend.get_session_logs.assert_called_once_with(
                    test_case.config["session_name"], follow=False
                )
        elif "session_name" in test_case.config:
            result = spark_client.get_session(test_case.config["session_name"])
            assert result.name == test_case.expected_output
            mock_backend.get_session.assert_called_with(test_case.config["session_name"])
        else:
            result = spark_client.list_sessions()
            assert len(result) == test_case.expected_output
            mock_backend.list_sessions.assert_called_once()

        # If we reach here but expected an exception, fail
        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )
    except Exception as e:
        # If we got an exception but expected success, fail
        assert test_case.expected_status == EXCEPTION, (
            f"Unexpected exception in {test_case.name}: {e}"
        )
        # Validate the exception type/message if specified
        if test_case.expected_error:
            assert test_case.expected_error in str(e), (
                f"Expected error '{test_case.expected_error}' but got '{str(e)}'"
            )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="connect with name option",
            expected_status=SUCCESS,
            config={"options": [Name("custom-session")]},
        ),
        TestCase(
            name="connect without options auto-generates",
            expected_status=SUCCESS,
            config={},
        ),
    ],
)
def test_spark_client_connect_with_options(test_case: TestCase, spark_client, mock_backend):
    """Test SparkClient connect method with Name option scenarios."""

    mock_session = Mock()
    mock_backend.create_and_connect.return_value = mock_session

    try:
        if "options" in test_case.config:
            options = test_case.config["options"]
            spark_client.connect(options=options)
            mock_backend.create_and_connect.assert_called_once()
            call_args = mock_backend.create_and_connect.call_args
            assert call_args.kwargs["options"] == options
        else:
            spark_client.connect()
            mock_backend.create_and_connect.assert_called_once()
            call_args = mock_backend.create_and_connect.call_args
            assert call_args.kwargs["options"] is None

        # If we reach here but expected an exception, fail
        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )
    except Exception as e:
        # If we got an exception but expected success, fail
        assert test_case.expected_status == EXCEPTION, (
            f"Unexpected exception in {test_case.name}: {e}"
        )
        # Validate the exception type/message if specified
        if test_case.expected_error:
            assert test_case.expected_error in str(e), (
                f"Expected error '{test_case.expected_error}' but got '{str(e)}'"
            )
