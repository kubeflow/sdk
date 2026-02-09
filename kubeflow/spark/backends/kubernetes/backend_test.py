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

"""Unit tests for KubernetesBackend."""

from dataclasses import dataclass
import multiprocessing
from typing import Optional
from unittest.mock import Mock, patch

from kubernetes.client import ApiException
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend
from kubeflow.spark.test.common import (
    DEFAULT_NAMESPACE,
    SPARK_CONNECT_FAILED,
    SPARK_CONNECT_PROVISIONING,
    SPARK_CONNECT_READY,
)
from kubeflow.spark.types.options import Labels, Name
from kubeflow.spark.types.types import SparkConnectState


@dataclass
class Case:
    """Test case parameter container."""

    name: str
    expected_state: Optional[SparkConnectState] = None
    should_raise: bool = False
    error_match: Optional[str] = None
    num_executors: Optional[int] = None
    expected_name_prefix: Optional[str] = None
    session_name: Optional[str] = None
    error_type: Optional[str] = None


def create_mock_thread_with_error(response=None, raise_timeout=False, raise_error=False):
    """Create mock thread that simulates async K8s API response or errors."""
    mock_thread = Mock()

    if raise_timeout:
        mock_thread.get.side_effect = multiprocessing.TimeoutError()
    elif raise_error:
        mock_thread.get.side_effect = RuntimeError("Simulated K8s API error")
    else:
        mock_thread.get.return_value = response

    return mock_thread


def mock_get_response(name: str) -> dict:
    """Return mock CRD response based on session name."""
    if name == SPARK_CONNECT_READY:
        return {
            "metadata": {"name": name, "namespace": DEFAULT_NAMESPACE},
            "status": {
                "state": "Ready",
                "server": {"podName": f"{name}-0", "podIp": "10.0.0.5"},
            },
        }
    elif name == SPARK_CONNECT_PROVISIONING:
        return {
            "metadata": {"name": name, "namespace": DEFAULT_NAMESPACE},
            "status": {"state": "Provisioning"},
        }
    elif name == SPARK_CONNECT_FAILED:
        return {
            "metadata": {"name": name, "namespace": DEFAULT_NAMESPACE},
            "status": {"state": "Failed"},
        }
    raise ApiException(status=404, reason="Not Found")


def mock_list_response(*args, **kwargs) -> dict:
    """Return mock list response."""
    return {
        "items": [
            {
                "metadata": {"name": "session-1", "namespace": DEFAULT_NAMESPACE},
                "status": {"state": "Ready"},
            },
            {
                "metadata": {"name": "session-2", "namespace": DEFAULT_NAMESPACE},
                "status": {"state": "Provisioning"},
            },
        ]
    }


def mock_create_response(*args, **kwargs) -> dict:
    """Return mock create response."""
    body = kwargs.get("body", {})
    return {
        "metadata": body.get("metadata", {}),
        "status": {"state": "Provisioning"},
    }


def mock_delete_response(name: str) -> None:
    """Mock delete - raise 404 for unknown sessions."""
    if name.startswith("unknown"):
        raise ApiException(status=404, reason="Not Found")
    return None


def _mock_create(**kw):
    """Mock create that returns thread with response."""
    response = mock_create_response(**kw)
    return create_mock_thread_with_error(response=response)


def _mock_get(**kw):
    """Mock get that returns thread, handling 404 in thread.get()."""
    mock_thread = Mock()

    def get_with_exception(timeout=None):
        # This simulates calling thread.get() which raises on error
        return mock_get_response(kw["name"])

    mock_thread.get = Mock(side_effect=get_with_exception)
    return mock_thread


def _mock_delete(**kw):
    """Mock delete that returns thread, handling 404 in thread.get()."""
    mock_thread = Mock()

    def get_with_exception(timeout=None):
        # This simulates calling thread.get() which raises on error
        mock_delete_response(kw["name"])
        return None

    mock_thread.get = Mock(side_effect=get_with_exception)
    return mock_thread


def _mock_list(**kw):
    """Mock list that returns thread with response."""
    response = mock_list_response(**kw)
    return create_mock_thread_with_error(response=response)


def _mock_read_logs(**kw):
    """Mock read_namespaced_pod_log that returns thread with logs."""
    logs = "log line 1\nlog line 2"
    return create_mock_thread_with_error(response=logs)


@pytest.fixture
def spark_backend():
    """Provide KubernetesBackend with mocked K8s APIs."""
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(side_effect=_mock_create),
                get_namespaced_custom_object=Mock(side_effect=_mock_get),
                list_namespaced_custom_object=Mock(side_effect=_mock_list),
                delete_namespaced_custom_object=Mock(side_effect=_mock_delete),
            ),
        ),
        patch(
            "kubernetes.client.CoreV1Api",
            return_value=Mock(
                read_namespaced_pod_log=Mock(side_effect=_mock_read_logs),
            ),
        ),
    ):
        yield KubernetesBackend(KubernetesBackendConfig())


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="with_name_option_and_executors",
            num_executors=3,
            expected_name_prefix="test-session",
            expected_state=SparkConnectState.PROVISIONING,
            session_name="test-session",
        ),
        Case(
            name="auto_generated_name",
            num_executors=None,
            expected_name_prefix="spark-connect-",
            expected_state=SparkConnectState.PROVISIONING,
            session_name=None,
        ),
    ],
)
def test_create_session(spark_backend, test_case):
    """Test session creation with different parameter combinations."""
    options = [Name(test_case.session_name)] if test_case.session_name else None
    info = spark_backend._create_session(num_executors=test_case.num_executors, options=options)
    assert info.name.startswith(test_case.expected_name_prefix)
    assert info.state == test_case.expected_state


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="existing_session",
            session_name=SPARK_CONNECT_READY,
            expected_state=SparkConnectState.READY,
            should_raise=False,
        ),
        Case(
            name="session_not_found",
            session_name="unknown-session",
            should_raise=True,
            error_match="not found",
        ),
    ],
)
def test_get_session(spark_backend, test_case):
    """Test getting session with existing and non-existent names."""
    if test_case.should_raise:
        with pytest.raises(RuntimeError, match=test_case.error_match):
            spark_backend.get_session(test_case.session_name)
    else:
        info = spark_backend.get_session(test_case.session_name)
        assert info.name == test_case.session_name
        assert info.state == test_case.expected_state


def test_list_sessions(spark_backend):
    """Test listing multiple sessions."""
    sessions = spark_backend.list_sessions()
    assert len(sessions) == 2
    assert sessions[0].name == "session-1"
    assert sessions[1].name == "session-2"


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="delete_existing",
            session_name=SPARK_CONNECT_READY,
            should_raise=False,
        ),
        Case(
            name="delete_not_found",
            session_name="unknown-session",
            should_raise=True,
            error_match="not found",
        ),
    ],
)
def test_delete_session(spark_backend, test_case):
    """Test deleting existing and non-existent sessions."""
    if test_case.should_raise:
        with pytest.raises(RuntimeError, match=test_case.error_match):
            spark_backend.delete_session(test_case.session_name)
    else:
        spark_backend.delete_session(test_case.session_name)


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="already_ready",
            session_name=SPARK_CONNECT_READY,
            expected_state=SparkConnectState.READY,
            should_raise=False,
        ),
        Case(
            name="session_failed",
            session_name=SPARK_CONNECT_FAILED,
            should_raise=True,
            error_match="failed",
        ),
    ],
)
def test_wait_for_session_ready(spark_backend, test_case):
    """Test waiting for session with different states."""
    if test_case.should_raise:
        with pytest.raises(RuntimeError, match=test_case.error_match):
            spark_backend._wait_for_session_ready(test_case.session_name, timeout=5)
    else:
        info = spark_backend._wait_for_session_ready(test_case.session_name, timeout=5)
        assert info.state == test_case.expected_state


def test_get_session_logs(spark_backend):
    """Test retrieving session logs."""
    logs = list(spark_backend.get_session_logs(SPARK_CONNECT_READY))
    assert len(logs) == 2
    assert logs[0] == "log line 1"


def test_get_connect_url_in_cluster(spark_backend):
    """When KUBERNETES_SERVICE_HOST is set, get_connect_url returns in-cluster URL and no process."""
    from kubeflow.spark.types.types import SparkConnectInfo, SparkConnectState

    info = SparkConnectInfo(
        name="test-session",
        namespace="default",
        state=SparkConnectState.READY,
        service_name="test-session-svc",
    )
    with patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.96.0.1"}, clear=False):
        url, proc = spark_backend.get_connect_url(info)
    assert "svc.cluster.local" in url
    assert proc is None


def test_get_connect_url_port_forward(spark_backend):
    """When not in cluster, get_connect_url starts port-forward and returns localhost URL."""
    from kubeflow.spark.types.types import SparkConnectInfo, SparkConnectState

    info = SparkConnectInfo(
        name="test-session",
        namespace="default",
        state=SparkConnectState.READY,
        service_name="test-session-svc",
    )
    mock_popen = Mock()
    mock_popen.poll.return_value = None
    with (
        patch.dict(
            "os.environ",
            {"KUBERNETES_SERVICE_HOST": "", "SPARK_CONNECT_LOCAL_PORT": "15002"},
            clear=False,
        ),
        patch(
            "kubeflow.spark.backends.kubernetes.backend.subprocess.Popen", return_value=mock_popen
        ),
        patch("kubeflow.spark.backends.kubernetes.backend.time.sleep"),
        patch.object(spark_backend, "_wait_for_connect_port", return_value=True),
    ):
        url, proc = spark_backend.get_connect_url(info)
    assert url == "sc://127.0.0.1:15002"  # Uses 127.0.0.1 to force IPv4 for gRPC
    assert proc is mock_popen


def test_wait_for_connect_port_success(spark_backend):
    """_wait_for_connect_port returns True when TCP connect succeeds."""
    with patch("kubeflow.spark.backends.kubernetes.backend.socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__ = Mock(return_value=None)
        mock_conn.return_value.__exit__ = Mock(return_value=False)
        assert spark_backend._wait_for_connect_port("127.0.0.1", 15002, timeout_sec=2) is True


def test_wait_for_connect_port_timeout(spark_backend):
    """_wait_for_connect_port returns False when TCP connect never succeeds."""
    with patch(
        "kubeflow.spark.backends.kubernetes.backend.socket.create_connection",
        side_effect=OSError("Connection refused"),
    ):
        assert spark_backend._wait_for_connect_port("127.0.0.1", 15002, timeout_sec=1) is False


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="timeout_error_when_creating_session",
            error_type="timeout",
        ),
        Case(
            name="runtime_error_when_creating_session",
            error_type="runtime",
        ),
    ],
)
def test_create_session_errors(test_case):
    """Test create_session error handling."""
    with patch("kubernetes.config.load_kube_config"):
        mock_custom_api = Mock()

        if test_case.error_type == "timeout":
            mock_custom_api.create_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_timeout=True)
            )
        else:
            mock_custom_api.create_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_error=True)
            )

        with patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api):
            backend = KubernetesBackend(KubernetesBackendConfig())

            expected_error = TimeoutError if test_case.error_type == "timeout" else RuntimeError
            with pytest.raises(expected_error, match="SparkConnect"):
                backend._create_session(options=[Name("test-session")])


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="timeout_error_when_getting_session",
            error_type="timeout",
        ),
        Case(
            name="runtime_error_when_getting_session",
            error_type="runtime",
        ),
    ],
)
def test_get_session_errors(test_case):
    """Test get_session error handling."""
    with patch("kubernetes.config.load_kube_config"):
        mock_custom_api = Mock()

        if test_case.error_type == "timeout":
            mock_custom_api.get_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_timeout=True)
            )
        else:
            mock_custom_api.get_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_error=True)
            )

        with patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api):
            backend = KubernetesBackend(KubernetesBackendConfig())

            expected_error = TimeoutError if test_case.error_type == "timeout" else RuntimeError
            with pytest.raises(expected_error, match="SparkConnect"):
                backend.get_session(name="test-session")


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="timeout_error_when_listing_sessions",
            error_type="timeout",
        ),
        Case(
            name="runtime_error_when_listing_sessions",
            error_type="runtime",
        ),
    ],
)
def test_list_sessions_errors(test_case):
    """Test list_sessions error handling."""
    with patch("kubernetes.config.load_kube_config"):
        mock_custom_api = Mock()

        if test_case.error_type == "timeout":
            mock_custom_api.list_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_timeout=True)
            )
        else:
            mock_custom_api.list_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_error=True)
            )

        with patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api):
            backend = KubernetesBackend(KubernetesBackendConfig())

            expected_error = TimeoutError if test_case.error_type == "timeout" else RuntimeError
            with pytest.raises(expected_error, match="SparkConnect"):
                backend.list_sessions()


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="timeout_error_when_deleting_session",
            error_type="timeout",
        ),
        Case(
            name="runtime_error_when_deleting_session",
            error_type="runtime",
        ),
    ],
)
def test_delete_session_errors(test_case):
    """Test delete_session error handling."""
    with patch("kubernetes.config.load_kube_config"):
        mock_custom_api = Mock()

        if test_case.error_type == "timeout":
            mock_custom_api.delete_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_timeout=True)
            )
        else:
            mock_custom_api.delete_namespaced_custom_object.return_value = (
                create_mock_thread_with_error(raise_error=True)
            )

        with patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api):
            backend = KubernetesBackend(KubernetesBackendConfig())

            expected_error = TimeoutError if test_case.error_type == "timeout" else RuntimeError
            with pytest.raises(expected_error, match="SparkConnect"):
                backend.delete_session("test-session")


@pytest.mark.parametrize(
    "test_case",
    [
        Case(
            name="timeout_error_when_reading_pod_logs",
            error_type="timeout",
        ),
        Case(
            name="runtime_error_when_reading_pod_logs",
            error_type="runtime",
        ),
    ],
)
def test_get_session_logs_errors(test_case):
    """Test get_session_logs error handling."""
    with patch("kubernetes.config.load_kube_config"):
        mock_session_info = Mock(pod_name="test-pod")
        mock_core_api = Mock()

        if test_case.error_type == "timeout":
            mock_core_api.read_namespaced_pod_log.return_value = create_mock_thread_with_error(
                raise_timeout=True
            )
        else:
            mock_core_api.read_namespaced_pod_log.return_value = create_mock_thread_with_error(
                raise_error=True
            )

        with (
            patch("kubernetes.client.CustomObjectsApi"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_core_api),
        ):
            backend = KubernetesBackend(KubernetesBackendConfig())
            backend.get_session = Mock(return_value=mock_session_info)

            expected_error = TimeoutError if test_case.error_type == "timeout" else RuntimeError
            with pytest.raises(expected_error, match="SparkConnect"):
                list(backend.get_session_logs(name="test-session", follow=False))


class TestNameOptionExtraction:
    """Tests for Name option extraction and auto-generation."""

    def test_extract_name_option_with_name(self, spark_backend):
        """Extract name from Name option."""
        options = [Name("test-name"), Labels({"app": "spark"})]
        name, filtered = spark_backend._extract_name_option(options)

        assert name == "test-name"
        assert len(filtered) == 1
        assert isinstance(filtered[0], Labels)

    def test_extract_name_option_auto_generates(self, spark_backend):
        """Auto-generate name when no Name option provided."""
        options = [Labels({"app": "spark"})]
        name, filtered = spark_backend._extract_name_option(options)

        assert name.startswith("spark-connect-")
        assert len(filtered) == 1
        assert isinstance(filtered[0], Labels)

    def test_extract_name_option_handles_none(self, spark_backend):
        """Auto-generate name when options is None."""
        name, filtered = spark_backend._extract_name_option(None)

        assert name.startswith("spark-connect-")
        assert filtered == []

    def test_extract_name_option_handles_empty_list(self, spark_backend):
        """Auto-generate name when options list is empty."""
        name, filtered = spark_backend._extract_name_option([])

        assert name.startswith("spark-connect-")
        assert filtered == []
