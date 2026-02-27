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

import multiprocessing
from unittest.mock import Mock, patch

from kubernetes.client import ApiException
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend
from kubeflow.spark.backends.kubernetes.utils import validate_spark_connect_url
from kubeflow.spark.test.common import (
    DEFAULT_NAMESPACE,
    FAILED,
    RUNTIME,
    SPARK_CONNECT_FAILED,
    SPARK_CONNECT_PROVISIONING,
    SPARK_CONNECT_READY,
    SUCCESS,
    TIMEOUT,
    TestCase,
)
from kubeflow.spark.types.options import Labels, Name
from kubeflow.spark.types.types import SparkConnectInfo, SparkConnectState

# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def kubernates_backend():
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


# --------------------------
# Mock Handlers
# --------------------------


def create_mock_thread(response=None):
    """Create mock thread that returns response on .get()."""
    mock_thread = Mock()
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


def mock_delete_response(name: str) -> None:
    """Mock delete - raise 404 for unknown sessions."""
    if name.startswith("unknown"):
        raise ApiException(status=404, reason="Not Found")
    return None


def _mock_create(*args, **kw):
    """Mock create_namespaced_custom_object: raises on sentinel namespace, else returns thread."""
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[2] == RUNTIME:
        raise RuntimeError()
    body = kw.get("body", {})
    return create_mock_thread(
        response={"metadata": body.get("metadata", {}), "status": {"state": "Provisioning"}}
    )


def _mock_get(*args, **kw):
    """Mock get_namespaced_custom_object: raises on sentinel namespace, 404 on unknown name."""
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[2] == RUNTIME:
        raise RuntimeError()
    mock_thread = Mock()

    def get_with_exception(timeout=None):
        return mock_get_response(args[4])

    mock_thread.get = Mock(side_effect=get_with_exception)
    return mock_thread


def _mock_delete(*args, **kw):
    """Mock delete_namespaced_custom_object: raises on sentinel namespace, 404 on unknown name."""
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[2] == RUNTIME:
        raise RuntimeError()
    mock_thread = Mock()

    def get_with_exception(timeout=None):
        mock_delete_response(args[4])
        return None

    mock_thread.get = Mock(side_effect=get_with_exception)
    return mock_thread


def _mock_list(*args, **kw):
    """Mock list_namespaced_custom_object: raises on sentinel namespace, else returns thread."""
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[2] == RUNTIME:
        raise RuntimeError()
    return create_mock_thread(
        response={
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
    )


def _mock_read_logs(*args, **kw):
    """Mock read_namespaced_pod_log: raises on sentinel namespace, else returns log thread."""
    if args[1] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[1] == RUNTIME:
        raise RuntimeError()
    return create_mock_thread(response="log line 1\nlog line 2")


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with name option and executors",
            expected_status=SUCCESS,
            config={
                "num_executors": 3,
                "session_name": "test-session",
                "expected_name_prefix": "test-session",
            },
        ),
        TestCase(
            name="valid flow with auto generated name",
            expected_status=SUCCESS,
            config={
                "num_executors": None,
                "session_name": None,
                "expected_name_prefix": "spark-connect-",
            },
        ),
        TestCase(
            name="timeout error when creating session",
            expected_status=FAILED,
            config={"namespace": TIMEOUT, "session_name": "test-session"},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when creating session",
            expected_status=FAILED,
            config={"namespace": RUNTIME, "session_name": "test-session"},
            expected_error=RuntimeError,
        ),
    ],
)
def test_create_session(kubernates_backend, test_case):
    """Test KubernetesBackend._create_session with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernates_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        session_name = test_case.config.get("session_name")
        options = [Name(session_name)] if session_name else None

        info = kubernates_backend._create_session(
            num_executors=test_case.config.get("num_executors"),
            options=options,
        )

        assert test_case.expected_status == SUCCESS
        assert info.name.startswith(test_case.config["expected_name_prefix"])
        assert info.state == SparkConnectState.PROVISIONING

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with existing session",
            expected_status=SUCCESS,
            config={"name": SPARK_CONNECT_READY},
            expected_output=SparkConnectState.READY,
        ),
        TestCase(
            name="session not found error",
            expected_status=FAILED,
            config={"name": "unknown-session"},
            expected_error=RuntimeError,
        ),
        TestCase(
            name="timeout error when getting session",
            expected_status=FAILED,
            config={"namespace": TIMEOUT, "name": SPARK_CONNECT_READY},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting session",
            expected_status=FAILED,
            config={"namespace": RUNTIME, "name": SPARK_CONNECT_READY},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_session(kubernates_backend, test_case):
    """Test KubernetesBackend.get_session with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernates_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        info = kubernates_backend.get_session(test_case.config["name"])

        assert test_case.expected_status == SUCCESS
        assert info.name == test_case.config["name"]
        assert info.state == test_case.expected_output

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="timeout error when listing sessions",
            expected_status=FAILED,
            config={"namespace": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when listing sessions",
            expected_status=FAILED,
            config={"namespace": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_list_sessions(kubernates_backend, test_case):
    """Test KubernetesBackend.list_sessions with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernates_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        sessions = kubernates_backend.list_sessions()

        assert test_case.expected_status == SUCCESS
        assert len(sessions) == 2
        assert sessions[0].name == "session-1"
        assert sessions[1].name == "session-2"

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with existing session",
            expected_status=SUCCESS,
            config={"name": SPARK_CONNECT_READY},
        ),
        TestCase(
            name="session not found error",
            expected_status=FAILED,
            config={"name": "unknown-session"},
            expected_error=RuntimeError,
        ),
        TestCase(
            name="timeout error when deleting session",
            expected_status=FAILED,
            config={"namespace": TIMEOUT, "name": SPARK_CONNECT_READY},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when deleting session",
            expected_status=FAILED,
            config={"namespace": RUNTIME, "name": SPARK_CONNECT_READY},
            expected_error=RuntimeError,
        ),
    ],
)
def test_delete_session(kubernates_backend, test_case):
    """Test KubernetesBackend.delete_session with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernates_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        kubernates_backend.delete_session(test_case.config["name"])

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with already ready session",
            expected_status=SUCCESS,
            config={"name": SPARK_CONNECT_READY},
            expected_output=SparkConnectState.READY,
        ),
        TestCase(
            name="runtime error when session has failed",
            expected_status=FAILED,
            config={"name": SPARK_CONNECT_FAILED},
            expected_error=RuntimeError,
        ),
    ],
)
def test_wait_for_session_ready(kubernates_backend, test_case):
    """Test KubernetesBackend._wait_for_session_ready with different session states."""
    print("Executing test:", test_case.name)
    try:
        info = kubernates_backend._wait_for_session_ready(test_case.config["name"], timeout=5)

        assert test_case.expected_status == SUCCESS
        assert info.state == test_case.expected_output

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": SPARK_CONNECT_READY},
        ),
        TestCase(
            name="timeout error when reading pod logs",
            expected_status=FAILED,
            config={"namespace": TIMEOUT, "name": SPARK_CONNECT_READY},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when reading pod logs",
            expected_status=FAILED,
            config={"namespace": RUNTIME, "name": SPARK_CONNECT_READY},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_session_logs(kubernates_backend, test_case):
    """Test KubernetesBackend.get_session_logs with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernates_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)

        # Mock get_session so execution always reaches the log-reading code path.
        kubernates_backend.get_session = Mock(
            return_value=Mock(pod_name=f"{test_case.config['name']}-0")
        )

        logs = list(kubernates_backend.get_session_logs(test_case.config["name"], follow=False))

        assert test_case.expected_status == SUCCESS
        assert len(logs) == 2
        assert logs[0] == "log line 1"

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_get_connect_url_in_cluster(kubernates_backend):
    """When KUBERNETES_SERVICE_HOST is set, get_connect_url returns in-cluster URL and no process."""
    info = SparkConnectInfo(
        name="test-session",
        namespace="default",
        state=SparkConnectState.READY,
        service_name="test-session-svc",
    )
    with patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.96.0.1"}, clear=False):
        url, proc = kubernates_backend.get_connect_url(info)
    assert "svc.cluster.local" in url
    assert proc is None


def test_get_connect_url_port_forward(kubernates_backend):
    """When not in cluster, get_connect_url starts port-forward and returns localhost URL."""
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
        patch.object(kubernates_backend, "_wait_for_connect_port", return_value=True),
    ):
        url, proc = kubernates_backend.get_connect_url(info)
    assert url == "sc://127.0.0.1:15002"  # Uses 127.0.0.1 to force IPv4 for gRPC
    assert proc is mock_popen


def test_wait_for_connect_port_success(kubernates_backend):
    """_wait_for_connect_port returns True when TCP connect succeeds."""
    with patch("kubeflow.spark.backends.kubernetes.backend.socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__ = Mock(return_value=None)
        mock_conn.return_value.__exit__ = Mock(return_value=False)
        assert kubernates_backend._wait_for_connect_port("127.0.0.1", 15002, timeout_sec=2) is True


def test_wait_for_connect_port_timeout(kubernates_backend):
    """_wait_for_connect_port returns False when TCP connect never succeeds."""
    with patch(
        "kubeflow.spark.backends.kubernetes.backend.socket.create_connection",
        side_effect=OSError("Connection refused"),
    ):
        assert kubernates_backend._wait_for_connect_port("127.0.0.1", 15002, timeout_sec=1) is False


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid spark connect url",
            expected_status=SUCCESS,
            config={"url": "sc://localhost:15002"},
        ),
        TestCase(
            name="invalid http url error",
            expected_status=FAILED,
            config={"url": "http://localhost:15002"},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid empty url error",
            expected_status=FAILED,
            config={"url": ""},
            expected_error=ValueError,
        ),
    ],
)
def test_validate_spark_connect_url(test_case):
    """Test URL validation for Spark Connect URLs."""
    print("Executing test:", test_case.name)
    try:
        result = validate_spark_connect_url(test_case.config["url"])
        assert test_case.expected_status == SUCCESS
        assert result is True
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with name option",
            expected_status=SUCCESS,
            config={"session_name": "custom-session"},
        ),
        TestCase(
            name="valid flow without options",
            expected_status=SUCCESS,
            config={"session_name": None},
        ),
    ],
)
def test_create_and_connect_with_options(kubernates_backend, test_case):
    """Test create_and_connect passes Name option correctly to backend."""
    print("Executing test:", test_case.name)
    try:
        options = (
            [Name(test_case.config["session_name"])] if test_case.config["session_name"] else None
        )
        ready_info = SparkConnectInfo(
            name=test_case.config["session_name"] or "spark-connect-abc",
            namespace=DEFAULT_NAMESPACE,
            state=SparkConnectState.READY,
            service_name="svc",
        )

        with (
            patch.object(
                kubernates_backend, "_create_session", return_value=ready_info
            ) as mock_create,
            patch.object(kubernates_backend, "_wait_for_session_ready", return_value=ready_info),
            patch.object(
                kubernates_backend, "get_connect_url", return_value=("sc://localhost:15002", None)
            ),
            patch("kubeflow.spark.backends.kubernetes.backend.SparkSession"),
        ):
            kubernates_backend.create_and_connect(options=options)
            mock_create.assert_called_once()
            assert mock_create.call_args.kwargs.get("options") == options

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with name option provided",
            expected_status=SUCCESS,
            config={"options": [Name("test-name"), Labels({"app": "spark"})]},
            expected_output={"name": "test-name", "remaining_count": 1, "remaining_type": Labels},
        ),
        TestCase(
            name="valid flow with no name option auto generates",
            expected_status=SUCCESS,
            config={"options": [Labels({"app": "spark"})]},
            expected_output={
                "name_prefix": "spark-connect-",
                "remaining_count": 1,
                "remaining_type": Labels,
            },
        ),
        TestCase(
            name="valid flow with none options auto generates",
            expected_status=SUCCESS,
            config={"options": None},
            expected_output={"name_prefix": "spark-connect-", "remaining_count": 0},
        ),
        TestCase(
            name="valid flow with empty options auto generates",
            expected_status=SUCCESS,
            config={"options": []},
            expected_output={"name_prefix": "spark-connect-", "remaining_count": 0},
        ),
    ],
)
def test_extract_name_option(kubernates_backend, test_case):
    """Test KubernetesBackend._extract_name_option for name extraction and auto-generation."""
    print("Executing test:", test_case.name)
    try:
        name, filtered = kubernates_backend._extract_name_option(test_case.config["options"])

        assert test_case.expected_status == SUCCESS
        if "name" in test_case.expected_output:
            assert name == test_case.expected_output["name"]
        else:
            assert name.startswith(test_case.expected_output["name_prefix"])
        assert len(filtered) == test_case.expected_output["remaining_count"]
        if "remaining_type" in test_case.expected_output:
            assert isinstance(filtered[0], test_case.expected_output["remaining_type"])

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
