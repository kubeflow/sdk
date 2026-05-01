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

from kubeflow_spark_api import models
from kubernetes.client import ApiException
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.kubernetes import constants
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
def kubernetes_backend():
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


def create_error_thread(exc: Exception):
    """Create mock thread whose .get() raises the given exception."""
    mock_thread = Mock()
    mock_thread.get.side_effect = exc
    return mock_thread


def get_spark_connect(
    name: str,
    namespace: str = DEFAULT_NAMESPACE,
    state: str | None = None,
    server_status: models.SparkV1alpha1SparkConnectServerStatus | None = None,
) -> models.SparkV1alpha1SparkConnect:
    """Create a mock SparkConnect model for testing."""
    return models.SparkV1alpha1SparkConnect(
        api_version=f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}",
        kind=constants.SPARK_CONNECT_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
        ),
        spec=models.SparkV1alpha1SparkConnectSpec(
            spark_version=constants.DEFAULT_SPARK_VERSION,
            image=constants.DEFAULT_SPARK_IMAGE,
            server=models.SparkV1alpha1ServerSpec(
                cores=constants.DEFAULT_DRIVER_CPU,
                memory=constants.DEFAULT_DRIVER_MEMORY,
            ),
            executor=models.SparkV1alpha1ExecutorSpec(
                instances=2,
                cores=constants.DEFAULT_EXECUTOR_CPU,
                memory=constants.DEFAULT_EXECUTOR_MEMORY,
            ),
        ),
        status=models.SparkV1alpha1SparkConnectStatus(
            state=state,
            server=server_status,
        )
        if state
        else None,
    )


def mock_get_response(name: str) -> dict:
    """Return mock CR response based on session name."""
    if name == SPARK_CONNECT_READY:
        return get_spark_connect(
            name=name,
            state="Ready",
            server_status=models.SparkV1alpha1SparkConnectServerStatus(
                pod_name=f"{name}-0",
                pod_ip="10.0.0.5",
            ),
        ).to_dict()
    elif name == SPARK_CONNECT_PROVISIONING:
        return get_spark_connect(name=name, state="Provisioning").to_dict()
    elif name == SPARK_CONNECT_FAILED:
        return get_spark_connect(name=name, state="Failed").to_dict()
    raise ApiException(status=404, reason="Not Found")


def mock_list_response(*args, **kwargs) -> dict:
    """Return mock list response."""
    spark_connect_list = models.SparkV1alpha1SparkConnectList(
        api_version=f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}",
        kind="SparkConnectList",
        items=[
            get_spark_connect(name="session-1", state="Ready"),
            get_spark_connect(name="session-2", state="Provisioning"),
        ],
    )
    return spark_connect_list.to_dict()


def mock_create_response(*args, **kwargs) -> dict:
    """Return mock create response."""
    body = kwargs.get("body", {})
    # Parse the request body and add status
    spark_connect = models.SparkV1alpha1SparkConnect.from_dict(body)
    spark_connect.status = models.SparkV1alpha1SparkConnectStatus(state="Provisioning")
    return spark_connect.to_dict()


def mock_delete_response(name: str) -> None:
    """Mock delete - raise 404 for unknown sessions."""
    if name.startswith("unknown"):
        raise ApiException(status=404, reason="Not Found")
    return None


def _mock_create(*args, **kw):
    """Mock create_namespaced_custom_object: returns thread whose .get() raises on sentinel."""
    namespace = kw.get("namespace", args[2] if len(args) > 2 else None)
    if namespace == TIMEOUT:
        return create_error_thread(multiprocessing.TimeoutError())
    elif namespace == RUNTIME:
        return create_error_thread(RuntimeError())
    return create_mock_thread(response=mock_create_response(**kw))


def _mock_get(*args, **kw):
    """Mock get_namespaced_custom_object: returns thread whose .get() raises on sentinel."""
    namespace = kw.get("namespace", args[2] if len(args) > 2 else None)
    name = kw.get("name", args[4] if len(args) > 4 else None)
    if namespace == TIMEOUT:
        return create_error_thread(multiprocessing.TimeoutError())
    elif namespace == RUNTIME:
        return create_error_thread(RuntimeError())
    mock_thread = Mock()

    def get_with_exception(timeout=None):
        return mock_get_response(name)

    mock_thread.get = Mock(side_effect=get_with_exception)
    return mock_thread


def _mock_delete(*args, **kw):
    """Mock delete_namespaced_custom_object: returns thread whose .get() raises on sentinel."""
    namespace = kw.get("namespace", args[2] if len(args) > 2 else None)
    name = kw.get("name", args[4] if len(args) > 4 else None)
    if namespace == TIMEOUT:
        return create_error_thread(multiprocessing.TimeoutError())
    elif namespace == RUNTIME:
        return create_error_thread(RuntimeError())
    mock_thread = Mock()

    def get_with_exception(timeout=None):
        mock_delete_response(name)
        return None

    mock_thread.get = Mock(side_effect=get_with_exception)
    return mock_thread


def _mock_list(*args, **kw):
    """Mock list_namespaced_custom_object: returns thread whose .get() raises on sentinel."""
    namespace = kw.get("namespace", args[2] if len(args) > 2 else None)
    if namespace == TIMEOUT:
        return create_error_thread(multiprocessing.TimeoutError())
    elif namespace == RUNTIME:
        return create_error_thread(RuntimeError())
    return create_mock_thread(response=mock_list_response())


def _mock_read_logs(*args, **kw):
    """Mock read_namespaced_pod_log: returns thread whose .get() raises on sentinel."""
    name = kw.get("name", args[1] if len(args) > 1 else None)
    if name == TIMEOUT:
        return create_error_thread(multiprocessing.TimeoutError())
    elif name == RUNTIME:
        return create_error_thread(RuntimeError())
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
def test_create_session(kubernetes_backend, test_case):
    """Test KubernetesBackend._create_session with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        session_name = test_case.config.get("session_name")
        options = [Name(session_name)] if session_name else None

        info = kubernetes_backend._create_session(
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
def test_get_session(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_session with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        info = kubernetes_backend.get_session(test_case.config["name"])

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
def test_list_sessions(kubernetes_backend, test_case):
    """Test KubernetesBackend.list_sessions with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        sessions = kubernetes_backend.list_sessions()

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
def test_delete_session(kubernetes_backend, test_case):
    """Test KubernetesBackend.delete_session with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        kubernetes_backend.delete_session(test_case.config["name"])

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
def test_wait_for_session_ready(kubernetes_backend, test_case):
    """Test KubernetesBackend._wait_for_session_ready with different session states."""
    print("Executing test:", test_case.name)
    try:
        info = kubernetes_backend._wait_for_session_ready(test_case.config["name"], timeout=5)

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
            config={"pod_name": TIMEOUT, "name": SPARK_CONNECT_READY},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when reading pod logs",
            expected_status=FAILED,
            config={"pod_name": RUNTIME, "name": SPARK_CONNECT_READY},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_session_logs(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_session_logs with success and error scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)

        # Mock get_session so execution always reaches the log-reading code path.
        pod_name = test_case.config.get("pod_name", f"{test_case.config['name']}-0")
        kubernetes_backend.get_session = Mock(return_value=Mock(pod_name=pod_name))

        logs = list(kubernetes_backend.get_session_logs(test_case.config["name"], follow=False))

        if test_case.expected_status == SUCCESS:
            assert len(logs) == 2
            assert logs[0] == "log line 1"
        else:
            # Should not reach here for failed test cases
            raise AssertionError(f"Expected {test_case.expected_error.__name__} but test succeeded")

    except Exception as e:
        if test_case.expected_status == FAILED:
            assert type(e) is test_case.expected_error
        else:
            raise
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="in-cluster returns svc URL and no process",
            expected_status=SUCCESS,
            config={"in_cluster": True},
            expected_output={"url_contains": "svc.cluster.local", "proc_is_none": True},
        ),
        TestCase(
            name="out-of-cluster starts port-forward and returns localhost URL",
            expected_status=SUCCESS,
            config={"in_cluster": False},
            expected_output={"url": "sc://127.0.0.1:15002", "proc_is_none": False},
        ),
    ],
)
def test_get_connect_url(kubernetes_backend, test_case):
    """Test get_connect_url for in-cluster and port-forward scenarios."""
    print("Executing test:", test_case.name)
    info = SparkConnectInfo(
        name="test-session",
        namespace="default",
        state=SparkConnectState.READY,
        service_name="test-session-svc",
    )

    if test_case.config["in_cluster"]:
        with patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.96.0.1"}, clear=False):
            url, proc = kubernetes_backend.get_connect_url(info)
    else:
        mock_popen = Mock()
        mock_popen.poll.return_value = None
        with (
            patch.dict(
                "os.environ",
                {"KUBERNETES_SERVICE_HOST": "", "SPARK_CONNECT_LOCAL_PORT": "15002"},
                clear=False,
            ),
            patch(
                "kubeflow.spark.backends.kubernetes.backend.subprocess.Popen",
                return_value=mock_popen,
            ),
            patch("kubeflow.spark.backends.kubernetes.backend.time.sleep"),
            patch.object(kubernetes_backend, "_wait_for_connect_port", return_value=True),
        ):
            url, proc = kubernetes_backend.get_connect_url(info)

    if "url_contains" in test_case.expected_output:
        assert test_case.expected_output["url_contains"] in url
    else:
        assert url == test_case.expected_output["url"]

    if test_case.expected_output["proc_is_none"]:
        assert proc is None
    else:
        assert proc is mock_popen

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="TCP connect succeeds returns True",
            expected_status=SUCCESS,
            config={"side_effect": None},
            expected_output=True,
        ),
        TestCase(
            name="TCP connect never succeeds returns False",
            expected_status=SUCCESS,
            config={"side_effect": OSError("Connection refused")},
            expected_output=False,
        ),
    ],
)
def test_wait_for_connect_port(kubernetes_backend, test_case):
    """Test _wait_for_connect_port returns True on success and False on timeout."""
    print("Executing test:", test_case.name)
    if test_case.config["side_effect"] is None:
        with patch(
            "kubeflow.spark.backends.kubernetes.backend.socket.create_connection"
        ) as mock_conn:
            mock_conn.return_value.__enter__ = Mock(return_value=None)
            mock_conn.return_value.__exit__ = Mock(return_value=False)
            result = kubernetes_backend._wait_for_connect_port("127.0.0.1", 15002, timeout_sec=2)
    else:
        with patch(
            "kubeflow.spark.backends.kubernetes.backend.socket.create_connection",
            side_effect=test_case.config["side_effect"],
        ):
            result = kubernetes_backend._wait_for_connect_port("127.0.0.1", 15002, timeout_sec=1)

    assert result is test_case.expected_output
    print("test execution complete")


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
def test_create_and_connect(kubernetes_backend, test_case):
    """Test create_and_connect with and without Name option."""
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
                kubernetes_backend, "_create_session", return_value=ready_info
            ) as mock_create,
            patch.object(kubernetes_backend, "_wait_for_session_ready", return_value=ready_info),
            patch.object(
                kubernetes_backend, "get_connect_url", return_value=("sc://localhost:15002", None)
            ),
            patch("kubeflow.spark.backends.kubernetes.backend.SparkSession"),
        ):
            kubernetes_backend.create_and_connect(options=options)
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
def test_extract_name_option(kubernetes_backend, test_case):
    """Test KubernetesBackend._extract_name_option for name extraction and auto-generation."""
    print("Executing test:", test_case.name)
    try:
        name, filtered = kubernetes_backend._extract_name_option(test_case.config["options"])

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
