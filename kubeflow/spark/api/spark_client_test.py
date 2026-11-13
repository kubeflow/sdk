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

from unittest.mock import patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.api.spark_client import SparkClient
from kubeflow.spark.options import Labels
from kubeflow.spark.types.types import (
    FileJob,
    FuncJob,
    SparkConnectInfo,
    SparkConnectState,
    SparkJob,
)
from kubeflow.test.common import FAILED, SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="connect to existing server",
            expected_status=SUCCESS,
            config={"base_url": "sc://spark-server:15002"},
            expected_output="existing-session",
        ),
        TestCase(
            name="connect to existing server with token",
            expected_status=SUCCESS,
            config={"base_url": "sc://spark-server:15002", "token": "secret-token"},
            expected_output="authenticated-session",
        ),
        TestCase(
            name="invalid url scheme",
            expected_status=FAILED,
            config={"base_url": "http://spark-server:15002"},
            expected_error=ValueError,
        ),
        TestCase(
            name="url without host",
            expected_status=FAILED,
            config={"base_url": "sc://:15002"},
            expected_error=ValueError,
        ),
        TestCase(
            name="url without port",
            expected_status=FAILED,
            config={"base_url": "sc://spark-server"},
            expected_error=ValueError,
        ),
    ],
)
def test_connect_existing_server(test_case: TestCase):
    """Test connect() against an existing Spark Connect server."""
    print(f"Testing {test_case.name}...")

    with (
        patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend,
        patch("kubeflow.spark.api.spark_client.SparkSession") as mock_session,
    ):
        backend = mock_backend.return_value
        builder = mock_session.builder.remote.return_value
        builder.getOrCreate.return_value = test_case.expected_output
        builder.config.return_value.getOrCreate.return_value = test_case.expected_output

        client = SparkClient()

        if test_case.expected_status == FAILED:
            with pytest.raises(test_case.expected_error):
                client.connect(base_url=test_case.config["base_url"])
            mock_session.builder.remote.assert_not_called()
        else:
            session = client.connect(
                base_url=test_case.config["base_url"],
                token=test_case.config.get("token"),
            )

            assert session == test_case.expected_output
            mock_session.builder.remote.assert_called_once_with(test_case.config["base_url"])
            if "token" in test_case.config:
                builder.config.assert_called_once_with(
                    "spark.connect.authenticate.token", test_case.config["token"]
                )
            else:
                builder.config.assert_not_called()

        backend.create_and_connect.assert_not_called()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="create session with default arguments",
            expected_status=SUCCESS,
            config={},
            expected_output="new-session",
        ),
        TestCase(
            name="create session with executor resources and spark configuration",
            expected_status=SUCCESS,
            config={
                "num_executors": 3,
                "resources_per_executor": {"cpu": "2", "memory": "4Gi"},
                "spark_conf": {"spark.executor.memoryOverhead": "512m"},
                "options": [Labels({"team": "ml"})],
                "timeout": 600,
                "connect_timeout": 60,
            },
            expected_output="configured-session",
        ),
        TestCase(
            name="session creation times out",
            expected_status=FAILED,
            config={},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="session creation fails",
            expected_status=FAILED,
            config={},
            expected_error=RuntimeError,
        ),
    ],
)
def test_connect_create_session(test_case: TestCase):
    """Test connect() delegation to KubernetesBackend when no base_url is given."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        if test_case.expected_status == FAILED:
            backend.create_and_connect.side_effect = test_case.expected_error(
                "Session creation failed"
            )
            with pytest.raises(test_case.expected_error):
                client.connect()
        else:
            backend.create_and_connect.return_value = test_case.expected_output
            session = client.connect(**test_case.config)

            assert session == test_case.expected_output

        backend.create_and_connect.assert_called_once_with(
            num_executors=test_case.config.get("num_executors"),
            resources_per_executor=test_case.config.get("resources_per_executor"),
            spark_conf=test_case.config.get("spark_conf"),
            driver=None,
            executor=None,
            options=test_case.config.get("options"),
            timeout=test_case.config.get("timeout", 300),
            connect_timeout=test_case.config.get("connect_timeout", 120),
        )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list sessions",
            expected_status=SUCCESS,
            expected_output=[
                SparkConnectInfo(
                    name="session-1",
                    namespace="default",
                    state=SparkConnectState.READY,
                ),
                SparkConnectInfo(
                    name="session-2",
                    namespace="default",
                    state=SparkConnectState.PROVISIONING,
                ),
            ],
        ),
        TestCase(
            name="list sessions when none exist",
            expected_status=SUCCESS,
            expected_output=[],
        ),
    ],
)
def test_list_sessions(test_case: TestCase):
    """Test list_sessions delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        backend.list_sessions.return_value = test_case.expected_output

        client = SparkClient()
        sessions = client.list_sessions()

        assert sessions == test_case.expected_output
        backend.list_sessions.assert_called_once_with()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get session by name",
            expected_status=SUCCESS,
            config={"session_name": "session-1"},
            expected_output=SparkConnectInfo(
                name="session-1",
                namespace="default",
                state=SparkConnectState.READY,
            ),
        ),
        TestCase(
            name="get session not found raises error",
            expected_status=FAILED,
            config={"session_name": "nonexistent-session"},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_session(test_case: TestCase):
    """Test get_session delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        if test_case.expected_status == FAILED:
            backend.get_session.side_effect = test_case.expected_error("Session not found")
            with pytest.raises(test_case.expected_error):
                client.get_session(test_case.config["session_name"])
        else:
            backend.get_session.return_value = test_case.expected_output
            session = client.get_session(test_case.config["session_name"])

            assert session == test_case.expected_output

        backend.get_session.assert_called_once_with(test_case.config["session_name"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete session by name",
            expected_status=SUCCESS,
            config={"session_name": "session-1"},
        ),
        TestCase(
            name="delete session not found raises error",
            expected_status=FAILED,
            config={"session_name": "nonexistent-session"},
            expected_error=RuntimeError,
        ),
    ],
)
def test_delete_session(test_case: TestCase):
    """Test delete_session delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        if test_case.expected_status == FAILED:
            backend.delete_session.side_effect = test_case.expected_error("Session not found")
            with pytest.raises(test_case.expected_error):
                client.delete_session(test_case.config["session_name"])
        else:
            client.delete_session(test_case.config["session_name"])

        backend.delete_session.assert_called_once_with(test_case.config["session_name"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get session logs with default follow invocation",
            expected_status=SUCCESS,
            config={"session_name": "session-1"},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get session logs explicit non-follow",
            expected_status=SUCCESS,
            config={"session_name": "session-1", "follow": False},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get session logs explicit follow",
            expected_status=SUCCESS,
            config={"session_name": "session-1", "follow": True},
            expected_output=["log stream 1", "log stream 2"],
        ),
        TestCase(
            name="get session logs without a driver pod raises error",
            expected_status=FAILED,
            config={"session_name": "session-1"},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_session_logs(test_case: TestCase):
    """Test get_session_logs delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()
        follow = test_case.config.get("follow", False)

        if test_case.expected_status == FAILED:
            backend.get_session_logs.side_effect = test_case.expected_error("No driver pod")
            with pytest.raises(test_case.expected_error):
                list(client.get_session_logs(test_case.config["session_name"]))
        else:
            backend.get_session_logs.return_value = iter(test_case.expected_output)
            if "follow" in test_case.config:
                logs = list(
                    client.get_session_logs(test_case.config["session_name"], follow=follow)
                )
            else:
                logs = list(client.get_session_logs(test_case.config["session_name"]))

            assert logs == test_case.expected_output

        backend.get_session_logs.assert_called_once_with(
            test_case.config["session_name"], follow=follow
        )


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
            expected_status=FAILED,
            config={"backend_config": "invalid"},
            expected_error=ValueError,
        ),
    ],
)
def test_create_and_connect(test_case: TestCase):
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
        assert test_case.expected_status == FAILED, f"Unexpected exception in {test_case.name}: {e}"
        # Validate the exception type if specified
        if test_case.expected_error:
            assert isinstance(e, test_case.expected_error), (
                f"Expected exception type '{test_case.expected_error.__name__}' but got '{type(e).__name__}: {str(e)}'"
            )


@pytest.mark.parametrize(
    "job,spark_conf,options,backend_error,expected_error",
    [
        (
            "not-a-job",
            None,
            None,
            TypeError("job must be an instance of FileJob or FuncJob."),
            TypeError,
        ),
        (
            FileJob(file_source=""),
            None,
            None,
            ValueError("`job.file_source` must be a non-empty string."),
            ValueError,
        ),
        (
            FileJob(file_source="s3://bucket/job.py"),
            [],
            None,
            ValueError("spark_conf must be a dictionary."),
            ValueError,
        ),
    ],
)
def test_submit_job_validation(
    job,
    spark_conf,
    options,
    backend_error,
    expected_error,
):
    """Test SparkClient submit_job validation."""

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value

        if backend_error is not None:
            backend.submit_job.side_effect = backend_error

        client = SparkClient()

        with pytest.raises(expected_error):
            client.submit_job(
                job=job,
                spark_conf=spark_conf,
                options=options,
            )


@pytest.mark.parametrize(
    "job,options",
    [
        (
            FileJob(file_source="s3://bucket/job.py"),
            None,
        ),
        (
            FileJob(file_source="s3://bucket/job.py"),
            [Labels({"team": "ml"})],
        ),
        (
            FuncJob(func=lambda: None),
            None,
        ),
    ],
)
def test_submit_job_success(job, options):
    """Test successful submit_job."""

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value

        backend.submit_job.return_value = SparkJob(
            name="spark-job-123",
            namespace="default",
        )

        client = SparkClient()

        name = client.submit_job(job=job, options=options)

        assert name == "spark-job-123"

        backend.submit_job.assert_called_once_with(
            job=job,
            num_executors=None,
            resources_per_executor=None,
            spark_conf=None,
            options=options,
        )
