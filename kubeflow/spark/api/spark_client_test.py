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

from datetime import datetime, timezone
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
    SparkJobStatus,
)
from kubeflow.test.common import FAILED, SUCCESS, TestCase

# ------------------------------------------------------------------
# Helper: create a SparkClient with a mocked KubernetesBackend
# ------------------------------------------------------------------


def _make_client():
    """Return (client, mock_backend) with KubernetesBackend patched out."""
    patcher = patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    mock_backend_cls = patcher.start()
    client = SparkClient()
    return client, mock_backend_cls.return_value, patcher


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# connect() — connect mode (base_url provided)
# ------------------------------------------------------------------


class TestConnectMode:
    """Tests for SparkClient.connect() when base_url is provided."""

    def test_connect_valid_url(self):
        """connect() with a valid sc:// URL returns a SparkSession."""
        client, backend, patcher = _make_client()
        try:
            with patch("kubeflow.spark.api.spark_client.SparkSession") as mock_session:
                mock_builder = mock_session.builder.remote.return_value
                mock_builder.getOrCreate.return_value = "fake-session"

                result = client.connect(base_url="sc://my-host:15002")

                mock_session.builder.remote.assert_called_once_with("sc://my-host:15002")
                mock_builder.getOrCreate.assert_called_once()
                assert result == "fake-session"
                # Backend should NOT be called in connect mode
                backend.create_and_connect.assert_not_called()
        finally:
            patcher.stop()

    def test_connect_valid_url_with_token(self):
        """connect() with base_url and token configures authentication."""
        client, backend, patcher = _make_client()
        try:
            with patch("kubeflow.spark.api.spark_client.SparkSession") as mock_session:
                mock_builder = mock_session.builder.remote.return_value
                mock_configured = mock_builder.config.return_value
                mock_configured.getOrCreate.return_value = "authed-session"

                result = client.connect(
                    base_url="sc://my-host:15002",
                    token="my-secret-token",
                )

                mock_builder.config.assert_called_once_with(
                    "spark.connect.authenticate.token", "my-secret-token"
                )
                assert result == "authed-session"
        finally:
            patcher.stop()

    @pytest.mark.parametrize(
        "invalid_url,expected_msg",
        [
            ("http://host:15002", "Invalid scheme"),
            ("sc://:15002", "Host is required"),
            ("sc://host", "Port is required"),
        ],
    )
    def test_connect_invalid_url(self, invalid_url, expected_msg):
        """connect() with an invalid URL raises ValueError."""
        client, _, patcher = _make_client()
        try:
            with pytest.raises(ValueError, match=expected_msg):
                client.connect(base_url=invalid_url)
        finally:
            patcher.stop()


# ------------------------------------------------------------------
# connect() — create mode (no base_url)
# ------------------------------------------------------------------


class TestConnectCreateMode:
    """Tests for SparkClient.connect() when no base_url is provided (create mode)."""

    def test_connect_create_mode_defaults(self):
        """connect() without base_url delegates to backend.create_and_connect."""
        client, backend, patcher = _make_client()
        try:
            backend.create_and_connect.return_value = "new-session"

            result = client.connect()

            backend.create_and_connect.assert_called_once_with(
                num_executors=None,
                resources_per_executor=None,
                spark_conf=None,
                driver=None,
                executor=None,
                options=None,
                timeout=300,
                connect_timeout=120,
            )
            assert result == "new-session"
        finally:
            patcher.stop()

    def test_connect_create_mode_custom_params(self):
        """connect() forwards all parameters to backend.create_and_connect."""
        client, backend, patcher = _make_client()
        try:
            backend.create_and_connect.return_value = "custom-session"

            result = client.connect(
                num_executors=5,
                resources_per_executor={"cpu": "2", "memory": "4Gi"},
                spark_conf={"spark.executor.memoryOverhead": "512m"},
                timeout=600,
                connect_timeout=60,
            )

            backend.create_and_connect.assert_called_once_with(
                num_executors=5,
                resources_per_executor={"cpu": "2", "memory": "4Gi"},
                spark_conf={"spark.executor.memoryOverhead": "512m"},
                driver=None,
                executor=None,
                options=None,
                timeout=600,
                connect_timeout=60,
            )
            assert result == "custom-session"
        finally:
            patcher.stop()

    def test_connect_create_mode_timeout_error(self):
        """connect() propagates TimeoutError from backend."""
        client, backend, patcher = _make_client()
        try:
            backend.create_and_connect.side_effect = TimeoutError("Session creation timed out")

            with pytest.raises(TimeoutError, match="Session creation timed out"):
                client.connect()
        finally:
            patcher.stop()

    def test_connect_create_mode_runtime_error(self):
        """connect() propagates RuntimeError from backend."""
        client, backend, patcher = _make_client()
        try:
            backend.create_and_connect.side_effect = RuntimeError("Failed to create session")

            with pytest.raises(RuntimeError, match="Failed to create session"):
                client.connect()
        finally:
            patcher.stop()


# ------------------------------------------------------------------
# submit_job()
# ------------------------------------------------------------------


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


def test_submit_job_with_executors():
    """submit_job() forwards num_executors and resources_per_executor."""
    client, backend, patcher = _make_client()
    try:
        backend.submit_job.return_value = SparkJob(
            name="spark-job-456",
            namespace="default",
        )

        job = FileJob(file_source="s3://bucket/app.py")
        name = client.submit_job(
            job=job,
            num_executors=10,
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
        )

        assert name == "spark-job-456"
        backend.submit_job.assert_called_once_with(
            job=job,
            num_executors=10,
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
            spark_conf=None,
            options=None,
        )
    finally:
        patcher.stop()


# ------------------------------------------------------------------
# get_job()
# ------------------------------------------------------------------


class TestGetJob:
    """Tests for SparkClient.get_job()."""

    def test_get_job_success(self):
        """get_job() returns the SparkJob from the backend."""
        client, backend, patcher = _make_client()
        try:
            expected_job = SparkJob(
                name="spark-job-abc",
                namespace="default",
                status=SparkJobStatus.RUNNING,
                creation_timestamp=datetime(2026, 10, 30, tzinfo=timezone.utc),
                num_executors=3,
                driver_pod_name="spark-job-abc-driver",
            )
            backend.get_job.return_value = expected_job

            result = client.get_job("spark-job-abc")

            backend.get_job.assert_called_once_with("spark-job-abc")
            assert result == expected_job
            assert result.name == "spark-job-abc"
            assert result.status == SparkJobStatus.RUNNING
        finally:
            patcher.stop()

    def test_get_job_not_found(self):
        """get_job() propagates RuntimeError when job is not found."""
        client, backend, patcher = _make_client()
        try:
            backend.get_job.side_effect = RuntimeError("SparkApplication not found")

            with pytest.raises(RuntimeError, match="not found"):
                client.get_job("nonexistent-job")
        finally:
            patcher.stop()


# ------------------------------------------------------------------
# list_jobs()
# ------------------------------------------------------------------


class TestListJobs:
    """Tests for SparkClient.list_jobs()."""

    def test_list_jobs_no_filter(self):
        """list_jobs() without status filter returns all jobs."""
        client, backend, patcher = _make_client()
        try:
            jobs = [
                SparkJob(name="job-1", namespace="default", status=SparkJobStatus.RUNNING),
                SparkJob(name="job-2", namespace="default", status=SparkJobStatus.COMPLETED),
                SparkJob(name="job-3", namespace="default", status=SparkJobStatus.FAILED),
            ]
            backend.list_jobs.return_value = jobs

            result = client.list_jobs()

            backend.list_jobs.assert_called_once_with(status=None)
            assert len(result) == 3
        finally:
            patcher.stop()

    def test_list_jobs_with_status_filter(self):
        """list_jobs() with status filter passes it to the backend."""
        client, backend, patcher = _make_client()
        try:
            running_jobs = [
                SparkJob(name="job-1", namespace="default", status=SparkJobStatus.RUNNING),
            ]
            backend.list_jobs.return_value = running_jobs

            status_filter = {SparkJobStatus.RUNNING}
            result = client.list_jobs(status=status_filter)

            backend.list_jobs.assert_called_once_with(status=status_filter)
            assert len(result) == 1
            assert result[0].status == SparkJobStatus.RUNNING
        finally:
            patcher.stop()

    def test_list_jobs_empty(self):
        """list_jobs() returns empty list when no jobs exist."""
        client, backend, patcher = _make_client()
        try:
            backend.list_jobs.return_value = []

            result = client.list_jobs()

            assert result == []
        finally:
            patcher.stop()


# ------------------------------------------------------------------
# delete_job()
# ------------------------------------------------------------------


class TestDeleteJob:
    """Tests for SparkClient.delete_job()."""

    def test_delete_job_success(self):
        """delete_job() delegates to backend."""
        client, backend, patcher = _make_client()
        try:
            backend.delete_job.return_value = None

            client.delete_job("spark-job-to-delete")

            backend.delete_job.assert_called_once_with("spark-job-to-delete")
        finally:
            patcher.stop()

    def test_delete_job_not_found(self):
        """delete_job() propagates RuntimeError when job is not found."""
        client, backend, patcher = _make_client()
        try:
            backend.delete_job.side_effect = RuntimeError("SparkApplication not found")

            with pytest.raises(RuntimeError, match="not found"):
                client.delete_job("nonexistent-job")
        finally:
            patcher.stop()


# ------------------------------------------------------------------
# wait_for_job_status()
# ------------------------------------------------------------------


class TestWaitForJobStatus:
    """Tests for SparkClient.wait_for_job_status()."""

    def test_wait_for_job_status_defaults(self):
        """wait_for_job_status() uses default status={COMPLETED} and timeout=600."""
        client, backend, patcher = _make_client()
        try:
            expected_job = SparkJob(
                name="spark-job-done",
                namespace="default",
                status=SparkJobStatus.COMPLETED,
            )
            backend.wait_for_job_status.return_value = expected_job

            result = client.wait_for_job_status("spark-job-done")

            backend.wait_for_job_status.assert_called_once_with(
                name="spark-job-done",
                status={SparkJobStatus.COMPLETED},
                timeout=600,
                polling_interval=2,
            )
            assert result.status == SparkJobStatus.COMPLETED
        finally:
            patcher.stop()

    def test_wait_for_job_status_custom_params(self):
        """wait_for_job_status() forwards custom status, timeout, and polling_interval."""
        client, backend, patcher = _make_client()
        try:
            expected_job = SparkJob(
                name="spark-job-x",
                namespace="default",
                status=SparkJobStatus.RUNNING,
            )
            backend.wait_for_job_status.return_value = expected_job

            target = {SparkJobStatus.RUNNING, SparkJobStatus.COMPLETED}
            result = client.wait_for_job_status(
                "spark-job-x",
                status=target,
                timeout=120,
                polling_interval=5,
            )

            backend.wait_for_job_status.assert_called_once_with(
                name="spark-job-x",
                status=target,
                timeout=120,
                polling_interval=5,
            )
            assert result.status == SparkJobStatus.RUNNING
        finally:
            patcher.stop()

    def test_wait_for_job_status_invalid_polling_interval(self):
        """wait_for_job_status() raises ValueError for non-positive polling_interval."""
        client, _, patcher = _make_client()
        try:
            with pytest.raises(ValueError, match="[Pp]olling interval"):
                client.wait_for_job_status("job", polling_interval=0)
        finally:
            patcher.stop()

    def test_wait_for_job_status_invalid_timeout(self):
        """wait_for_job_status() raises ValueError for non-positive timeout."""
        client, _, patcher = _make_client()
        try:
            with pytest.raises(ValueError, match="[Tt]imeout"):
                client.wait_for_job_status("job", timeout=0)
        finally:
            patcher.stop()

    def test_wait_for_job_status_polling_exceeds_timeout(self):
        """wait_for_job_status() raises ValueError when polling_interval >= timeout."""
        client, _, patcher = _make_client()
        try:
            with pytest.raises(ValueError, match="[Pp]olling interval"):
                client.wait_for_job_status("job", timeout=10, polling_interval=10)
        finally:
            patcher.stop()


# ------------------------------------------------------------------
# Session management: list_sessions(), get_session(), delete_session()
# ------------------------------------------------------------------


class TestSessionManagement:
    """Tests for SparkClient session management methods."""

    def test_list_sessions(self):
        """list_sessions() returns sessions from the backend."""
        client, backend, patcher = _make_client()
        try:
            sessions = [
                SparkConnectInfo(
                    name="session-1",
                    namespace="default",
                    state=SparkConnectState.READY,
                    service_name="session-1-server",
                ),
                SparkConnectInfo(
                    name="session-2",
                    namespace="default",
                    state=SparkConnectState.PROVISIONING,
                ),
            ]
            backend.list_sessions.return_value = sessions

            result = client.list_sessions()

            backend.list_sessions.assert_called_once()
            assert len(result) == 2
            assert result[0].name == "session-1"
            assert result[0].state == SparkConnectState.READY
        finally:
            patcher.stop()

    def test_list_sessions_empty(self):
        """list_sessions() returns empty list when no sessions exist."""
        client, backend, patcher = _make_client()
        try:
            backend.list_sessions.return_value = []

            result = client.list_sessions()

            assert result == []
        finally:
            patcher.stop()

    def test_get_session(self):
        """get_session() returns session info from the backend."""
        client, backend, patcher = _make_client()
        try:
            expected = SparkConnectInfo(
                name="my-session",
                namespace="spark-ns",
                state=SparkConnectState.READY,
                driver_pod_name="my-session-driver",
                pod_ip="10.0.0.5",
                service_name="my-session-server",
                creation_timestamp=datetime(2026, 10, 30, tzinfo=timezone.utc),
            )
            backend.get_session.return_value = expected

            result = client.get_session("my-session")

            backend.get_session.assert_called_once_with("my-session")
            assert result.name == "my-session"
            assert result.state == SparkConnectState.READY
            assert result.driver_pod_name == "my-session-driver"
        finally:
            patcher.stop()

    def test_get_session_not_found(self):
        """get_session() propagates RuntimeError when session is not found."""
        client, backend, patcher = _make_client()
        try:
            backend.get_session.side_effect = RuntimeError("SparkConnect not found")

            with pytest.raises(RuntimeError, match="not found"):
                client.get_session("nonexistent")
        finally:
            patcher.stop()

    def test_delete_session(self):
        """delete_session() delegates to backend."""
        client, backend, patcher = _make_client()
        try:
            backend.delete_session.return_value = None

            client.delete_session("session-to-delete")

            backend.delete_session.assert_called_once_with("session-to-delete")
        finally:
            patcher.stop()

    def test_delete_session_not_found(self):
        """delete_session() propagates RuntimeError when session is not found."""
        client, backend, patcher = _make_client()
        try:
            backend.delete_session.side_effect = RuntimeError("SparkConnect not found")

            with pytest.raises(RuntimeError, match="not found"):
                client.delete_session("nonexistent")
        finally:
            patcher.stop()


# ------------------------------------------------------------------
# Log retrieval: get_session_logs(), get_job_logs()
# ------------------------------------------------------------------


class TestLogRetrieval:
    """Tests for SparkClient log retrieval methods."""

    def test_get_session_logs(self):
        """get_session_logs() returns log iterator from the backend."""
        client, backend, patcher = _make_client()
        try:
            backend.get_session_logs.return_value = iter(
                ["Starting Spark...", "Session ready", "Executor registered"]
            )

            logs = list(client.get_session_logs("my-session"))

            backend.get_session_logs.assert_called_once_with("my-session", follow=False)
            assert logs == ["Starting Spark...", "Session ready", "Executor registered"]
        finally:
            patcher.stop()

    def test_get_session_logs_follow(self):
        """get_session_logs(follow=True) passes follow flag to backend."""
        client, backend, patcher = _make_client()
        try:
            backend.get_session_logs.return_value = iter(["log line 1"])

            list(client.get_session_logs("my-session", follow=True))

            backend.get_session_logs.assert_called_once_with("my-session", follow=True)
        finally:
            patcher.stop()

    def test_get_session_logs_no_driver_pod(self):
        """get_session_logs() propagates RuntimeError when driver pod missing."""
        client, backend, patcher = _make_client()
        try:
            backend.get_session_logs.side_effect = RuntimeError("No driver pod")

            with pytest.raises(RuntimeError, match="No driver pod"):
                list(client.get_session_logs("session-no-pod"))
        finally:
            patcher.stop()

    def test_get_job_logs(self):
        """get_job_logs() returns log iterator from the backend."""
        client, backend, patcher = _make_client()
        try:
            backend.get_job_logs.return_value = iter(
                ["Job started", "Processing...", "Job completed"]
            )

            logs = list(client.get_job_logs("spark-job-123"))

            backend.get_job_logs.assert_called_once_with(name="spark-job-123", follow=False)
            assert logs == ["Job started", "Processing...", "Job completed"]
        finally:
            patcher.stop()

    def test_get_job_logs_follow(self):
        """get_job_logs(follow=True) passes follow flag to backend."""
        client, backend, patcher = _make_client()
        try:
            backend.get_job_logs.return_value = iter(["streaming log"])

            list(client.get_job_logs("spark-job-123", follow=True))

            backend.get_job_logs.assert_called_once_with(name="spark-job-123", follow=True)
        finally:
            patcher.stop()
