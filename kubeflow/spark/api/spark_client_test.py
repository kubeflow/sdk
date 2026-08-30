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
from kubeflow.spark.options import Labels
from kubeflow.spark.test.common import FAILED, SUCCESS, TestCase
from kubeflow.spark.types.types import (
    FileJob,
    FuncJob,
    SparkJob,
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


@pytest.fixture
def client_with_mock_backend() -> SparkClient:
    """Create a SparkClient without initializing Kubernetes clients."""
    client = object.__new__(SparkClient)
    client.backend = Mock()
    return client


def test_connect_to_existing_server_uses_url_and_token(client_with_mock_backend: SparkClient):
    """Connect mode validates the URL and configures the supplied token."""
    session = Mock()
    builder = Mock()
    builder.config.return_value = builder
    builder.getOrCreate.return_value = session

    with (
        patch("kubeflow.spark.api.spark_client.validate_spark_connect_url") as validate_url,
        patch("kubeflow.spark.api.spark_client.SparkSession") as mock_spark_session,
    ):
        mock_spark_session.builder.remote.return_value = builder
        result = client_with_mock_backend.connect(
            base_url="sc://spark.example:15002",
            token="test-token",
        )

    assert result is session
    validate_url.assert_called_once_with("sc://spark.example:15002")
    mock_spark_session.builder.remote.assert_called_once_with("sc://spark.example:15002")
    builder.config.assert_called_once_with("spark.connect.authenticate.token", "test-token")
    builder.getOrCreate.assert_called_once_with()
    client_with_mock_backend.backend.create_and_connect.assert_not_called()


def test_connect_creates_session_with_requested_configuration(
    client_with_mock_backend: SparkClient,
):
    """Create mode forwards every session setting to the backend."""
    expected_session = Mock()
    client_with_mock_backend.backend.create_and_connect.return_value = expected_session
    driver = Mock()
    executor = Mock()
    options = [Mock()]
    spark_conf = {"spark.sql.adaptive.enabled": "true"}
    resources = {"cpu": "2", "memory": "4Gi"}

    result = client_with_mock_backend.connect(
        num_executors=3,
        resources_per_executor=resources,
        spark_conf=spark_conf,
        driver=driver,
        executor=executor,
        options=options,
        timeout=42,
        connect_timeout=24,
    )

    assert result is expected_session
    client_with_mock_backend.backend.create_and_connect.assert_called_once_with(
        num_executors=3,
        resources_per_executor=resources,
        spark_conf=spark_conf,
        driver=driver,
        executor=executor,
        options=options,
        timeout=42,
        connect_timeout=24,
    )


@pytest.mark.parametrize(
    ("method_name", "args", "expected_kwargs", "returns_backend_value"),
    [
        ("list_sessions", (), {}, True),
        ("get_session", ("daily-orders",), {}, True),
        ("delete_session", ("daily-orders",), {}, False),
        ("get_session_logs", ("daily-orders", True), {"follow": True}, True),
    ],
)
def test_session_management_methods_delegate_to_backend(
    client_with_mock_backend: SparkClient,
    method_name: str,
    args: tuple,
    expected_kwargs: dict[str, bool],
    returns_backend_value: bool,
):
    """Session-management methods preserve their documented delegation behavior."""
    backend_method = getattr(client_with_mock_backend.backend, method_name)
    expected_result = Mock()
    backend_method.return_value = expected_result

    result = getattr(client_with_mock_backend, method_name)(*args)

    if returns_backend_value:
        assert result is expected_result
    else:
        assert result is None
    if expected_kwargs:
        backend_method.assert_called_once_with(args[0], **expected_kwargs)
    else:
        backend_method.assert_called_once_with(*args)


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
