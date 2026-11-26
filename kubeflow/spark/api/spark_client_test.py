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

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.api.spark_client import SparkClient, _append_headers_to_url
from kubeflow.spark.options import Labels
from kubeflow.spark.types.types import (
    FileJob,
    FuncJob,
    SparkConnectAuth,
    SparkJob,
)
from kubeflow.test.common import FAILED, SUCCESS, TestCase


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


# ------------------------------------------------------------------
# SparkConnectAuth dataclass validation
# ------------------------------------------------------------------


class TestSparkConnectAuth:
    """Tests for the SparkConnectAuth dataclass."""

    def test_token_only(self):
        auth = SparkConnectAuth(token="my-token")
        assert auth.token == "my-token"
        assert auth.headers is None
        assert auth.channel_builder is None

    def test_headers_only(self):
        auth = SparkConnectAuth(headers={"x-api-key": "key-123"})
        assert auth.token is None
        assert auth.headers == {"x-api-key": "key-123"}
        assert auth.channel_builder is None

    def test_channel_builder_only(self):
        builder = MagicMock()
        auth = SparkConnectAuth(channel_builder=builder)
        assert auth.token is None
        assert auth.headers is None
        assert auth.channel_builder is builder

    def test_token_and_headers(self):
        auth = SparkConnectAuth(token="tok", headers={"x-api-key": "k"})
        assert auth.token == "tok"
        assert auth.headers == {"x-api-key": "k"}

    def test_channel_builder_with_token_raises(self):
        with pytest.raises(ValueError, match="channel_builder cannot be combined"):
            SparkConnectAuth(token="tok", channel_builder=MagicMock())

    def test_channel_builder_with_headers_raises(self):
        with pytest.raises(ValueError, match="channel_builder cannot be combined"):
            SparkConnectAuth(headers={"x-api-key": "k"}, channel_builder=MagicMock())

    def test_all_none_is_valid(self):
        auth = SparkConnectAuth()
        assert auth.token is None
        assert auth.headers is None
        assert auth.channel_builder is None


# ------------------------------------------------------------------
# connect() – authentication paths
# ------------------------------------------------------------------


class TestConnectAuth:
    """Tests for SparkClient.connect() with SparkConnectAuth."""

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    @patch("kubeflow.spark.api.spark_client.SparkSession")
    def test_connect_token_shorthand_backward_compat(self, mock_session, _mock_backend):
        """token= alone still works exactly as before."""
        mock_builder = MagicMock()
        mock_session.builder.remote.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        client = SparkClient()
        client.connect(base_url="sc://host:15002", token="my-tok")

        mock_session.builder.remote.assert_called_once_with("sc://host:15002")
        mock_builder.config.assert_called_once_with("spark.connect.authenticate.token", "my-tok")
        mock_builder.getOrCreate.assert_called_once()

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    @patch("kubeflow.spark.api.spark_client.SparkSession")
    def test_connect_auth_token(self, mock_session, _mock_backend):
        """auth=SparkConnectAuth(token=...) sets the bearer token."""
        mock_builder = MagicMock()
        mock_session.builder.remote.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        client = SparkClient()
        client.connect(
            base_url="sc://host:15002",
            auth=SparkConnectAuth(token="auth-tok"),
        )

        mock_session.builder.remote.assert_called_once_with("sc://host:15002")
        mock_builder.config.assert_called_once_with("spark.connect.authenticate.token", "auth-tok")

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    @patch("kubeflow.spark.api.spark_client.SparkSession")
    def test_connect_auth_headers(self, mock_session, _mock_backend):
        """auth=SparkConnectAuth(headers=...) appends headers to URL."""
        mock_builder = MagicMock()
        mock_session.builder.remote.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        client = SparkClient()
        client.connect(
            base_url="sc://gateway:443",
            auth=SparkConnectAuth(headers={"x-api-key": "key-123"}),
        )

        # Headers are appended as ;key=value to the URL
        expected_url = "sc://gateway:443;x-api-key=key-123"
        mock_session.builder.remote.assert_called_once_with(expected_url)
        # No token config should be set
        mock_builder.config.assert_not_called()

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    @patch("kubeflow.spark.api.spark_client.SparkSession")
    def test_connect_auth_token_and_headers(self, mock_session, _mock_backend):
        """Token + headers both work together."""
        mock_builder = MagicMock()
        mock_session.builder.remote.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        client = SparkClient()
        client.connect(
            base_url="sc://host:15002",
            auth=SparkConnectAuth(token="tok", headers={"x-identity": "user-1"}),
        )

        expected_url = "sc://host:15002;x-identity=user-1"
        mock_session.builder.remote.assert_called_once_with(expected_url)
        mock_builder.config.assert_called_once_with("spark.connect.authenticate.token", "tok")

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    @patch("kubeflow.spark.api.spark_client.SparkSession")
    def test_connect_auth_channel_builder(self, mock_session, _mock_backend):
        """auth=SparkConnectAuth(channel_builder=...) uses channelBuilder."""
        mock_builder = MagicMock()
        mock_session.builder.channelBuilder.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        channel = MagicMock()
        client = SparkClient()
        client.connect(
            base_url="sc://host:15002",
            auth=SparkConnectAuth(channel_builder=channel),
        )

        mock_session.builder.channelBuilder.assert_called_once_with(channel)
        mock_builder.getOrCreate.assert_called_once()
        # remote() should NOT be called when using channel_builder
        mock_session.builder.remote.assert_not_called()

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    def test_connect_token_and_auth_token_conflict(self, _mock_backend):
        """Both token= and auth.token set raises ValueError."""
        client = SparkClient()
        with pytest.raises(ValueError, match="Ambiguous authentication"):
            client.connect(
                base_url="sc://host:15002",
                token="standalone-tok",
                auth=SparkConnectAuth(token="auth-tok"),
            )

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    @patch("kubeflow.spark.api.spark_client.SparkSession")
    def test_connect_token_shorthand_with_auth_headers(self, mock_session, _mock_backend):
        """token= combined with auth(headers=...) merges both."""
        mock_builder = MagicMock()
        mock_session.builder.remote.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        client = SparkClient()
        client.connect(
            base_url="sc://host:15002",
            token="my-tok",
            auth=SparkConnectAuth(headers={"x-api-key": "key-123"}),
        )

        expected_url = "sc://host:15002;x-api-key=key-123"
        mock_session.builder.remote.assert_called_once_with(expected_url)
        mock_builder.config.assert_called_once_with("spark.connect.authenticate.token", "my-tok")

    @patch("kubeflow.spark.api.spark_client.KubernetesBackend")
    @patch("kubeflow.spark.api.spark_client.SparkSession")
    def test_connect_no_auth(self, mock_session, _mock_backend):
        """base_url without any auth falls back to plain connect."""
        mock_builder = MagicMock()
        mock_session.builder.remote.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        client = SparkClient()
        client.connect(base_url="sc://host:15002")

        mock_session.builder.remote.assert_called_once_with("sc://host:15002")
        mock_builder.config.assert_not_called()


# ------------------------------------------------------------------
# _append_headers_to_url helper
# ------------------------------------------------------------------


class TestAppendHeadersToUrl:
    """Tests for the _append_headers_to_url helper."""

    def test_single_header(self):
        result = _append_headers_to_url("sc://host:15002", {"x-api-key": "key-123"})
        assert result == "sc://host:15002;x-api-key=key-123"

    def test_multiple_headers(self):
        result = _append_headers_to_url(
            "sc://host:443",
            {"x-api-key": "k", "x-identity": "user-1"},
        )
        assert "x-api-key=k" in result
        assert "x-identity=user-1" in result

    def test_url_already_has_params(self):
        result = _append_headers_to_url("sc://host:15002;use_ssl=true", {"x-api-key": "k"})
        assert result == "sc://host:15002;use_ssl=true;x-api-key=k"
