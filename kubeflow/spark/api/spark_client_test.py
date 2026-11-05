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

from pyspark.sql.connect.client.core import DefaultChannelBuilder
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.api.spark_client import SparkClient
from kubeflow.spark.options import Labels
from kubeflow.spark.types.types import (
    FileJob,
    FuncJob,
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


@pytest.mark.parametrize("base_url", ["sc://server", "sc://:15002", "https://server:15002"])
def test_connect_validates_base_url_before_composing_the_token(base_url: str):
    """Test that an invalid base_url is rejected even when a token is given."""

    with (
        patch("kubeflow.spark.api.spark_client.KubernetesBackend"),
        patch("kubeflow.spark.api.spark_client.SparkSession"),
    ):
        client = SparkClient()

        with pytest.raises(ValueError):
            client.connect(base_url=base_url, token="team-token")


@pytest.mark.parametrize("base_url", ["sc://server:15002", "sc://server:15002/"])
def test_connect_passes_the_url_through_without_a_token(base_url: str):
    """Test that connect mode leaves a token-less URL as it was given."""

    with (
        patch("kubeflow.spark.api.spark_client.KubernetesBackend"),
        patch("kubeflow.spark.api.spark_client.SparkSession") as mock_session,
    ):
        client = SparkClient()

        client.connect(base_url=base_url)

        mock_session.builder.remote.assert_called_once_with(base_url)
        mock_session.builder.remote.return_value.getOrCreate.assert_called_once_with()


@pytest.mark.parametrize("base_url", ["sc://server:15002", "sc://server:15002/"])
def test_connect_sends_the_token_in_the_url(base_url: str):
    """Test that the token reaches PySpark as a URL parameter, not as a config."""

    with (
        patch("kubeflow.spark.api.spark_client.KubernetesBackend"),
        patch("kubeflow.spark.api.spark_client.SparkSession") as mock_session,
    ):
        client = SparkClient()

        client.connect(base_url=base_url, token="team-token")

        mock_session.builder.remote.assert_called_once_with("sc://server:15002/;token=team-token")
        mock_session.builder.remote.return_value.config.assert_not_called()
        mock_session.builder.remote.return_value.getOrCreate.assert_called_once_with()


def test_connect_keeps_existing_url_parameters():
    """Test that URLs already carrying parameters keep them next to the token."""

    with (
        patch("kubeflow.spark.api.spark_client.KubernetesBackend"),
        patch("kubeflow.spark.api.spark_client.SparkSession") as mock_session,
    ):
        client = SparkClient()

        client.connect(base_url="sc://server:15002/;use_ssl=true", token="team-token")

        mock_session.builder.remote.assert_called_once_with(
            "sc://server:15002/;use_ssl=true;token=team-token"
        )


def test_connect_quotes_a_token_with_delimiters():
    """Test that a token keeps its meaning through PySpark's URL decoding."""

    token = "team token;session=1&=x"

    with (
        patch("kubeflow.spark.api.spark_client.KubernetesBackend"),
        patch("kubeflow.spark.api.spark_client.SparkSession") as mock_session,
    ):
        client = SparkClient()

        client.connect(base_url="sc://server:15002", token=token)

        composed = mock_session.builder.remote.call_args.args[0]

    assert composed == "sc://server:15002/;token=team%20token%3Bsession%3D1%26%3Dx"

    builder = DefaultChannelBuilder(composed)

    assert builder.endpoint == "server:15002"
    assert builder.token == token
