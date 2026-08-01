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
                    backend_config=KubernetesBackendConfig(
                        namespace=test_case.config["namespace"],
                    )
                )
                mock.assert_called_once()
        elif "backend_config" in test_case.config:
            SparkClient(
                backend_config=test_case.config["backend_config"],
            )
        else:
            with patch("kubeflow.spark.api.spark_client.KubernetesBackend"):
                client = SparkClient()
                assert client.backend is not None

        # If we reach here but expected an exception, fail.
        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )
    except Exception as e:
        # If we got an exception but expected success, fail.
        assert test_case.expected_status == FAILED, f"Unexpected exception in {test_case.name}: {e}"

        # Validate the exception type if specified.
        if test_case.expected_error:
            assert isinstance(e, test_case.expected_error), (
                f"Expected exception type "
                f"'{test_case.expected_error.__name__}' but got "
                f"'{type(e).__name__}: {str(e)}'"
            )


@pytest.mark.parametrize(
    "job,backend_error,expected_error",
    [
        (
            "not-a-job",
            TypeError(
                "job must be an instance of FileJob or FuncJob.",
            ),
            TypeError,
        ),
        (
            FileJob(file_source=""),
            ValueError(
                "`job.file_source` must be a non-empty string.",
            ),
            ValueError,
        ),
    ],
)
def test_submit_job_validation(
    job,
    backend_error,
    expected_error,
):
    """Test SparkClient submit_job validation."""

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        backend.submit_job.side_effect = backend_error

        client = SparkClient()

        with pytest.raises(expected_error):
            client.submit_job(job=job)

        backend.submit_job.assert_called_once_with(
            job=job,
            num_executors=None,
            resources_per_executor=None,
            spark_conf=None,
        )


def test_submit_job_rejects_options():
    """Test that unsupported batch job options are rejected."""

    job = FileJob(
        file_source="s3://bucket/job.py",
    )

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        with pytest.raises(
            NotImplementedError,
            match="options support is not yet implemented",
        ):
            client.submit_job(
                job=job,
                options=[object()],
            )

        backend.submit_job.assert_not_called()


@pytest.mark.parametrize(
    "job",
    [
        FileJob(
            file_source="s3://bucket/job.py",
        ),
        FuncJob(
            func=lambda: None,
        ),
    ],
)
@pytest.mark.parametrize(
    "spark_conf",
    [
        None,
        {},
        {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.shuffle.partitions": "200",
        },
    ],
)
def test_submit_job_success(
    job,
    spark_conf,
):
    """Test successful submit_job."""

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value

        backend.submit_job.return_value = SparkJob(
            name="spark-job-123",
            namespace="default",
        )

        client = SparkClient()

        name = client.submit_job(
            job=job,
            spark_conf=spark_conf,
        )

        assert name == "spark-job-123"

        backend.submit_job.assert_called_once_with(
            job=job,
            num_executors=None,
            resources_per_executor=None,
            spark_conf=spark_conf,
        )
