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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get job by name",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123"},
        ),
    ],
)
def test_get_job(test_case: TestCase):
    """Test get_job delegation to KubernetesBackend."""
    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        expected_job = SparkJob(
            name=test_case.config["job_name"],
            namespace="default",
        )
        backend.get_job.return_value = expected_job

        client = SparkClient()
        job = client.get_job(name=test_case.config["job_name"])

        assert job == expected_job
        backend.get_job.assert_called_once_with(test_case.config["job_name"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list all jobs",
            expected_status=SUCCESS,
            config={"status": None},
        ),
        TestCase(
            name="list jobs filtered by status",
            expected_status=SUCCESS,
            config={"status": "Running"},
        ),
    ],
)
def test_list_jobs(test_case: TestCase):
    """Test list_jobs delegation to KubernetesBackend."""
    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        mock_jobs = [
            SparkJob(name="job-1", namespace="default"),
            SparkJob(name="job-2", namespace="default"),
        ]
        backend.list_jobs.return_value = mock_jobs

        client = SparkClient()
        jobs = client.list_jobs(status=test_case.config["status"])

        assert jobs == mock_jobs
        backend.list_jobs.assert_called_once_with(status=test_case.config["status"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete job by name",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123"},
        ),
    ],
)
def test_delete_job(test_case: TestCase):
    """Test delete_job delegation to KubernetesBackend."""
    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value

        client = SparkClient()
        client.delete_job(name=test_case.config["job_name"])

        backend.delete_job.assert_called_once_with(test_case.config["job_name"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait for job status success",
            expected_status=SUCCESS,
            config={
                "job_name": "spark-job-123",
                "status": "Completed",
                "timeout": 60,
                "polling_interval": 2,
            },
        ),
    ],
)
def test_wait_for_job_status(test_case: TestCase):
    """Test wait_for_job_status delegation to KubernetesBackend."""
    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        backend.wait_for_job_status.return_value = True

        client = SparkClient()
        result = client.wait_for_job_status(
            name=test_case.config["job_name"],
            status=test_case.config["status"],
            timeout=test_case.config["timeout"],
        )

        assert result is True
        backend.wait_for_job_status.assert_called_once_with(
            name=test_case.config["job_name"],
            status=test_case.config["status"],
            timeout=test_case.config["timeout"],
            polling_interval=test_case.config["polling_interval"],
        )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get job logs non-follow",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123", "follow": False},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get job logs follow",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123", "follow": True},
            expected_output=["log stream 1", "log stream 2"],
        ),
    ],
)
def test_get_job_logs(test_case: TestCase):
    """Test get_job_logs delegation to KubernetesBackend."""
    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        backend.get_job_logs.return_value = iter(test_case.expected_output)

        client = SparkClient()
        logs = list(
            client.get_job_logs(
                name=test_case.config["job_name"],
                follow=test_case.config["follow"],
            )
        )

        assert logs == test_case.expected_output
        backend.get_job_logs.assert_called_once_with(
            name=test_case.config["job_name"],
            follow=test_case.config["follow"],
        )
