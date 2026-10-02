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
    SparkJobStatus,
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
    print(f"Testing {test_case.name}...")

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

        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )
    except Exception as e:
        assert test_case.expected_status == FAILED, f"Unexpected exception in {test_case.name}: {e}"
        if test_case.expected_error:
            assert isinstance(e, test_case.expected_error), (
                f"Expected exception type '{test_case.expected_error.__name__}' but got '{type(e).__name__}: {str(e)}'"
            )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="submit job invalid job type raises TypeError",
            expected_status=FAILED,
            config={"job": "not-a-job"},
            expected_error=TypeError,
        ),
        TestCase(
            name="submit job empty file source raises ValueError",
            expected_status=FAILED,
            config={"job": FileJob(file_source="")},
            expected_error=ValueError,
        ),
        TestCase(
            name="submit job invalid spark conf raises ValueError",
            expected_status=FAILED,
            config={"job": FileJob(file_source="s3://bucket/job.py"), "spark_conf": []},
            expected_error=ValueError,
        ),
        TestCase(
            name="file job with default arguments",
            expected_status=SUCCESS,
            config={
                "job": FileJob(file_source="s3://bucket/job.py"),
                "options": None,
                "num_executors": None,
                "resources_per_executor": None,
                "spark_conf": None,
            },
            expected_output="spark-job-123",
        ),
        TestCase(
            name="file job with options",
            expected_status=SUCCESS,
            config={
                "job": FileJob(file_source="s3://bucket/job.py"),
                "options": [Labels({"team": "ml"})],
                "num_executors": None,
                "resources_per_executor": None,
                "spark_conf": None,
            },
            expected_output="spark-job-123",
        ),
        TestCase(
            name="func job with default arguments",
            expected_status=SUCCESS,
            config={
                "job": FuncJob(func=lambda: None),
                "options": None,
                "num_executors": None,
                "resources_per_executor": None,
                "spark_conf": None,
            },
            expected_output="spark-job-123",
        ),
        TestCase(
            name="job with all executor resources and spark configuration set",
            expected_status=SUCCESS,
            config={
                "job": FileJob(file_source="s3://bucket/job.py"),
                "options": [Labels({"env": "prod"})],
                "num_executors": 3,
                "resources_per_executor": {"cpu": "2", "memory": "4Gi"},
                "spark_conf": {"spark.executor.memory": "4g"},
            },
            expected_output="spark-job-all-args",
        ),
    ],
)
def test_submit_job(test_case: TestCase):
    """Test SparkClient submit_job validation and delegation."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        kwargs = {"job": test_case.config["job"]}
        if "options" in test_case.config and test_case.config["options"] is not None:
            kwargs["options"] = test_case.config["options"]
        if "num_executors" in test_case.config and test_case.config["num_executors"] is not None:
            kwargs["num_executors"] = test_case.config["num_executors"]
        if (
            "resources_per_executor" in test_case.config
            and test_case.config["resources_per_executor"] is not None
        ):
            kwargs["resources_per_executor"] = test_case.config["resources_per_executor"]
        if "spark_conf" in test_case.config and test_case.config["spark_conf"] is not None:
            kwargs["spark_conf"] = test_case.config["spark_conf"]

        if test_case.expected_status == FAILED:
            backend.submit_job.side_effect = test_case.expected_error("Validation error")
            with pytest.raises(test_case.expected_error):
                client.submit_job(**kwargs)
            backend.submit_job.assert_called_once()
        else:
            backend.submit_job.return_value = SparkJob(
                name=test_case.expected_output,
                namespace="default",
            )
            name = client.submit_job(**kwargs)

            assert name == test_case.expected_output
            backend.submit_job.assert_called_once_with(
                job=test_case.config["job"],
                num_executors=test_case.config.get("num_executors"),
                resources_per_executor=test_case.config.get("resources_per_executor"),
                spark_conf=test_case.config.get("spark_conf"),
                options=test_case.config.get("options"),
            )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get job by name",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123"},
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="get job not found raises error",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job"},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_job(test_case: TestCase):
    """Test get_job delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        if test_case.expected_status == FAILED:
            backend.get_job.side_effect = test_case.expected_error("Job not found")
            with pytest.raises(test_case.expected_error):
                client.get_job(name=test_case.config["job_name"])
            backend.get_job.assert_called_once_with(test_case.config["job_name"])
        else:
            backend.get_job.return_value = test_case.expected_output
            job = client.get_job(name=test_case.config["job_name"])

            assert job == test_case.expected_output
            backend.get_job.assert_called_once_with(test_case.config["job_name"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list all jobs",
            expected_status=SUCCESS,
            config={"status": None},
            expected_output=[
                SparkJob(name="job-1", namespace="default"),
                SparkJob(name="job-2", namespace="default"),
            ],
        ),
        TestCase(
            name="list jobs filtered by single status set",
            expected_status=SUCCESS,
            config={"status": {SparkJobStatus.RUNNING}},
            expected_output=[
                SparkJob(name="job-1", namespace="default"),
            ],
        ),
        TestCase(
            name="list jobs filtered by multiple status set",
            expected_status=SUCCESS,
            config={"status": {SparkJobStatus.COMPLETED, SparkJobStatus.FAILED}},
            expected_output=[
                SparkJob(name="job-2", namespace="default"),
            ],
        ),
        TestCase(
            name="list jobs backend failure raises error",
            expected_status=FAILED,
            config={"status": None},
            expected_error=RuntimeError,
        ),
    ],
)
def test_list_jobs(test_case: TestCase):
    """Test list_jobs delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        if test_case.expected_status == FAILED:
            backend.list_jobs.side_effect = test_case.expected_error("Failed to list jobs")
            with pytest.raises(test_case.expected_error):
                client.list_jobs(status=test_case.config["status"])
            backend.list_jobs.assert_called_once_with(status=test_case.config["status"])
        else:
            backend.list_jobs.return_value = test_case.expected_output
            jobs = client.list_jobs(status=test_case.config["status"])

            assert jobs == test_case.expected_output
            backend.list_jobs.assert_called_once_with(status=test_case.config["status"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete job by name",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123"},
        ),
        TestCase(
            name="delete nonexistent job raises error",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job"},
            expected_error=RuntimeError,
        ),
    ],
)
def test_delete_job(test_case: TestCase):
    """Test delete_job delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        if test_case.expected_status == FAILED:
            backend.delete_job.side_effect = test_case.expected_error("Job not found")
            with pytest.raises(test_case.expected_error):
                client.delete_job(name=test_case.config["job_name"])
            backend.delete_job.assert_called_once_with(test_case.config["job_name"])
        else:
            client.delete_job(name=test_case.config["job_name"])
            backend.delete_job.assert_called_once_with(test_case.config["job_name"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default status invocation",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123"},
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="default timeout invocation",
            expected_status=SUCCESS,
            config={
                "job_name": "spark-job-123",
                "status": {SparkJobStatus.RUNNING},
                "polling_interval": 5,
            },
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="default polling interval",
            expected_status=SUCCESS,
            config={
                "job_name": "spark-job-123",
                "status": {SparkJobStatus.RUNNING},
                "timeout": 120,
            },
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="explicit polling interval",
            expected_status=SUCCESS,
            config={
                "job_name": "spark-job-123",
                "status": {SparkJobStatus.RUNNING},
                "timeout": 60,
                "polling_interval": 3,
            },
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="single status {RUNNING}",
            expected_status=SUCCESS,
            config={
                "job_name": "spark-job-123",
                "status": {SparkJobStatus.RUNNING},
                "timeout": 60,
            },
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="multiple statuses {COMPLETED, FAILED}",
            expected_status=SUCCESS,
            config={
                "job_name": "spark-job-123",
                "status": {SparkJobStatus.COMPLETED, SparkJobStatus.FAILED},
                "timeout": 60,
            },
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="invalid timeout < 0",
            expected_status=FAILED,
            config={
                "job_name": "spark-job-123",
                "timeout": -1,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="timeout = 0",
            expected_status=FAILED,
            config={
                "job_name": "spark-job-123",
                "timeout": 0,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="polling_interval <= 0",
            expected_status=FAILED,
            config={
                "job_name": "spark-job-123",
                "polling_interval": 0,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="polling_interval >= timeout",
            expected_status=FAILED,
            config={
                "job_name": "spark-job-123",
                "timeout": 10,
                "polling_interval": 10,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="successful wait",
            expected_status=SUCCESS,
            config={
                "job_name": "spark-job-123",
                "status": {SparkJobStatus.COMPLETED},
                "timeout": 30,
                "polling_interval": 2,
            },
            expected_output=SparkJob(name="spark-job-123", namespace="default"),
        ),
        TestCase(
            name="backend timeout",
            expected_status=FAILED,
            config={
                "job_name": "spark-job-123",
                "status": {SparkJobStatus.COMPLETED},
                "timeout": 30,
            },
            expected_error=RuntimeError,
        ),
    ],
)
def test_wait_for_job_status(test_case: TestCase):
    """Test wait_for_job_status delegation and validation."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        client = SparkClient()

        kwargs = {"name": test_case.config["job_name"]}
        if "status" in test_case.config:
            kwargs["status"] = test_case.config["status"]
        if "timeout" in test_case.config:
            kwargs["timeout"] = test_case.config["timeout"]
        if "polling_interval" in test_case.config:
            kwargs["polling_interval"] = test_case.config["polling_interval"]

        if test_case.expected_status == FAILED:
            if test_case.expected_error is ValueError:
                with pytest.raises(ValueError):
                    client.wait_for_job_status(**kwargs)
                backend.wait_for_job_status.assert_not_called()
            else:
                backend.wait_for_job_status.side_effect = test_case.expected_error(
                    "Timeout waiting for job"
                )
                with pytest.raises(test_case.expected_error):
                    client.wait_for_job_status(**kwargs)
                backend.wait_for_job_status.assert_called_once()
        else:
            backend.wait_for_job_status.return_value = test_case.expected_output
            result = client.wait_for_job_status(**kwargs)

            assert result == test_case.expected_output

            call_kwargs = backend.wait_for_job_status.call_args.kwargs
            assert call_kwargs["name"] == test_case.config["job_name"]
            if "status" in test_case.config:
                assert call_kwargs["status"] == test_case.config["status"]
            if "timeout" in test_case.config:
                assert call_kwargs["timeout"] == test_case.config["timeout"]
            if "polling_interval" in test_case.config:
                assert call_kwargs["polling_interval"] == test_case.config["polling_interval"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get job logs with default follow invocation",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123"},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get job logs explicit non-follow",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123", "follow": False},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get job logs explicit follow",
            expected_status=SUCCESS,
            config={"job_name": "spark-job-123", "follow": True},
            expected_output=["log stream 1", "log stream 2"],
        ),
    ],
)
def test_get_job_logs(test_case: TestCase):
    """Test get_job_logs delegation to KubernetesBackend."""
    print(f"Testing {test_case.name}...")

    with patch("kubeflow.spark.api.spark_client.KubernetesBackend") as mock_backend:
        backend = mock_backend.return_value
        backend.get_job_logs.return_value = iter(test_case.expected_output)

        client = SparkClient()

        if "follow" in test_case.config:
            logs = list(
                client.get_job_logs(
                    name=test_case.config["job_name"],
                    follow=test_case.config["follow"],
                )
            )
            backend.get_job_logs.assert_called_once_with(
                name=test_case.config["job_name"],
                follow=test_case.config["follow"],
            )
        else:
            logs = list(client.get_job_logs(name=test_case.config["job_name"]))
            backend.get_job_logs.assert_called_once_with(
                name=test_case.config["job_name"],
                follow=False,
            )

        assert logs == test_case.expected_output
