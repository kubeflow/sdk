# Copyright The Kubeflow Authors.
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

"""Unit tests for SparkClient integration in Kubeflow Pipelines."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

pytest.importorskip("kfp")

from kubeflow.spark.types.types import SparkJobStatus


def _load_spark_pipeline_module():
    """Dynamically load the spark_pipeline example module."""
    example_path = (
        Path(__file__).resolve().parent.parent.parent / "examples" / "spark" / "spark_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("spark_pipeline", example_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load example from {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline_module():
    """Fixture providing the loaded spark_pipeline module."""
    return _load_spark_pipeline_module()


def test_spark_pipeline_compilation(pipeline_module, tmp_path):
    """Verify that the Spark KFP pipeline compiles into valid KFP v2 IR YAML."""
    output_yaml = tmp_path / "compiled_pipeline.yaml"
    compiled_path = pipeline_module.compile_pipeline(str(output_yaml))

    assert Path(compiled_path).exists()
    assert Path(compiled_path).stat().st_size > 0

    with open(compiled_path) as f:
        pipeline_spec = yaml.safe_load(f)

    assert "pipelineInfo" in pipeline_spec
    assert pipeline_spec["pipelineInfo"]["name"] == "spark-batch-pipeline"

    components = pipeline_spec.get("components", {})
    assert "comp-submit-spark-job" in components
    assert "comp-inspect-and-cleanup-spark-job" in components

    root_tasks = pipeline_spec.get("root", {}).get("dag", {}).get("tasks", {})
    assert "submit-spark-job" in root_tasks
    assert "inspect-and-cleanup-spark-job" in root_tasks

    cleanup_task = root_tasks["inspect-and-cleanup-spark-job"]
    assert "dependentTasks" in cleanup_task
    assert "submit-spark-job" in cleanup_task["dependentTasks"]


def test_submit_spark_job_logic(pipeline_module):
    """Verify submit_spark_job component logic with mocked SparkClient."""
    mock_client = MagicMock()
    mock_client.submit_job.return_value = "spark-test-job-1234"
    mock_job = MagicMock()
    mock_job.status = SparkJobStatus.COMPLETED
    mock_client.wait_for_job_status.return_value = mock_job

    with patch("kubeflow.spark.SparkClient", return_value=mock_client):
        job_name = pipeline_module.submit_spark_job.python_func(
            file_source="https://example.com/script.py",
            namespace="spark-test-ns",
            num_executors=2,
            executor_cpu="2",
            executor_memory="1Gi",
            timeout=300,
        )

    assert job_name == "spark-test-job-1234"
    assert mock_client.submit_job.called
    submit_kwargs = mock_client.submit_job.call_args.kwargs
    assert submit_kwargs["job"].file_source == "https://example.com/script.py"
    assert submit_kwargs["num_executors"] == 2
    assert submit_kwargs["resources_per_executor"] == {"cpu": "2", "memory": "1Gi"}

    assert mock_client.wait_for_job_status.called
    wait_kwargs = mock_client.wait_for_job_status.call_args.kwargs
    assert wait_kwargs["name"] == "spark-test-job-1234"
    assert wait_kwargs["status"] == {SparkJobStatus.COMPLETED}
    assert wait_kwargs["timeout"] == 300


def test_inspect_and_cleanup_spark_job_logic(pipeline_module):
    """Verify inspect_and_cleanup_spark_job component logic with mocked SparkClient."""
    mock_client = MagicMock()
    mock_client.get_job_logs.return_value = iter(
        ["Starting Spark driver", "Running stage 0", "Job finished successfully"]
    )
    mock_client.delete_job.return_value = None

    with patch("kubeflow.spark.SparkClient", return_value=mock_client):
        summary = pipeline_module.inspect_and_cleanup_spark_job.python_func(
            job_name="spark-test-job-1234",
            namespace="spark-test-ns",
        )

    assert "Retrieved 3 log lines" in summary
    assert "spark-test-job-1234" in summary
    assert mock_client.get_job_logs.called
    assert mock_client.get_job_logs.call_args.kwargs["name"] == "spark-test-job-1234"

    assert mock_client.delete_job.called
    assert mock_client.delete_job.call_args.kwargs["name"] == "spark-test-job-1234"
