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

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid datacacheinitializer creation",
            expected_status=SUCCESS,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_output=None,
        ),
        TestCase(
            name="invalid num_data_nodes raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": 1,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="zero num_data_nodes raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": 0,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="negative num_data_nodes raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": -1,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri without cache:// prefix raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "invalid://test_schema/test_table",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri format raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri with too many parts raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table/extra",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
    ],
)
def test_data_cache_initializer(test_case: TestCase):
    """Test DataCacheInitializer creation and validation."""
    print("Executing test:", test_case.name)

    try:
        initializer = types.DataCacheInitializer(
            storage_uri=test_case.config["storage_uri"],
            num_data_nodes=test_case.config["num_data_nodes"],
            metadata_loc=test_case.config["metadata_loc"],
        )

        assert test_case.expected_status == SUCCESS
        # Only check the fields that were passed in config, not auto-generated ones
        for key in test_case.config:
            assert getattr(initializer, key) == test_case.config[key]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="all steps running",
            expected_status=SUCCESS,
            config={
                "steps": [
                    types.Step(
                        name="step-1",
                        status="Running",
                        pod_name="pod-1",
                    ),
                    types.Step(
                        name="step-2",
                        status="Running",
                        pod_name="pod-2",
                    ),
                ],
                "job_status": "Running",
            },
            expected_output={
                "overall_status": "Running",
                "total_steps": 2,
                "completed_steps": 0,
                "running_steps": ["step-1", "step-2"],
                "failed_steps": [],
                "healthy_pods": 2,
                "total_pods": 2,
                "completion_percentage": 0.0,
            },
        ),
        TestCase(
            name="some steps completed and some running",
            expected_status=SUCCESS,
            config={
                "steps": [
                    types.Step(
                        name="step-1",
                        status="Succeeded",
                        pod_name="pod-1",
                    ),
                    types.Step(
                        name="step-2",
                        status="Running",
                        pod_name="pod-2",
                    ),
                ],
                "job_status": "Running",
            },
            expected_output={
                "overall_status": "Running",
                "total_steps": 2,
                "completed_steps": 1,
                "running_steps": ["step-2"],
                "failed_steps": [],
                "healthy_pods": 2,
                "total_pods": 2,
                "completion_percentage": 50.0,
            },
        ),
        TestCase(
            name="all steps completed",
            expected_status=SUCCESS,
            config={
                "steps": [
                    types.Step(
                        name="step-1",
                        status="Succeeded",
                        pod_name="pod-1",
                    ),
                    types.Step(
                        name="step-2",
                        status="Complete",
                        pod_name="pod-2",
                    ),
                ],
                "job_status": "Complete",
            },
            expected_output={
                "overall_status": "Complete",
                "total_steps": 2,
                "completed_steps": 2,
                "running_steps": [],
                "failed_steps": [],
                "healthy_pods": 2,
                "total_pods": 2,
                "completion_percentage": 100.0,
            },
        ),
        TestCase(
            name="some steps failed",
            expected_status=SUCCESS,
            config={
                "steps": [
                    types.Step(
                        name="step-1",
                        status="Succeeded",
                        pod_name="pod-1",
                    ),
                    types.Step(
                        name="step-2",
                        status="Failed",
                        pod_name="pod-2",
                    ),
                ],
                "job_status": "Failed",
            },
            expected_output={
                "overall_status": "Failed",
                "total_steps": 2,
                "completed_steps": 1,
                "running_steps": [],
                "failed_steps": ["step-2"],
                "healthy_pods": 1,
                "total_pods": 2,
                "completion_percentage": 50.0,
            },
        ),
        TestCase(
            name="no steps",
            expected_status=SUCCESS,
            config={
                "steps": [],
                "job_status": "Created",
            },
            expected_output={
                "overall_status": "Created",
                "total_steps": 0,
                "completed_steps": 0,
                "running_steps": [],
                "failed_steps": [],
                "healthy_pods": 0,
                "total_pods": 0,
                "completion_percentage": 0.0,
            },
        ),
    ],
)
def test_job_progress_from_job(test_case: TestCase):
    """Test JobProgress creation from TrainJob."""
    print("Executing test:", test_case.name)

    try:
        # Create a minimal TrainJob for testing
        runtime = types.Runtime(
            name="test-runtime",
            trainer=types.RuntimeTrainer(
                trainer_type=types.TrainerType.CUSTOM_TRAINER,
                framework="pytorch",
                image="pytorch:latest",
            ),
        )

        job = types.TrainJob(
            name="test-job",
            runtime=runtime,
            steps=test_case.config["steps"],
            num_nodes=len(test_case.config["steps"]),
            creation_timestamp=__import__("datetime").datetime.now(),
            status=test_case.config["job_status"],
        )

        progress = types.JobProgress.from_job(job)

        assert test_case.expected_status == SUCCESS
        expected = test_case.expected_output

        assert progress.job_name == "test-job"
        assert progress.overall_status == expected["overall_status"]
        assert progress.total_steps == expected["total_steps"]
        assert progress.completed_steps == expected["completed_steps"]
        assert progress.running_steps == expected["running_steps"]
        assert progress.failed_steps == expected["failed_steps"]
        assert progress.healthy_pods == expected["healthy_pods"]
        assert progress.total_pods == expected["total_pods"]
        assert progress.completion_percentage == expected["completion_percentage"]

    except Exception:
        assert test_case.expected_status == FAILED
        raise
    print("test execution complete")


def test_job_progress_string_representation():
    """Test JobProgress string representation."""
    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch:latest",
        ),
    )

    job = types.TrainJob(
        name="test-job",
        runtime=runtime,
        steps=[
            types.Step(
                name="step-1",
                status="Succeeded",
                pod_name="pod-1",
            ),
            types.Step(
                name="step-2",
                status="Running",
                pod_name="pod-2",
            ),
        ],
        num_nodes=2,
        creation_timestamp=__import__("datetime").datetime.now(),
        status="Running",
    )

    progress = types.JobProgress.from_job(job)
    progress_str = str(progress)

    # Verify the string representation contains expected information
    assert "test-job" in progress_str
    assert "Running" in progress_str
    assert "50.0%" in progress_str
    assert "1/2 steps" in progress_str
    assert "2/2 healthy" in progress_str
    assert "Running steps: step-2" in progress_str
