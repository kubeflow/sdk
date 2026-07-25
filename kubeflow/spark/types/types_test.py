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

"""Unit tests for Kubeflow Spark types."""

from datetime import datetime
from unittest.mock import patch

import pytest

from kubeflow.spark.types.types import (
    Driver,
    Executor,
    FileJob,
    FuncJob,
    SparkConnectInfo,
    SparkConnectState,
    SparkJob,
    SparkJobStatus,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "state, expected",
    [
        (SparkConnectState.PROVISIONING, "Provisioning"),
        (SparkConnectState.READY, "Ready"),
        (SparkConnectState.RUNNING, "Running"),
        (SparkConnectState.NOT_READY, "NotReady"),
        (SparkConnectState.FAILED, "Failed"),
    ],
)
def test_spark_connect_state_values(state, expected):
    """Test SparkConnectState enum values."""
    assert state == expected


@pytest.mark.parametrize(
    "state",
    [
        SparkConnectState.PROVISIONING,
        SparkConnectState.READY,
        SparkConnectState.RUNNING,
        SparkConnectState.NOT_READY,
        SparkConnectState.FAILED,
    ],
)
def test_spark_connect_state_is_string(state):
    """Test SparkConnectState inherits from str."""
    assert isinstance(state, str)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="spark connect info with required fields",
            expected_status=SUCCESS,
            config={
                "name": "test-session",
                "namespace": "default",
                "state": SparkConnectState.READY,
            },
        ),
        TestCase(
            name="spark connect info with all fields",
            expected_status=SUCCESS,
            config={
                "name": "full-session",
                "namespace": "spark-ns",
                "state": SparkConnectState.READY,
                "driver_pod_name": "spark-connect-server-0",
                "pod_ip": "10.0.0.5",
                "service_name": "spark-connect-svc",
                "creation_timestamp": datetime(2025, 1, 12, 10, 30, 0),
            },
        ),
    ],
)
def test_spark_connect_info(test_case: TestCase):
    """Test SparkConnectInfo creation."""

    print("Executing test:", test_case.name)

    info = SparkConnectInfo(**test_case.config)

    assert test_case.expected_status == SUCCESS

    for key, value in test_case.config.items():
        assert getattr(info, key) == value

    if test_case.name == "spark connect info with required fields":
        assert info.driver_pod_name is None
        assert info.pod_ip is None
        assert info.service_name is None
        assert info.creation_timestamp is None

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default driver",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="driver with resources",
            expected_status=SUCCESS,
            config={
                "resources": {
                    "cpu": "2",
                    "memory": "4Gi",
                },
            },
        ),
        TestCase(
            name="driver with gpu resources",
            expected_status=SUCCESS,
            config={
                "resources": {
                    "cpu": "4",
                    "memory": "8Gi",
                    "nvidia.com/gpu": "1",
                },
            },
        ),
        TestCase(
            name="driver with service account",
            expected_status=SUCCESS,
            config={
                "service_account": "spark-sa",
            },
        ),
        TestCase(
            name="kep107 driver example",
            expected_status=SUCCESS,
            config={
                "resources": {
                    "cpu": "4",
                    "memory": "8Gi",
                },
                "service_account": "spark-driver-prod",
            },
        ),
    ],
)
def test_driver(test_case: TestCase):
    """Test Driver creation."""

    print("Executing test:", test_case.name)

    driver = Driver(**test_case.config)

    assert test_case.expected_status == SUCCESS

    if not test_case.config:
        assert driver.image is None
        assert driver.resources is None
        assert driver.java_options is None
        assert driver.service_account is None
    else:
        for key, value in test_case.config.items():
            assert getattr(driver, key) == value

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default executor",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="executor with num instances",
            expected_status=SUCCESS,
            config={
                "num_instances": 5,
            },
        ),
        TestCase(
            name="executor with resources per executor",
            expected_status=SUCCESS,
            config={
                "num_instances": 3,
                "resources_per_executor": {
                    "cpu": "4",
                    "memory": "8Gi",
                },
            },
        ),
        TestCase(
            name="executor with gpu resources",
            expected_status=SUCCESS,
            config={
                "num_instances": 10,
                "resources_per_executor": {
                    "cpu": "8",
                    "memory": "32Gi",
                    "nvidia.com/gpu": "2",
                },
            },
        ),
        TestCase(
            name="kep107 executor example",
            expected_status=SUCCESS,
            config={
                "num_instances": 20,
                "resources_per_executor": {
                    "cpu": "8",
                    "memory": "32Gi",
                    "nvidia.com/gpu": "2",
                },
            },
        ),
    ],
)
def test_executor(test_case: TestCase):
    """Test Executor creation."""

    print("Executing test:", test_case.name)

    executor = Executor(**test_case.config)

    assert test_case.expected_status == SUCCESS

    if test_case.name == "default executor":
        assert executor.num_instances is None
        assert executor.resources_per_executor is None
        assert executor.java_options is None
    else:
        for key, value in test_case.config.items():
            assert getattr(executor, key) == value

    print("test execution complete")


@pytest.mark.parametrize(
    "status, expected",
    [
        (SparkJobStatus.CREATED, "Created"),
        (SparkJobStatus.RUNNING, "Running"),
        (SparkJobStatus.COMPLETED, "Completed"),
        (SparkJobStatus.FAILED, "Failed"),
    ],
)
def test_spark_job_status_values(status, expected):
    """Test SparkJobStatus enum values."""
    assert status == expected


@pytest.mark.parametrize(
    "status",
    [
        SparkJobStatus.CREATED,
        SparkJobStatus.RUNNING,
        SparkJobStatus.COMPLETED,
        SparkJobStatus.FAILED,
    ],
)
def test_spark_job_status_is_string(status):
    """Test SparkJobStatus inherits from str."""
    assert isinstance(status, str)


@pytest.mark.parametrize(
    "operator_state, expected_status",
    [
        (None, SparkJobStatus.CREATED),
        ("", SparkJobStatus.CREATED),
        ("SUBMITTED", SparkJobStatus.CREATED),
        ("RUNNING", SparkJobStatus.RUNNING),
        ("SUCCEEDING", SparkJobStatus.RUNNING),
        ("SUSPENDING", SparkJobStatus.RUNNING),
        ("SUSPENDED", SparkJobStatus.RUNNING),
        ("RESUMING", SparkJobStatus.RUNNING),
        ("COMPLETED", SparkJobStatus.COMPLETED),
        ("FAILED", SparkJobStatus.FAILED),
        ("SUBMISSION_FAILED", SparkJobStatus.FAILED),
        ("FAILING", SparkJobStatus.FAILED),
        ("PENDING_RERUN", SparkJobStatus.FAILED),
        ("INVALIDATING", SparkJobStatus.FAILED),
        ("UNKNOWN", SparkJobStatus.FAILED),
    ],
)
def test_from_operator_state(operator_state, expected_status):
    """Verify SparkApplication states map to SparkJobStatus."""
    assert SparkJobStatus.from_operator_state(operator_state) == expected_status


def test_unknown_operator_state():
    """Verify unknown SparkApplication states default to FAILED."""
    with patch("kubeflow.spark.types.types.logger") as mock_logger:
        status = SparkJobStatus.from_operator_state("SOME_NEW_STATE")

    assert status == SparkJobStatus.FAILED
    mock_logger.warning.assert_called_once_with(
        "Unknown SparkApplication state '%s'. Defaulting to FAILED.",
        "SOME_NEW_STATE",
    )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default spark job",
            expected_status=SUCCESS,
            config={
                "name": "test-job",
                "namespace": "default",
            },
        ),
        TestCase(
            name="spark job with all fields",
            expected_status=SUCCESS,
            config={
                "name": "full-job",
                "namespace": "spark-ns",
                "status": SparkJobStatus.RUNNING,
                "creation_timestamp": datetime(2025, 1, 12, 10, 30, 0),
                "num_executors": 10,
                "driver_pod_name": "driver-pod-1",
            },
        ),
    ],
)
def test_spark_job(test_case: TestCase):
    """Test SparkJob creation."""

    print("Executing test:", test_case.name)

    job = SparkJob(**test_case.config)

    assert test_case.expected_status == SUCCESS

    for key, value in test_case.config.items():
        assert getattr(job, key) == value

    if test_case.name == "default spark job":
        assert job.status is None
        assert job.creation_timestamp is None
        assert job.num_executors is None
        assert job.driver_pod_name is None

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default file job",
            expected_status=SUCCESS,
            config={
                "file_source": "s3://bucket/job.py",
            },
        ),
        TestCase(
            name="file job with all fields",
            expected_status=SUCCESS,
            config={
                "file_source": "local:///opt/spark/app.py",
                "args": ["--date", "2026-06-30"],
            },
        ),
    ],
)
def test_file_job(test_case: TestCase):
    """Test FileJob creation."""

    print("Executing test:", test_case.name)

    job = FileJob(**test_case.config)

    assert test_case.expected_status == SUCCESS

    for key, value in test_case.config.items():
        assert getattr(job, key) == value

    if test_case.name == "default file job":
        assert job.args is None

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default func job",
            expected_status=SUCCESS,
            config={
                "func": lambda: "ok",
            },
        ),
        TestCase(
            name="func job with arguments",
            expected_status=SUCCESS,
            config={
                "func": lambda x, y: x + y,
                "func_args": {
                    "x": 1,
                    "y": 2,
                },
            },
        ),
    ],
)
def test_func_job(test_case: TestCase):
    """Test FuncJob creation."""

    print("Executing test:", test_case.name)

    job = FuncJob(**test_case.config)

    assert test_case.expected_status == SUCCESS

    for key, value in test_case.config.items():
        assert getattr(job, key) == value

    if test_case.name == "default func job":
        assert job.func_args is None

    print("test execution complete")
