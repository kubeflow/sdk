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

from unittest.mock import Mock

import pytest

from kubeflow.test.common import FAILED, SUCCESS, TestCase
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
            name="valid storage_uri with user and model",
            expected_status=SUCCESS,
            config={"storage_uri": "hf://user/model"},
        ),
        TestCase(
            name="valid storage_uri with unscoped model",
            expected_status=SUCCESS,
            config={"storage_uri": "hf://model"},
        ),
    ],
)
def test_hugging_face_model_initializer(test_case: TestCase):
    """Test HuggingFaceModelInitializer creation."""
    print("Executing test:", test_case.name)

    try:
        initializer = types.HuggingFaceModelInitializer(
            storage_uri=test_case.config["storage_uri"],
        )

        assert test_case.expected_status == SUCCESS
        assert initializer.storage_uri == test_case.config["storage_uri"]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid storage_uri with user and dataset",
            expected_status=SUCCESS,
            config={"storage_uri": "hf://user/dataset"},
        ),
        TestCase(
            name="valid storage_uri with unscoped dataset",
            expected_status=SUCCESS,
            config={"storage_uri": "hf://dataset"},
        ),
    ],
)
def test_hugging_face_dataset_initializer(test_case: TestCase):
    """Test HuggingFaceDatasetInitializer creation."""
    print("Executing test:", test_case.name)

    try:
        initializer = types.HuggingFaceDatasetInitializer(
            storage_uri=test_case.config["storage_uri"],
        )

        assert test_case.expected_status == SUCCESS
        assert initializer.storage_uri == test_case.config["storage_uri"]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_exec_result_str_returns_stdout():
    """Test that printing an ExecResult prints the standard output."""
    result = types.ExecResult(stdout="hello\n", stderr="warning\n", returncode=0)

    assert str(result) == "hello\n"


def test_train_job_session_execute_delegates_to_backend():
    """Test that TrainJobSession.execute passes the command to the backend."""
    backend = Mock()
    backend.exec_command.return_value = types.ExecResult(stdout="out", stderr="", returncode=0)
    session = types.TrainJobSession(name="interactive-job", backend=backend)

    result = session.execute(["echo", "hello"], container="trainer")

    assert session.name == "interactive-job"
    assert result.stdout == "out"
    backend.exec_command.assert_called_once_with(
        "interactive-job", ["echo", "hello"], container="trainer"
    )


def test_train_job_session_close_deletes_job_and_waits():
    """Test that TrainJobSession.close deletes the TrainJob and waits for its Pods."""
    backend = Mock()
    session = types.TrainJobSession(name="interactive-job", backend=backend)

    session.close(timeout=42)

    backend.delete_job.assert_called_once_with("interactive-job")
    backend.wait_for_job_deletion.assert_called_once_with("interactive-job", timeout=42)


def test_train_job_session_close_is_idempotent():
    """Test that closing a session twice deletes the TrainJob only once."""
    backend = Mock()
    session = types.TrainJobSession(name="interactive-job", backend=backend)

    session.close()
    session.close()

    backend.delete_job.assert_called_once_with("interactive-job")


def test_train_job_session_context_manager_closes():
    """Test that leaving the context manager closes the session."""
    backend = Mock()

    with types.TrainJobSession(name="interactive-job", backend=backend) as session:
        assert isinstance(session, types.TrainJobSession)

    backend.delete_job.assert_called_once_with("interactive-job")
    backend.wait_for_job_deletion.assert_called_once_with("interactive-job", timeout=180)


def test_train_job_session_execute_after_close_raises():
    """Test that a closed session can't execute commands."""
    backend = Mock()
    session = types.TrainJobSession(name="interactive-job", backend=backend)
    session.close()

    with pytest.raises(RuntimeError):
        session.execute(["echo", "hello"])

    backend.exec_command.assert_not_called()
