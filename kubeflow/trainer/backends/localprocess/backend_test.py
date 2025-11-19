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

"""
Unit tests for the LocalProcessBackend class in the Kubeflow Trainer SDK.
"""

from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.backends.localprocess.backend import LocalProcessBackend
from kubeflow.trainer.backends.localprocess.constants import LOCAL_RUNTIME_IMAGE
from kubeflow.trainer.backends.localprocess.types import (
    LocalProcessBackendConfig,
    LocalRuntimeTrainer,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.options import (
    Annotations,
    Labels,
    Name,
    PodTemplateOverride,
    PodTemplateOverrides,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types

# Test constants
TORCH_RUNTIME = constants.TORCH_RUNTIME
BASIC_TRAIN_JOB_NAME = "test-job"


def dummy_training_function():
    """Dummy training function for testing."""
    print("Training started")
    return {"loss": 0.5, "accuracy": 0.95}


@pytest.fixture
def local_backend():
    """Create LocalProcessBackend for testing."""
    cfg = LocalProcessBackendConfig()
    backend = LocalProcessBackend(cfg)
    yield backend
    # Cleanup: Clear jobs to prevent test pollution
    backend._LocalProcessBackend__local_jobs.clear()


@pytest.fixture
def mock_train_environment():
    """Mock the training environment to avoid actual subprocess execution."""
    with (
        patch("kubeflow.trainer.backends.localprocess.job.LocalJob.start") as mock_start,
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_local_runtime_trainer"
        ) as mock_get_trainer,
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_local_train_job_script"
        ) as mock_get_script,
        patch("tempfile.mkdtemp") as mock_mkdtemp,
    ):
        # Setup mock return values
        mock_mkdtemp.return_value = "/tmp/test-venv"
        mock_get_script.return_value = ["/bin/bash", "-c", "echo 'training'"]

        mock_trainer = LocalRuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            num_nodes=1,
            device_count="1",
            device="cpu",
            packages=["torch"],
            image=LOCAL_RUNTIME_IMAGE,
        )
        mock_trainer.set_command = Mock()
        mock_get_trainer.return_value = mock_trainer

        yield {
            "start": mock_start,
            "get_trainer": mock_get_trainer,
            "get_script": mock_get_script,
            "mkdtemp": mock_mkdtemp,
        }


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list_all_local_runtimes",
            expected_status=SUCCESS,
            config={},
        ),
    ],
)
def test_list_runtimes(local_backend, test_case):
    """Test LocalProcessBackend.list_runtimes()."""
    runtimes = local_backend.list_runtimes()
    assert len(runtimes) > 0
    assert all(isinstance(rt, types.Runtime) for rt in runtimes)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_existing_runtime",
            expected_status=SUCCESS,
            config={"runtime_name": TORCH_RUNTIME},
        ),
        TestCase(
            name="get_nonexistent_runtime",
            expected_status=FAILED,
            config={"runtime_name": "nonexistent-runtime"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_runtime(local_backend, test_case):
    """Test LocalProcessBackend.get_runtime()."""
    runtime_name = test_case.config.get("runtime_name")

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            local_backend.get_runtime(runtime_name)
    else:
        runtime = local_backend.get_runtime(runtime_name)
        assert runtime is not None
        assert runtime.name == runtime_name


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_packages_for_existing_runtime",
            expected_status=SUCCESS,
            config={
                "runtime": types.Runtime(
                    name=TORCH_RUNTIME,
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
            },
        ),
        TestCase(
            name="get_packages_for_nonexistent_runtime",
            expected_status=FAILED,
            config={
                "runtime": types.Runtime(
                    name="nonexistent-runtime",
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_runtime_packages(local_backend, test_case):
    """Test LocalProcessBackend.get_runtime_packages()."""
    runtime = test_case.config.get("runtime")

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            local_backend.get_runtime_packages(runtime)
    else:
        packages = local_backend.get_runtime_packages(runtime)
        assert packages is not None
        assert isinstance(packages, list)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train with basic custom trainer - no options",
            expected_status=SUCCESS,
            config={
                "runtime": types.Runtime(
                    name=TORCH_RUNTIME,
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
                "trainer": types.CustomTrainer(
                    func=dummy_training_function,
                    packages_to_install=["numpy", "torch"],
                ),
                "options": [],
            },
        ),
        TestCase(
            name="train with custom trainer and environment variables",
            expected_status=SUCCESS,
            config={
                "runtime": types.Runtime(
                    name=TORCH_RUNTIME,
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
                "trainer": types.CustomTrainer(
                    func=dummy_training_function,
                    packages_to_install=["torch"],
                    env={"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4"},
                ),
                "options": [],
            },
        ),
        TestCase(
            name="train rejects kubernetes labels option",
            expected_status=FAILED,
            config={
                "runtime": types.Runtime(
                    name=TORCH_RUNTIME,
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
                "trainer": types.CustomTrainer(
                    func=dummy_training_function,
                ),
                "options": [Labels({"app": "test"})],
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="train rejects kubernetes annotations option",
            expected_status=FAILED,
            config={
                "runtime": types.Runtime(
                    name=TORCH_RUNTIME,
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
                "trainer": types.CustomTrainer(
                    func=dummy_training_function,
                ),
                "options": [Annotations({"description": "test"})],
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="train rejects pod template overrides option",
            expected_status=FAILED,
            config={
                "runtime": types.Runtime(
                    name=TORCH_RUNTIME,
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
                "trainer": types.CustomTrainer(
                    func=dummy_training_function,
                ),
                "options": [
                    PodTemplateOverrides(
                        PodTemplateOverride(
                            target_jobs=["node"],
                        )
                    )
                ],
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="train fails without runtime",
            expected_status=FAILED,
            config={
                "runtime": None,
                "trainer": types.CustomTrainer(
                    func=dummy_training_function,
                ),
                "options": [],
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="train fails without custom trainer",
            expected_status=FAILED,
            config={
                "runtime": types.Runtime(
                    name=TORCH_RUNTIME,
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image=LOCAL_RUNTIME_IMAGE,
                    ),
                ),
                "trainer": None,
            },
            expected_error=ValueError,
        ),
    ],
)
def test_train(local_backend, mock_train_environment, test_case):
    """Test LocalProcessBackend.train() with success and failure cases."""
    runtime = test_case.config.get("runtime")
    trainer = test_case.config.get("trainer")
    options = test_case.config.get("options", [])

    mocks = mock_train_environment

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error) as exc_info:
            local_backend.train(
                runtime=runtime,
                trainer=trainer,
                options=options,
            )

        # Verify specific error messages
        error_msg = str(exc_info.value)
        if "rejects kubernetes" in test_case.name:
            assert "not compatible with" in error_msg
        elif "without runtime" in test_case.name:
            assert "Runtime must be provided" in error_msg
        elif "without custom trainer" in test_case.name:
            assert "CustomTrainer must be set" in error_msg
    else:
        train_job_name = local_backend.train(
            runtime=runtime,
            trainer=trainer,
            options=options,
        )

        assert train_job_name is not None
        assert len(train_job_name) > 0
        mocks["start"].assert_called_once()
        mocks["get_trainer"].assert_called_once()
        mocks["get_script"].assert_called_once()

        # Verify job is tracked
        jobs = local_backend.list_jobs(runtime=runtime)
        assert any(job.name == train_job_name for job in jobs)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_nonexistent_job",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_job(local_backend, test_case):
    """Test LocalProcessBackend.get_job()."""
    job_name = test_case.config.get("job_name")

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            local_backend.get_job(job_name)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list_jobs_empty",
            expected_status=SUCCESS,
            config={"runtime": None},
        ),
    ],
)
def test_list_jobs(local_backend, test_case):
    """Test LocalProcessBackend.list_jobs()."""
    runtime = test_case.config.get("runtime")
    jobs = local_backend.list_jobs(runtime=runtime)
    assert isinstance(jobs, list)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_logs_nonexistent_job",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job", "step": "train"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_job_logs(local_backend, test_case):
    """Test LocalProcessBackend.get_job_logs()."""
    job_name = test_case.config.get("job_name")
    step = test_case.config.get("step", "train")

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            list(local_backend.get_job_logs(job_name, step=step))


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait_for_nonexistent_job",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_wait_for_job_status(local_backend, test_case):
    """Test LocalProcessBackend.wait_for_job_status()."""
    job_name = test_case.config.get("job_name")

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            local_backend.wait_for_job_status(job_name)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete_nonexistent_job",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_delete_job(local_backend, test_case):
    """Test LocalProcessBackend.delete_job()."""
    job_name = test_case.config.get("job_name")

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            local_backend.delete_job(job_name)


def test_name_option_sets_job_name(local_backend, mock_train_environment):
    """Test that Name option sets the custom job name."""
    custom_name = "my-custom-job-name"

    def dummy_func():
        pass

    runtime = types.Runtime(
        name=TORCH_RUNTIME,
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image=LOCAL_RUNTIME_IMAGE,
        ),
    )

    trainer = types.CustomTrainer(func=dummy_func)
    options = [Name(name=custom_name)]

    job_name = local_backend.train(
        runtime=runtime,
        trainer=trainer,
        options=options,
    )

    assert job_name == custom_name
