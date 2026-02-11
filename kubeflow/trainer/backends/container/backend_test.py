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
Unit tests for ContainerBackend.

Tests the ContainerBackend class with mocked container adapters.
"""

from collections.abc import Iterator
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.backends.container.adapters.base import (
    BaseContainerClientAdapter,
)
from kubeflow.trainer.backends.container.backend import ContainerBackend
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


# Mock Container Adapter
class MockContainerAdapter(BaseContainerClientAdapter):
    """Mock adapter for testing ContainerBackend without Docker/Podman."""

    def __init__(self):
        self._runtime_type = "mock"
        self.networks_created = []
        self.containers_created = []
        self.containers_stopped = []
        self.containers_removed = []
        self.networks_deleted = []
        self.images_pulled = []
        self.ping_called = False

    def ping(self):
        self.ping_called = True

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        network_id = f"net-{name}"
        self.networks_created.append({"id": network_id, "name": name, "labels": labels})
        return network_id

    def delete_network(self, network_id: str):
        self.networks_deleted.append(network_id)

    def create_and_start_container(
        self,
        image: str,
        command: list[str],
        name: str,
        network_id: str,
        environment: dict[str, str],
        labels: dict[str, str],
        volumes: dict[str, dict[str, str]],
        working_dir: str,
    ) -> str:
        container_id = f"container-{len(self.containers_created)}"
        self.containers_created.append(
            {
                "id": container_id,
                "name": name,
                "image": image,
                "command": command,
                "network": network_id,
                "environment": environment,
                "labels": labels,
                "volumes": volumes,
                "working_dir": working_dir,
                "status": "running",
                "exit_code": None,
            }
        )
        return container_id

    def get_container(self, container_id: str):
        for container in self.containers_created:
            if container["id"] == container_id:
                return Mock(id=container_id, status=container["status"])
        return None

    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        if follow:
            yield f"Log line 1 from {container_id}\n"
            yield f"Log line 2 from {container_id}\n"
        else:
            yield f"Complete log from {container_id}\n"

    def stop_container(self, container_id: str, timeout: int = 10):
        self.containers_stopped.append(container_id)
        for container in self.containers_created:
            if container["id"] == container_id:
                container["status"] = "exited"
                container["exit_code"] = 0

    def remove_container(self, container_id: str, force: bool = True):
        self.containers_removed.append(container_id)

    def pull_image(self, image: str):
        self.images_pulled.append(image)

    def image_exists(self, image: str) -> bool:
        return "local" in image or image in self.images_pulled

    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        return "Python 3.10.0\npip 21.0.1\nnvidia-smi not found\n"

    def container_status(self, container_id: str) -> tuple[str, int | None]:
        for container in self.containers_created:
            if container["id"] == container_id:
                return (container["status"], container.get("exit_code"))
        return ("unknown", None)

    def set_container_status(self, container_id: str, status: str, exit_code: int | None = None):
        """Helper method to set container status for testing."""
        for container in self.containers_created:
            if container["id"] == container_id:
                container["status"] = status
                container["exit_code"] = exit_code

    def get_container_ip(self, container_id: str, network_id: str) -> str | None:
        """Get container IP address on a specific network."""
        for container in self.containers_created:
            if container["id"] == container_id:
                return f"192.168.1.{len(self.containers_created)}"
        return None

    def list_containers(self, filters: dict[str, list[str]] | None = None) -> list[dict]:
        """List containers with optional filters."""
        if not filters:
            return [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "labels": c["labels"],
                    "status": c["status"],
                    "created": "2025-01-01T00:00:00Z",
                }
                for c in self.containers_created
            ]

        # Simple label filtering
        result = []
        for container in self.containers_created:
            if "label" in filters:
                match = True
                for label_filter in filters["label"]:
                    if "=" in label_filter:
                        key, value = label_filter.split("=", 1)
                        if container["labels"].get(key) != value:
                            match = False
                            break
                    else:
                        if label_filter not in container["labels"]:
                            match = False
                            break
                if match:
                    result.append(
                        {
                            "id": container["id"],
                            "name": container["name"],
                            "labels": container["labels"],
                            "status": container["status"],
                            "created": "2025-01-01T00:00:00Z",
                        }
                    )
        return result

    def get_network(self, network_id: str) -> dict | None:
        """Get network information."""
        for network in self.networks_created:
            if network["id"] == network_id or network["name"] == network_id:
                return {
                    "id": network["id"],
                    "name": network["name"],
                    "labels": network["labels"],
                }
        return None

    def wait_for_container(self, container_id: str, timeout: int | None = None) -> int:
        """
        Wait for a container to exit and return its exit code.

        For testing, immediately returns the container's exit code if it has exited,
        or raises TimeoutError if the container is still running.

        Args:
            container_id: Container ID
            timeout: Maximum time to wait in seconds (not used in mock)

        Returns:
            Container exit code

        Raises:
            TimeoutError: If container is still running
        """
        for container in self.containers_created:
            if container["id"] == container_id:
                if container["status"] == "exited":
                    return container.get("exit_code", 0)
                # In mock, if not exited, simulate timeout
                raise TimeoutError(f"Container {container_id} did not exit within timeout")
        raise RuntimeError(f"Container {container_id} not found")


# Fixtures
@pytest.fixture
def container_backend():
    """Provide ContainerBackend with mocked adapter."""
    with patch("kubeflow.trainer.backends.container.backend.DockerClientAdapter") as mock_docker:
        mock_docker.return_value = MockContainerAdapter()
        backend = ContainerBackend(ContainerBackendConfig())
        return backend


@pytest.fixture
def temp_workdir():
    """Provide a temporary working directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


# Helper Function
def simple_train_func():
    """Simple training function for tests."""
    print("Training")


# Tests
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="auto-detect docker first",
            expected_status=SUCCESS,
        ),
        TestCase(
            name="auto-detect falls back to podman",
            expected_status=SUCCESS,
        ),
        TestCase(
            name="both unavailable raises error",
            expected_status=FAILED,
            expected_error=RuntimeError,
        ),
    ],
)
def test_backend_initialization(test_case):
    """Test ContainerBackend initialization and adapter creation."""
    print("Executing test:", test_case.name)
    try:
        if test_case.name == "auto-detect docker first":
            with (
                patch(
                    "kubeflow.trainer.backends.container.backend.DockerClientAdapter"
                ) as mock_docker,
                patch(
                    "kubeflow.trainer.backends.container.backend.PodmanClientAdapter"
                ) as mock_podman,
            ):
                mock_docker_instance = Mock()
                mock_docker.return_value = mock_docker_instance

                _ = ContainerBackend(ContainerBackendConfig())

                # Docker should be called (could be with Colima socket or None)
                assert mock_docker.call_count == 1
                mock_docker_instance.ping.assert_called_once()
                mock_podman.assert_not_called()
                assert test_case.expected_status == SUCCESS

        elif test_case.name == "auto-detect falls back to podman":
            with (
                patch(
                    "kubeflow.trainer.backends.container.backend.DockerClientAdapter"
                ) as mock_docker,
                patch(
                    "kubeflow.trainer.backends.container.backend.PodmanClientAdapter"
                ) as mock_podman,
            ):
                mock_docker_instance = Mock()
                mock_docker_instance.ping.side_effect = Exception("Docker not available")
                mock_docker.return_value = mock_docker_instance

                mock_podman_instance = Mock()
                mock_podman.return_value = mock_podman_instance

                _ = ContainerBackend(ContainerBackendConfig())

                # Docker may be tried multiple times (different socket locations)
                assert mock_docker.call_count >= 1
                mock_podman.assert_called_once_with(None)
                mock_podman_instance.ping.assert_called_once()
                assert test_case.expected_status == SUCCESS

        elif test_case.name == "both unavailable raises error":
            with (
                patch(
                    "kubeflow.trainer.backends.container.backend.DockerClientAdapter"
                ) as mock_docker,
                patch(
                    "kubeflow.trainer.backends.container.backend.PodmanClientAdapter"
                ) as mock_podman,
            ):
                mock_docker_instance = Mock()
                mock_docker_instance.ping.side_effect = Exception("Docker not available")
                mock_docker.return_value = mock_docker_instance

                mock_podman_instance = Mock()
                mock_podman_instance.ping.side_effect = Exception("Podman not available")
                mock_podman.return_value = mock_podman_instance

                ContainerBackend(ContainerBackendConfig())

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_list_runtimes(container_backend):
    """Test listing available local runtimes."""
    print("Executing test: list_runtimes")
    runtimes = container_backend.list_runtimes()

    assert isinstance(runtimes, list)
    assert len(runtimes) > 0
    runtime_names = [r.name for r in runtimes]
    assert constants.DEFAULT_TRAINING_RUNTIME in runtime_names
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get valid runtime",
            expected_status=SUCCESS,
            config={"name": constants.DEFAULT_TRAINING_RUNTIME},
        ),
        TestCase(
            name="get invalid runtime",
            expected_status=FAILED,
            config={"name": "nonexistent-runtime"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_runtime(container_backend, test_case):
    """Test getting a specific runtime."""
    print("Executing test:", test_case.name)
    try:
        runtime = container_backend.get_runtime(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(runtime, types.Runtime)
        assert runtime.name == test_case.config["name"]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_get_runtime_packages(container_backend):
    """Test getting runtime packages."""
    print("Executing test: get_runtime_packages")
    runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)
    container_backend.get_runtime_packages(runtime)

    assert len(
        container_backend._adapter.images_pulled
    ) > 0 or container_backend._adapter.image_exists(runtime.trainer.image)
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train single node",
            expected_status=SUCCESS,
            config={"num_nodes": 1, "expected_containers": 1},
        ),
        TestCase(
            name="train multi-node",
            expected_status=SUCCESS,
            config={"num_nodes": 3, "expected_containers": 3},
        ),
        TestCase(
            name="train with custom env",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "env": {"MY_VAR": "my_value", "DEBUG": "true"},
                "expected_containers": 1,
            },
        ),
        TestCase(
            name="train with packages",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "packages": ["numpy", "pandas"],
                "expected_containers": 1,
            },
        ),
        TestCase(
            name="train with single GPU",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "resources_per_node": {"gpu": "1"},
                "expected_containers": 1,
                "expected_nproc_per_node": 1,
            },
        ),
        TestCase(
            name="train with multiple GPUs",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "resources_per_node": {"gpu": "4"},
                "expected_containers": 1,
                "expected_nproc_per_node": 4,
            },
        ),
        TestCase(
            name="train multi-node with GPUs",
            expected_status=SUCCESS,
            config={
                "num_nodes": 2,
                "resources_per_node": {"gpu": "2"},
                "expected_containers": 2,
                "expected_nproc_per_node": 2,
            },
        ),
        TestCase(
            name="train with CPU resources (nproc=1)",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "resources_per_node": {"cpu": "16"},
                "expected_containers": 1,
                "expected_nproc_per_node": 1,
            },
        ),
        TestCase(
            name="train with HuggingFace dataset initializer",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "initializer": types.Initializer(
                    dataset=types.HuggingFaceDatasetInitializer(
                        storage_uri="hf://username/dataset-repo",
                        access_token="hf_token_123",
                    )
                ),
                "expected_containers": 2,  # 1 dataset-initializer + 1 training node
                "expected_initializer_type": "dataset",
            },
        ),
        TestCase(
            name="train with S3 dataset initializer",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "initializer": types.Initializer(
                    dataset=types.S3DatasetInitializer(
                        storage_uri="s3://my-bucket/dataset",
                        endpoint="https://s3.amazonaws.com",
                        region="us-west-2",
                    )
                ),
                "expected_containers": 2,  # 1 dataset-initializer + 1 training node
                "expected_initializer_type": "dataset",
            },
        ),
        TestCase(
            name="train with HuggingFace model initializer",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "initializer": types.Initializer(
                    model=types.HuggingFaceModelInitializer(
                        storage_uri="hf://username/model-repo",
                        access_token="hf_token_456",
                        ignore_patterns=["*.bin", "*.h5"],
                    )
                ),
                "expected_containers": 2,  # 1 model-initializer + 1 training node
                "expected_initializer_type": "model",
            },
        ),
        TestCase(
            name="train with S3 model initializer",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "initializer": types.Initializer(
                    model=types.S3ModelInitializer(
                        storage_uri="s3://my-bucket/model",
                        endpoint="https://s3.amazonaws.com",
                        access_key_id="my_access_key",
                        secret_access_key="my_secret_key",
                    )
                ),
                "expected_containers": 2,  # 1 model-initializer + 1 training node
                "expected_initializer_type": "model",
            },
        ),
        TestCase(
            name="train with both dataset and model initializers",
            expected_status=SUCCESS,
            config={
                "num_nodes": 2,
                "initializer": types.Initializer(
                    dataset=types.HuggingFaceDatasetInitializer(
                        storage_uri="hf://username/dataset-repo"
                    ),
                    model=types.HuggingFaceModelInitializer(storage_uri="hf://username/model-repo"),
                ),
                "expected_containers": 4,  # 1 dataset + 1 model + 2 training nodes
            },
        ),
        TestCase(
            name="initializer fails with non-zero exit code",
            expected_status=FAILED,
            config={
                "num_nodes": 1,
                "initializer": types.Initializer(
                    dataset=types.HuggingFaceDatasetInitializer(
                        storage_uri="hf://user/invalid-dataset"
                    )
                ),
                "expected_containers": 1,  # Only initializer created before failure
                "initializer_exit_code": 1,
            },
            expected_error=RuntimeError,
        ),
        TestCase(
            name="initializer timeout",
            expected_status=FAILED,
            config={
                "num_nodes": 1,
                "initializer": types.Initializer(
                    model=types.S3ModelInitializer(storage_uri="s3://bucket/model")
                ),
                "expected_containers": 1,  # Only initializer created before timeout
                "initializer_timeout": True,
            },
            expected_error=TimeoutError,
        ),
    ],
)
def test_train(container_backend, test_case):
    """Test training job creation."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(
            func=simple_train_func,
            num_nodes=test_case.config.get("num_nodes", 1),
            env=test_case.config.get("env"),
            packages_to_install=test_case.config.get("packages"),
            resources_per_node=test_case.config.get("resources_per_node"),
        )
        runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)

        # Set up mocking for initializer tests
        initializer = test_case.config.get("initializer")
        if initializer:
            original_create = container_backend._adapter.create_and_start_container

            def mock_create_with_status(*args, **kwargs):
                container_id = original_create(*args, **kwargs)
                # If it's an initializer container, set status based on test config
                if "initializer" in kwargs.get("name", ""):
                    if "initializer_exit_code" in test_case.config:
                        container_backend._adapter.set_container_status(
                            container_id, "exited", test_case.config["initializer_exit_code"]
                        )
                    else:
                        # Mark as completed successfully
                        container_backend._adapter.set_container_status(container_id, "exited", 0)
                return container_id

            container_backend._adapter.create_and_start_container = mock_create_with_status

        # Handle timeout test case
        if test_case.config.get("initializer_timeout"):
            with patch(
                "kubeflow.trainer.backends.container.backend."
                "ContainerBackend._run_single_initializer"
            ) as mock_run:
                mock_run.side_effect = TimeoutError("Initializer timeout")
                container_backend.train(runtime=runtime, trainer=trainer, initializer=initializer)
        else:
            job_name = container_backend.train(
                runtime=runtime, trainer=trainer, initializer=initializer
            )

        assert test_case.expected_status == SUCCESS
        assert job_name is not None
        assert len(job_name) == 12
        assert (
            len(container_backend._adapter.containers_created)
            == test_case.config["expected_containers"]
        )
        assert len(container_backend._adapter.networks_created) == 1

        # Check environment if specified
        if "env" in test_case.config:
            container = container_backend._adapter.containers_created[0]
            for key, value in test_case.config["env"].items():
                assert container["environment"][key] == value

        # Check packages if specified
        if "packages" in test_case.config:
            container = container_backend._adapter.containers_created[0]
            command_str = " ".join(container["command"])
            assert "pip install" in command_str
            for package in test_case.config["packages"]:
                assert package in command_str

        # Check nproc_per_node if specified
        if "expected_nproc_per_node" in test_case.config:
            container = container_backend._adapter.containers_created[0]
            command_str = container["command"][2]  # Get bash script content
            expected_nproc = test_case.config["expected_nproc_per_node"]
            assert f"--nproc_per_node={expected_nproc}" in command_str, (
                f"Expected --nproc_per_node={expected_nproc} in command, but got: {command_str}"
            )

        # Check initializer assertions
        if initializer:
            # Check that initializer containers have correct labels
            initializer_containers = [
                c
                for c in container_backend._adapter.containers_created
                if "initializer" in c["labels"].get(f"{container_backend.label_prefix}/step", "")
            ]

            if "expected_initializer_type" in test_case.config:
                expected_type = test_case.config["expected_initializer_type"]
                assert any(expected_type in c["name"] for c in initializer_containers)

            # Check that initializer containers have correct environment variables
            for container in initializer_containers:
                assert "STORAGE_URI" in container["environment"]
                assert "OUTPUT_PATH" in container["environment"]

                # Verify OUTPUT_PATH is correct based on initializer type
                if "dataset-initializer" in container["name"]:
                    assert container["environment"]["OUTPUT_PATH"] == constants.DATASET_PATH
                elif "model-initializer" in container["name"]:
                    assert container["environment"]["OUTPUT_PATH"] == constants.MODEL_PATH

            # Verify the job can be retrieved and has correct steps
            job = container_backend.get_job(job_name)
            assert job.name == job_name

            # Check that initializer steps are included
            step_names = [step.name for step in job.steps]
            if initializer.dataset:
                assert "dataset-initializer" in step_names
            if initializer.model:
                assert "model-initializer" in step_names

            # Check that num_nodes only counts training nodes, not initializers
            assert job.num_nodes == test_case.config.get("num_nodes", 1)

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list all jobs",
            expected_status=SUCCESS,
            config={"num_jobs": 2},
        ),
        TestCase(
            name="list empty jobs",
            expected_status=SUCCESS,
            config={"num_jobs": 0},
        ),
    ],
)
def test_list_jobs(container_backend, test_case):
    """Test listing training jobs."""
    print("Executing test:", test_case.name)
    try:
        runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)
        created_jobs = []

        for _ in range(test_case.config["num_jobs"]):
            trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
            job_name = container_backend.train(runtime=runtime, trainer=trainer)
            created_jobs.append(job_name)

        jobs = container_backend.list_jobs()

        assert test_case.expected_status == SUCCESS
        assert len(jobs) == test_case.config["num_jobs"]
        if test_case.config["num_jobs"] > 0:
            job_names = [job.name for job in jobs]
            for created_job in created_jobs:
                assert created_job in job_names

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get existing job",
            expected_status=SUCCESS,
            config={"num_nodes": 2},
        ),
        TestCase(
            name="get nonexistent job",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_job(container_backend, test_case):
    """Test getting a specific job."""
    print("Executing test:", test_case.name)
    try:
        if test_case.name == "get existing job":
            trainer = types.CustomTrainer(
                func=simple_train_func, num_nodes=test_case.config["num_nodes"]
            )
            runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)
            job_name = container_backend.train(runtime=runtime, trainer=trainer)

            job = container_backend.get_job(job_name)

            assert test_case.expected_status == SUCCESS
            assert job.name == job_name
            assert job.num_nodes == test_case.config["num_nodes"]
            assert len(job.steps) == test_case.config["num_nodes"]

        elif test_case.name == "get nonexistent job":
            container_backend.get_job(test_case.config["job_name"])

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get logs no follow",
            expected_status=SUCCESS,
            config={"follow": False},
        ),
        TestCase(
            name="get logs with follow",
            expected_status=SUCCESS,
            config={"follow": True},
        ),
        TestCase(
            name="get logs from dataset initializer",
            expected_status=SUCCESS,
            config={
                "follow": False,
                "step": "dataset-initializer",
                "initializer": types.Initializer(
                    dataset=types.HuggingFaceDatasetInitializer(storage_uri="hf://user/dataset"),
                    model=types.HuggingFaceModelInitializer(storage_uri="hf://user/model"),
                ),
            },
        ),
        TestCase(
            name="get logs from model initializer",
            expected_status=SUCCESS,
            config={
                "follow": False,
                "step": "model-initializer",
                "initializer": types.Initializer(
                    dataset=types.HuggingFaceDatasetInitializer(storage_uri="hf://user/dataset"),
                    model=types.HuggingFaceModelInitializer(storage_uri="hf://user/model"),
                ),
            },
        ),
        TestCase(
            name="get logs from training node excludes initializers",
            expected_status=SUCCESS,
            config={
                "follow": False,
                "step": constants.NODE + "-0",
                "initializer": types.Initializer(
                    dataset=types.HuggingFaceDatasetInitializer(storage_uri="hf://user/dataset"),
                ),
            },
        ),
    ],
)
def test_get_job_logs(container_backend, test_case):
    """Test getting job logs."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
        runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)

        initializer = test_case.config.get("initializer")

        # Set up mocking for initializer tests
        if initializer:
            original_create = container_backend._adapter.create_and_start_container

            def mock_create_with_status(*args, **kwargs):
                container_id = original_create(*args, **kwargs)
                if "initializer" in kwargs.get("name", ""):
                    container_backend._adapter.set_container_status(container_id, "exited", 0)
                return container_id

            container_backend._adapter.create_and_start_container = mock_create_with_status

        job_name = container_backend.train(
            runtime=runtime, trainer=trainer, initializer=initializer
        )

        step = test_case.config.get("step")
        if step:
            logs = list(
                container_backend.get_job_logs(
                    job_name, follow=test_case.config["follow"], step=step
                )
            )
        else:
            logs = list(container_backend.get_job_logs(job_name, follow=test_case.config["follow"]))

        assert test_case.expected_status == SUCCESS
        assert len(logs) > 0

        if not initializer:
            if test_case.config["follow"]:
                assert any("Log line" in log for log in logs)
            else:
                assert any("Complete log" in log for log in logs)

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait for complete",
            expected_status=SUCCESS,
            config={"wait_status": constants.TRAINJOB_COMPLETE, "container_exit_code": 0},
        ),
        TestCase(
            name="wait timeout",
            expected_status=FAILED,
            config={"wait_status": constants.TRAINJOB_COMPLETE, "timeout": 2},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="job fails",
            expected_status=FAILED,
            config={"wait_status": constants.TRAINJOB_COMPLETE, "container_exit_code": 1},
            expected_error=RuntimeError,
        ),
    ],
)
def test_wait_for_job_status(container_backend, test_case):
    """Test waiting for job status."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
        runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)
        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        if test_case.name == "wait for complete":
            container_id = container_backend._adapter.containers_created[0]["id"]
            container_backend._adapter.set_container_status(
                container_id, "exited", test_case.config["container_exit_code"]
            )

            completed_job = container_backend.wait_for_job_status(
                job_name, status={test_case.config["wait_status"]}, timeout=5, polling_interval=1
            )

            assert test_case.expected_status == SUCCESS
            assert completed_job.status == constants.TRAINJOB_COMPLETE

        elif test_case.name == "wait timeout":
            container_backend.wait_for_job_status(
                job_name,
                status={test_case.config["wait_status"]},
                timeout=test_case.config["timeout"],
                polling_interval=1,
            )

        elif test_case.name == "job fails":
            container_id = container_backend._adapter.containers_created[0]["id"]
            container_backend._adapter.set_container_status(
                container_id, "exited", test_case.config["container_exit_code"]
            )

            container_backend.wait_for_job_status(
                job_name, status={test_case.config["wait_status"]}, timeout=5, polling_interval=1
            )

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete with auto_remove true",
            expected_status=SUCCESS,
            config={"auto_remove": True, "num_nodes": 2},
        ),
        TestCase(
            name="delete with auto_remove false",
            expected_status=SUCCESS,
            config={"auto_remove": False, "num_nodes": 2},
        ),
    ],
)
def test_delete_job(container_backend, temp_workdir, test_case):
    """Test deleting a job."""
    print("Executing test:", test_case.name)
    try:
        container_backend.cfg.auto_remove = test_case.config["auto_remove"]

        trainer = types.CustomTrainer(
            func=simple_train_func, num_nodes=test_case.config["num_nodes"]
        )
        runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)
        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        job_workdir = Path.home() / ".kubeflow" / "trainer" / "containers" / job_name
        assert job_workdir.exists()

        container_backend.delete_job(job_name)

        assert test_case.expected_status == SUCCESS
        assert len(container_backend._adapter.containers_stopped) == test_case.config["num_nodes"]
        assert len(container_backend._adapter.containers_removed) == test_case.config["num_nodes"]
        assert len(container_backend._adapter.networks_deleted) == 1

        if test_case.config["auto_remove"]:
            assert not job_workdir.exists()
        else:
            assert job_workdir.exists()

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="running container",
            expected_status=SUCCESS,
            config={
                "container_status": "running",
                "exit_code": None,
                "expected_job_status": constants.TRAINJOB_RUNNING,
            },
        ),
        TestCase(
            name="exited success",
            expected_status=SUCCESS,
            config={
                "container_status": "exited",
                "exit_code": 0,
                "expected_job_status": constants.TRAINJOB_COMPLETE,
            },
        ),
        TestCase(
            name="exited failure",
            expected_status=SUCCESS,
            config={
                "container_status": "exited",
                "exit_code": 1,
                "expected_job_status": constants.TRAINJOB_FAILED,
            },
        ),
    ],
)
def test_container_status_mapping(container_backend, test_case):
    """Test container status mapping to TrainJob status."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
        runtime = container_backend.get_runtime(constants.DEFAULT_TRAINING_RUNTIME)
        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        container_id = container_backend._adapter.containers_created[0]["id"]
        container_backend._adapter.set_container_status(
            container_id, test_case.config["container_status"], test_case.config["exit_code"]
        )

        job = container_backend.get_job(job_name)

        assert test_case.expected_status == SUCCESS
        assert job.status == test_case.config["expected_job_status"]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="docker socket locations with colima",
            expected_status=SUCCESS,
            config={
                "runtime_name": "docker",
                "container_host": None,
                "create_colima_socket": True,
                "expected_contains_none": True,
                "expected_has_colima": True,
            },
        ),
        TestCase(
            name="custom host has priority",
            expected_status=SUCCESS,
            config={
                "runtime_name": "docker",
                "container_host": "unix:///custom/path/docker.sock",
                "create_colima_socket": False,
                "expected_first": "unix:///custom/path/docker.sock",
            },
        ),
    ],
)
def test_get_common_socket_locations(test_case, tmp_path):
    """Test common socket location detection."""
    print("Executing test:", test_case.name)

    # Setup
    if test_case.config.get("create_colima_socket"):
        colima_dir = tmp_path / ".colima" / "default"
        colima_dir.mkdir(parents=True)
        colima_sock = colima_dir / "docker.sock"
        colima_sock.touch()

    cfg = ContainerBackendConfig(container_host=test_case.config["container_host"])

    # Test the method directly without creating the backend
    context_manager = (
        patch("pathlib.Path.home", return_value=tmp_path)
        if test_case.config.get("create_colima_socket")
        else nullcontext()
    )

    with context_manager:
        backend = ContainerBackend.__new__(ContainerBackend)
        backend.cfg = cfg
        locations = backend._get_common_socket_locations(test_case.config["runtime_name"])

        # Assertions
        if "expected_contains_none" in test_case.config:
            assert None in locations

        if "expected_has_colima" in test_case.config:
            assert f"unix://{colima_sock}" in locations

        if "expected_first" in test_case.config:
            assert locations[0] == test_case.config["expected_first"]

    print("test execution complete")


def test_create_adapter_error_message_format():
    """Test that error message includes attempted connections."""
    cfg = ContainerBackendConfig(container_runtime="docker")

    docker_adapter = "kubeflow.trainer.backends.container.adapters.docker.DockerClientAdapter"
    with patch(docker_adapter) as mock_docker:
        mock_docker.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError) as exc_info:
            ContainerBackend(cfg)

        # Error message should be helpful
        error_msg = str(exc_info.value)
        assert "Could not connect" in error_msg
        assert "tried:" in error_msg
