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

"""Unit tests for the Docker and Podman adapters' create_and_start_container."""

from unittest.mock import MagicMock

import pytest

from kubeflow.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.backends.container.adapters.docker import DockerClientAdapter
from kubeflow.trainer.backends.container.adapters.podman import PodmanClientAdapter

CONTAINER_ARGS = {
    "image": "pytorch/pytorch:latest",
    "command": ["python", "train.py"],
    "name": "job-node-0",
    "network_id": "job-net",
    "environment": {},
    "labels": {},
    "volumes": {},
    "working_dir": "/workspace",
}


def _podman_create(**kwargs):
    """Mimic podman-py's payload rendering, which iterates over `devices`."""
    for _ in kwargs.pop("devices", []):
        pass
    container = MagicMock()
    container.id = "podman-id"
    return container


def _adapter(adapter_cls, create_side_effect):
    # Skip __init__ so the tests don't need a running daemon or the podman package.
    adapter = adapter_cls.__new__(adapter_cls)
    adapter.client = MagicMock()
    adapter.client.containers.create.side_effect = create_side_effect
    return adapter


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="podman without GPUs does not pass devices",
            expected_status=SUCCESS,
            config={"adapter": "podman", "gpu_ids": None},
            expected_output=None,
        ),
        TestCase(
            name="podman with GPUs passes CDI devices",
            expected_status=SUCCESS,
            config={"adapter": "podman", "gpu_ids": [2, 3]},
            expected_output=["nvidia.com/gpu=2", "nvidia.com/gpu=3"],
        ),
        TestCase(
            name="docker without GPUs does not pass device_requests",
            expected_status=SUCCESS,
            config={"adapter": "docker", "gpu_ids": None},
            expected_output=None,
        ),
        TestCase(
            name="docker with GPUs requests the given device IDs",
            expected_status=SUCCESS,
            config={"adapter": "docker", "gpu_ids": [2, 3]},
            expected_output=["2", "3"],
        ),
        TestCase(
            name="docker removes the container when start fails",
            expected_status=FAILED,
            config={"adapter": "docker", "gpu_ids": [0], "start_error": True},
            expected_error=RuntimeError,
        ),
        TestCase(
            name="podman removes the container when start fails",
            expected_status=FAILED,
            config={"adapter": "podman", "gpu_ids": [0], "start_error": True},
            expected_error=RuntimeError,
        ),
    ],
)
def test_create_and_start_container(test_case):
    print("Executing test:", test_case.name)
    is_podman = test_case.config["adapter"] == "podman"
    if is_podman:
        adapter = _adapter(PodmanClientAdapter, _podman_create)
    else:
        container = MagicMock()
        container.id = "docker-id"
        adapter = _adapter(DockerClientAdapter, lambda **kwargs: container)

    if test_case.config.get("start_error"):
        started = MagicMock()
        started.start.side_effect = RuntimeError("could not select device driver")
        adapter.client.containers.create.side_effect = None
        adapter.client.containers.create.return_value = started

    try:
        container_id = adapter.create_and_start_container(
            **CONTAINER_ARGS, gpu_ids=test_case.config["gpu_ids"]
        )

        assert test_case.expected_status == SUCCESS
        assert container_id in ("podman-id", "docker-id")
        kwargs = adapter.client.containers.create.call_args.kwargs
        if is_podman:
            assert kwargs.get("devices") == test_case.expected_output
        elif test_case.expected_output is None:
            assert "device_requests" not in kwargs
        else:
            (request,) = kwargs["device_requests"]
            assert request["DeviceIDs"] == test_case.expected_output
            assert request["Capabilities"] == [["gpu"]]
            assert "Count" not in request or request["Count"] in (None, 0)

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        started.remove.assert_called_once_with(force=True)

    print("test execution complete")
