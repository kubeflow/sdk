# Copyright 2026 The Kubeflow Authors.
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

"""Unit tests for core Podman adapter operations without a Podman daemon."""

from unittest.mock import MagicMock

import pytest

from kubeflow.trainer.backends.container.adapters.podman import PodmanClientAdapter


@pytest.fixture
def adapter() -> PodmanClientAdapter:
    """Construct an adapter with a mocked client, avoiding optional Podman imports."""
    instance = PodmanClientAdapter.__new__(PodmanClientAdapter)
    instance.client = MagicMock()
    return instance


def test_ping(adapter: PodmanClientAdapter) -> None:
    adapter.ping()
    adapter.client.ping.assert_called_once_with()


def test_create_network_when_existing(adapter: PodmanClientAdapter) -> None:
    assert adapter.create_network("training", {"team": "ml"}) == "training"
    adapter.client.networks.get.assert_called_once_with("training")
    adapter.client.networks.create.assert_not_called()


def test_create_network_when_missing(adapter: PodmanClientAdapter) -> None:
    adapter.client.networks.get.side_effect = RuntimeError("not found")
    assert adapter.create_network("training", {"team": "ml"}) == "training"
    adapter.client.networks.create.assert_called_once_with(
        name="training", driver="bridge", dns_enabled=True, labels={"team": "ml"}
    )


def test_delete_network(adapter: PodmanClientAdapter) -> None:
    adapter.delete_network("training")
    adapter.client.networks.get.return_value.remove.assert_called_once_with()


def test_delete_missing_network_is_ignored(adapter: PodmanClientAdapter) -> None:
    adapter.client.networks.get.side_effect = RuntimeError("not found")
    adapter.delete_network("missing")


def test_create_and_start_container(adapter: PodmanClientAdapter) -> None:
    adapter.client.containers.run.return_value.id = "container-123"
    result = adapter.create_and_start_container(
        image="python:3.12",
        command=["python", "train.py"],
        name="training",
        network_id="training-net",
        environment={"MODE": "test"},
        labels={"app": "trainer"},
        volumes={},
        working_dir="/workspace",
    )
    assert result == "container-123"
    adapter.client.containers.run.assert_called_once_with(
        image="python:3.12",
        command=["python", "train.py"],
        name="training",
        network="training-net",
        working_dir="/workspace",
        environment={"MODE": "test"},
        labels={"app": "trainer"},
        volumes={},
        detach=True,
        remove=False,
    )


def test_pull_image(adapter: PodmanClientAdapter) -> None:
    adapter.pull_image("python:3.12")
    adapter.client.images.pull.assert_called_once_with("python:3.12")


def test_image_exists(adapter: PodmanClientAdapter) -> None:
    assert adapter.image_exists("python:3.12") is True
    adapter.client.images.get.side_effect = RuntimeError("not found")
    assert adapter.image_exists("missing") is False


def test_container_status_with_exit_code(adapter: PodmanClientAdapter) -> None:
    container = adapter.client.containers.get.return_value
    container.status = "exited"
    container.attrs = {"State": {"ExitCode": 3}}
    assert adapter.container_status("container-123") == ("exited", 3)


def test_container_status_missing(adapter: PodmanClientAdapter) -> None:
    adapter.client.containers.get.side_effect = RuntimeError("not found")
    assert adapter.container_status("missing") == ("unknown", None)


def test_get_container_ip_exact_network(adapter: PodmanClientAdapter) -> None:
    container = adapter.client.containers.get.return_value
    container.attrs = {
        "NetworkSettings": {"Networks": {"training": {"IPAddress": "10.0.0.2"}}}
    }
    assert adapter.get_container_ip("container-123", "training") == "10.0.0.2"


def test_get_container_ip_missing_container(adapter: PodmanClientAdapter) -> None:
    adapter.client.containers.get.side_effect = RuntimeError("not found")
    assert adapter.get_container_ip("missing", "training") is None


def test_wait_for_container(adapter: PodmanClientAdapter) -> None:
    adapter.client.containers.get.return_value.wait.return_value = "0"
    assert adapter.wait_for_container("container-123", timeout=15) == 0
    adapter.client.containers.get.return_value.wait.assert_called_once_with(timeout=15)


def test_wait_for_container_timeout(adapter: PodmanClientAdapter) -> None:
    adapter.client.containers.get.return_value.wait.side_effect = RuntimeError("timeout reached")
    with pytest.raises(TimeoutError, match="did not exit within 15 seconds"):
        adapter.wait_for_container("container-123", timeout=15)
