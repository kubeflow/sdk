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

"""Unit tests for PodmanClientAdapter."""

import builtins
from unittest.mock import MagicMock, Mock, patch

import pytest

from kubeflow.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.backends.container.adapters.podman import PodmanClientAdapter


@pytest.fixture
def podman_module():
    """Patch the optional `podman` package so tests do not need it installed."""
    module = MagicMock()
    with patch.dict("sys.modules", {"podman": module}):
        yield module


@pytest.fixture
def adapter(podman_module):
    """PodmanClientAdapter whose underlying client is a mock."""
    return PodmanClientAdapter()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="no host uses environment defaults",
            expected_status=SUCCESS,
            config={"host": None},
        ),
        TestCase(
            name="explicit host is passed as base_url",
            expected_status=SUCCESS,
            config={"host": "unix:///run/podman/podman.sock"},
        ),
    ],
)
def test_init(podman_module, test_case):
    """Test client construction with and without an explicit host."""
    host = test_case.config["host"]
    adapter = PodmanClientAdapter(host)

    if host:
        podman_module.PodmanClient.assert_called_once_with(base_url=host)
    else:
        podman_module.PodmanClient.assert_called_once_with()
    assert adapter.client is podman_module.PodmanClient.return_value
    assert adapter._runtime_type == "podman"


def test_init_missing_podman_package():
    """Test a helpful ImportError is raised when `podman` is not installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "podman":
            raise ImportError("No module named 'podman'")
        return real_import(name, *args, **kwargs)

    with (
        patch.object(builtins, "__import__", side_effect=fake_import),
        pytest.raises(ImportError, match=r"kubeflow\[podman\]"),
    ):
        PodmanClientAdapter()


def test_ping(adapter):
    """Test ping delegates to the Podman client."""
    adapter.ping()
    adapter.client.ping.assert_called_once()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="existing network is reused",
            expected_status=SUCCESS,
            config={"exists": True},
        ),
        TestCase(
            name="missing network is created with DNS enabled",
            expected_status=SUCCESS,
            config={"exists": False},
        ),
    ],
)
def test_create_network(adapter, test_case):
    """Test network creation is idempotent and enables DNS."""
    labels = {"app": "kubeflow"}
    if not test_case.config["exists"]:
        adapter.client.networks.get.side_effect = Exception("not found")

    result = adapter.create_network("net1", labels)

    assert result == "net1"
    adapter.client.networks.get.assert_called_once_with("net1")
    if test_case.config["exists"]:
        adapter.client.networks.create.assert_not_called()
    else:
        adapter.client.networks.create.assert_called_once_with(
            name="net1", driver="bridge", dns_enabled=True, labels=labels
        )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="network removed", expected_status=SUCCESS, config={"fails": False}),
        TestCase(
            name="errors are swallowed when network is missing",
            expected_status=SUCCESS,
            config={"fails": True},
        ),
    ],
)
def test_delete_network(adapter, test_case):
    """Test network deletion never raises."""
    if test_case.config["fails"]:
        adapter.client.networks.get.side_effect = Exception("not found")

    adapter.delete_network("net1")

    adapter.client.networks.get.assert_called_once_with("net1")
    if not test_case.config["fails"]:
        adapter.client.networks.get.return_value.remove.assert_called_once()


def test_create_and_start_container(adapter):
    """Test the container is run detached with the requested configuration."""
    adapter.client.containers.run.return_value.id = "abc123"

    container_id = adapter.create_and_start_container(
        image="python:3.11",
        command=["python", "train.py"],
        name="trainer-0",
        network_id="net1",
        environment={"A": "1"},
        labels={"l": "v"},
        volumes={"/host": {"bind": "/ctr", "mode": "rw"}},
        working_dir="/workspace",
    )

    assert container_id == "abc123"
    adapter.client.containers.run.assert_called_once_with(
        image="python:3.11",
        command=["python", "train.py"],
        name="trainer-0",
        network="net1",
        working_dir="/workspace",
        environment={"A": "1"},
        labels={"l": "v"},
        volumes={"/host": {"bind": "/ctr", "mode": "rw"}},
        detach=True,
        remove=False,
    )


def test_get_container(adapter):
    """Test get_container delegates to the client."""
    assert adapter.get_container("c1") is adapter.client.containers.get.return_value
    adapter.client.containers.get.assert_called_once_with("c1")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="non-follow bytes are decoded",
            config={"follow": False, "logs": b"hello"},
            expected_output=["hello"],
        ),
        TestCase(
            name="non-follow str is returned as is",
            config={"follow": False, "logs": "hello"},
            expected_output=["hello"],
        ),
        TestCase(
            name="follow streams and decodes bytes chunks",
            config={"follow": True, "logs": iter([b"a", "b", b"c"])},
            expected_output=["a", "b", "c"],
        ),
        TestCase(
            name="invalid utf-8 bytes are ignored instead of raising",
            config={"follow": False, "logs": b"ok\xff"},
            expected_output=["ok"],
        ),
    ],
)
def test_container_logs(adapter, test_case):
    """Test log streaming in follow and non-follow modes."""
    follow = test_case.config["follow"]
    container = adapter.client.containers.get.return_value
    container.logs.return_value = test_case.config["logs"]

    result = list(adapter.container_logs("c1", follow))

    assert result == test_case.expected_output
    container.logs.assert_called_once_with(stream=follow, follow=follow)


def test_stop_container(adapter):
    """Test stop forwards the timeout."""
    adapter.stop_container("c1", timeout=5)
    adapter.client.containers.get.return_value.stop.assert_called_once_with(timeout=5)


def test_stop_container_default_timeout(adapter):
    """Test stop uses a 10 second default timeout."""
    adapter.stop_container("c1")
    adapter.client.containers.get.return_value.stop.assert_called_once_with(timeout=10)


@pytest.mark.parametrize("force", [True, False])
def test_remove_container(adapter, force):
    """Test remove forwards the force flag."""
    adapter.remove_container("c1", force=force)
    adapter.client.containers.get.return_value.remove.assert_called_once_with(force=force)


def test_remove_container_defaults_to_force(adapter):
    """Test remove is forced by default."""
    adapter.remove_container("c1")
    adapter.client.containers.get.return_value.remove.assert_called_once_with(force=True)


def test_pull_image(adapter):
    """Test pull delegates to the images API."""
    adapter.pull_image("python:3.11")
    adapter.client.images.pull.assert_called_once_with("python:3.11")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="image present", config={"raises": False}, expected_output=True),
        TestCase(name="image missing", config={"raises": True}, expected_output=False),
    ],
)
def test_image_exists(adapter, test_case):
    """Test image_exists maps lookup failures to False."""
    if test_case.config["raises"]:
        adapter.client.images.get.side_effect = Exception("not found")

    assert adapter.image_exists("img") is test_case.expected_output
    adapter.client.images.get.assert_called_once_with("img")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="bytes output is decoded",
            expected_status=SUCCESS,
            config={"logs": b"out"},
            expected_output="out",
        ),
        TestCase(
            name="bytearray output is decoded",
            expected_status=SUCCESS,
            config={"logs": bytearray(b"out")},
            expected_output="out",
        ),
        TestCase(
            name="str output is returned as is",
            expected_status=SUCCESS,
            config={"logs": "out"},
            expected_output="out",
        ),
        TestCase(
            name="failure is wrapped in RuntimeError",
            expected_status=FAILED,
            config={"create_error": Exception("boom")},
            expected_error=RuntimeError,
        ),
    ],
)
def test_run_oneoff_container(adapter, test_case):
    """Test one-off containers return output and wrap failures."""
    if "create_error" in test_case.config:
        adapter.client.containers.create.side_effect = test_case.config["create_error"]
        with pytest.raises(test_case.expected_error, match="One-off container failed to run"):
            adapter.run_oneoff_container("img", ["echo"])
        return

    container = adapter.client.containers.create.return_value
    container.logs.return_value = test_case.config["logs"]

    assert adapter.run_oneoff_container("img", ["echo"]) == test_case.expected_output
    adapter.client.containers.create.assert_called_once_with(
        image="img", command=["echo"], detach=False, remove=True
    )
    container.start.assert_called_once()
    container.wait.assert_called_once()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="running container has no exit code",
            config={"status": "running", "attrs": {}},
            expected_output=("running", None),
        ),
        TestCase(
            name="exited container reports exit code",
            config={"status": "exited", "attrs": {"State": {"ExitCode": 3}}},
            expected_output=("exited", 3),
        ),
        TestCase(
            name="exited container without state has None exit code",
            config={"status": "exited", "attrs": {}},
            expected_output=("exited", None),
        ),
        TestCase(
            name="lookup failure returns unknown",
            config={"error": Exception("gone")},
            expected_output=("unknown", None),
        ),
    ],
)
def test_container_status(adapter, test_case):
    """Test container status and exit code reporting."""
    if "error" in test_case.config:
        adapter.client.containers.get.side_effect = test_case.config["error"]
    else:
        container = adapter.client.containers.get.return_value
        container.status = test_case.config["status"]
        container.attrs = test_case.config["attrs"]

    assert adapter.container_status("c1") == test_case.expected_output


def test_container_status_uses_inspect_without_attrs(adapter):
    """Test container.inspect() is used when `attrs` is unavailable."""
    container = Mock(spec=["status", "inspect"])
    container.status = "exited"
    container.inspect.return_value = {"State": {"ExitCode": 1}}
    adapter.client.containers.get.return_value = container

    assert adapter.container_status("c1") == ("exited", 1)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="exact network match",
            config={"networks": {"net1": {"IPAddress": "10.0.0.2"}}, "network_id": "net1"},
            expected_output="10.0.0.2",
        ),
        TestCase(
            name="falls back to first network with an IP",
            config={
                "networks": {"a": {"IPAddress": ""}, "b": {"IPAddress": "10.0.0.9"}},
                "network_id": "other",
            },
            expected_output="10.0.0.9",
        ),
        TestCase(
            name="no networks returns None",
            config={"networks": {}, "network_id": "net1"},
            expected_output=None,
        ),
        TestCase(
            name="lookup failure returns None",
            config={"error": Exception("gone"), "network_id": "net1"},
            expected_output=None,
        ),
    ],
)
def test_get_container_ip(adapter, test_case):
    """Test IP lookup by network with fallback behaviour."""
    if "error" in test_case.config:
        adapter.client.containers.get.side_effect = test_case.config["error"]
    else:
        container = adapter.client.containers.get.return_value
        container.attrs = {"NetworkSettings": {"Networks": test_case.config["networks"]}}

    result = adapter.get_container_ip("c1", test_case.config["network_id"])

    assert result == test_case.expected_output


def _fake_container(**kwargs):
    container = Mock()
    container.id = kwargs.get("id", "id1")
    container.name = kwargs.get("name", "name1")
    container.status = kwargs.get("status", "running")
    container.labels = kwargs.get("labels", {"k": "v"})
    container.attrs = kwargs.get("attrs", {"Created": "2025-01-01T00:00:00Z"})
    return container


def test_list_containers_formats_results(adapter):
    """Test containers are reloaded and converted to plain dicts."""
    container = _fake_container()
    adapter.client.containers.list.return_value = [container]

    result = adapter.list_containers(filters={"label": ["a=b", "c=d"]})

    assert result == [
        {
            "id": "id1",
            "name": "name1",
            "labels": {"k": "v"},
            "status": "running",
            "created": "2025-01-01T00:00:00Z",
        }
    ]
    container.reload.assert_called_once()


def test_list_containers_flattens_single_value_filters(adapter):
    """Test the podman-py workaround that unwraps single-item filter lists."""
    adapter.client.containers.list.return_value = []

    adapter.list_containers(filters={"label": ["a=b"], "name": ["x", "y"]})

    adapter.client.containers.list.assert_called_once_with(
        all=True, filters={"label": "a=b", "name": ["x", "y"]}
    )


def test_list_containers_returns_empty_on_error(adapter):
    """Test API failures are swallowed and an empty list is returned."""
    adapter.client.containers.list.side_effect = Exception("boom")

    assert adapter.list_containers(filters={"label": ["a=b"]}) == []


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="network info is mapped",
            config={"attrs": {"ID": "nid", "Name": "net1", "labels": {"x": "y"}}},
            expected_output={"id": "nid", "name": "net1", "labels": {"x": "y"}},
        ),
        TestCase(
            name="missing fields fall back to the requested id",
            config={"attrs": {}},
            expected_output={"id": "net1", "name": "net1", "labels": {}},
        ),
        TestCase(
            name="lookup failure returns None",
            config={"error": Exception("not found")},
            expected_output=None,
        ),
    ],
)
def test_get_network(adapter, test_case):
    """Test network info mapping and error handling."""
    if "error" in test_case.config:
        adapter.client.networks.get.side_effect = test_case.config["error"]
    else:
        adapter.client.networks.get.return_value.attrs = test_case.config["attrs"]

    assert adapter.get_network("net1") == test_case.expected_output


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="returns exit code",
            expected_status=SUCCESS,
            config={"wait_result": 0},
            expected_output=0,
        ),
        TestCase(
            name="non-zero exit code is returned",
            expected_status=SUCCESS,
            config={"wait_result": 2},
            expected_output=2,
        ),
        TestCase(
            name="timeout error becomes TimeoutError",
            expected_status=FAILED,
            config={"wait_error": Exception("Read timeout occurred")},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="other errors propagate unchanged",
            expected_status=FAILED,
            config={"wait_error": ValueError("bad")},
            expected_error=ValueError,
        ),
    ],
)
def test_wait_for_container(adapter, test_case):
    """Test wait returns the exit code and maps timeouts."""
    container = adapter.client.containers.get.return_value
    if "wait_error" in test_case.config:
        container.wait.side_effect = test_case.config["wait_error"]
        with pytest.raises(test_case.expected_error):
            adapter.wait_for_container("c1", timeout=5)
        return

    container.wait.return_value = test_case.config["wait_result"]

    assert adapter.wait_for_container("c1", timeout=5) == test_case.expected_output
    container.wait.assert_called_once_with(timeout=5)
