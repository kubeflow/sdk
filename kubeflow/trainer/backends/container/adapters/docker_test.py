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
Unit tests for DockerClientAdapter.

The Docker SDK client is mocked, so no Docker daemon or network is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.trainer.backends.container.adapters.docker import DockerClientAdapter
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.fixture
def client() -> MagicMock:
    """Mocked Docker SDK client injected into the adapter."""
    return MagicMock()


@pytest.fixture
def adapter(client: MagicMock) -> DockerClientAdapter:
    """DockerClientAdapter wired to a mocked Docker SDK client.

    Built with ``__new__`` so ``__init__`` never imports the real ``docker`` package: pytest puts
    this directory on ``sys.path``, where ``docker.py`` would shadow the SDK of the same name.
    """
    instance = DockerClientAdapter.__new__(DockerClientAdapter)
    instance.client = client
    return instance


def make_container(**attrs) -> MagicMock:
    container = MagicMock()
    for key, value in attrs.items():
        setattr(container, key, value)
    return container


def test_init_uses_environment_defaults_without_host() -> None:
    fake_docker = MagicMock()

    with patch.dict("sys.modules", {"docker": fake_docker}):
        adapter = DockerClientAdapter()

    fake_docker.from_env.assert_called_once_with()
    assert adapter.client is fake_docker.from_env.return_value
    assert adapter._runtime_type == "docker"


def test_init_uses_explicit_host() -> None:
    fake_docker = MagicMock()

    with patch.dict("sys.modules", {"docker": fake_docker}):
        adapter = DockerClientAdapter(host="unix:///custom.sock")

    fake_docker.DockerClient.assert_called_once_with(base_url="unix:///custom.sock")
    fake_docker.from_env.assert_not_called()
    assert adapter.client is fake_docker.DockerClient.return_value


def test_init_raises_helpful_error_when_docker_package_missing() -> None:
    with (
        patch.dict("sys.modules", {"docker": None}),
        pytest.raises(ImportError, match=r"pip install kubeflow\[docker\]"),
    ):
        DockerClientAdapter()


def test_create_network_skips_creation_when_network_exists(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    result = adapter.create_network("net", {"k": "v"})

    assert result == "net"
    client.networks.get.assert_called_once_with("net")
    client.networks.create.assert_not_called()


def test_create_network_creates_when_missing(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.networks.get.side_effect = Exception("not found")

    result = adapter.create_network("net", {"k": "v"})

    assert result == "net"
    client.networks.create.assert_called_once_with(
        name="net", check_duplicate=True, labels={"k": "v"}
    )


def test_delete_network_removes_network(adapter: DockerClientAdapter, client: MagicMock) -> None:
    network = MagicMock()
    client.networks.get.return_value = network

    adapter.delete_network("net")

    network.remove.assert_called_once_with()


def test_delete_network_ignores_errors(adapter: DockerClientAdapter, client: MagicMock) -> None:
    client.networks.get.side_effect = Exception("boom")

    adapter.delete_network("net")


def test_create_and_start_container_runs_detached_and_returns_id(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.containers.run.return_value = make_container(id="abc123")

    container_id = adapter.create_and_start_container(
        image="img:1",
        command=["python", "-c", "print(1)"],
        name="job-0",
        network_id="net",
        environment={"A": "1"},
        labels={"l": "v"},
        volumes={"/host": {"bind": "/c", "mode": "rw"}},
        working_dir="/work",
    )

    assert container_id == "abc123"
    client.containers.run.assert_called_once_with(
        image="img:1",
        command=("python", "-c", "print(1)"),
        name="job-0",
        detach=True,
        working_dir="/work",
        network="net",
        environment={"A": "1"},
        labels={"l": "v"},
        volumes={"/host": {"bind": "/c", "mode": "rw"}},
        auto_remove=False,
    )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="streaming logs decode bytes and keep str chunks",
            config={"follow": True, "logs": [b"line1\n", "line2\n"]},
            expected_output=["line1\n", "line2\n"],
        ),
        TestCase(
            name="non-streaming bytes logs are decoded",
            config={"follow": False, "logs": b"all logs"},
            expected_output=["all logs"],
        ),
        TestCase(
            name="non-streaming str logs are passed through",
            config={"follow": False, "logs": "all logs"},
            expected_output=["all logs"],
        ),
        TestCase(
            name="undecodable bytes are dropped instead of raising",
            config={"follow": False, "logs": b"ok\xff"},
            expected_output=["ok"],
        ),
    ],
)
def test_container_logs(
    test_case: TestCase, adapter: DockerClientAdapter, client: MagicMock
) -> None:
    container = make_container()
    container.logs.return_value = test_case.config["logs"]
    client.containers.get.return_value = container
    follow = test_case.config["follow"]

    output = list(adapter.container_logs("abc", follow=follow))

    assert output == test_case.expected_output
    container.logs.assert_called_once_with(stream=follow, follow=follow)


def test_image_exists_true_when_image_found(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    assert adapter.image_exists("img:1") is True
    client.images.get.assert_called_once_with("img:1")


def test_image_exists_false_when_lookup_fails(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.images.get.side_effect = Exception("not found")

    assert adapter.image_exists("img:1") is False


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="bytes output is decoded",
            config={"output": b"hello"},
            expected_output="hello",
        ),
        TestCase(
            name="str output is passed through",
            config={"output": "hello"},
            expected_output="hello",
        ),
        TestCase(
            name="bytearray output is decoded",
            config={"output": bytearray(b"hello")},
            expected_output="hello",
        ),
        TestCase(
            name="docker failure is wrapped in RuntimeError",
            expected_status=FAILED,
            config={"error": Exception("daemon down")},
            expected_error=RuntimeError,
        ),
    ],
)
def test_run_oneoff_container(
    test_case: TestCase, adapter: DockerClientAdapter, client: MagicMock
) -> None:
    if test_case.expected_status == SUCCESS:
        client.containers.run.return_value = test_case.config["output"]
        assert adapter.run_oneoff_container("img", ["echo", "hello"]) == test_case.expected_output
        client.containers.run.assert_called_once_with(
            image="img", command=("echo", "hello"), detach=False, remove=True
        )
    else:
        client.containers.run.side_effect = test_case.config["error"]
        with pytest.raises(test_case.expected_error, match="One-off container failed to run"):
            adapter.run_oneoff_container("img", ["echo", "hello"])


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="running container has no exit code",
            config={"status": "running", "attrs": {}},
            expected_output=("running", None),
        ),
        TestCase(
            name="exited container reports its exit code",
            config={"status": "exited", "attrs": {"State": {"ExitCode": 3}}},
            expected_output=("exited", 3),
        ),
        TestCase(
            name="exited container with exit code 0 reports success",
            config={"status": "exited", "attrs": {"State": {"ExitCode": 0}}},
            expected_output=("exited", 0),
        ),
        TestCase(
            name="exited container without state has no exit code",
            config={"status": "exited", "attrs": {}},
            expected_output=("exited", None),
        ),
    ],
)
def test_container_status(
    test_case: TestCase, adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.containers.get.return_value = make_container(
        status=test_case.config["status"], attrs=test_case.config["attrs"]
    )

    assert adapter.container_status("abc") == test_case.expected_output


def test_container_status_unknown_when_lookup_fails(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.containers.get.side_effect = Exception("gone")

    assert adapter.container_status("abc") == ("unknown", None)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="returns IP of the requested network",
            config={
                "networks": {"other": {"IPAddress": "10.0.0.2"}, "net": {"IPAddress": "10.0.0.9"}}
            },
            expected_output="10.0.0.9",
        ),
        TestCase(
            name="falls back to first network with an IP",
            config={"networks": {"a": {"IPAddress": ""}, "b": {"IPAddress": "10.0.0.5"}}},
            expected_output="10.0.0.5",
        ),
        TestCase(
            name="returns None when no network has an IP",
            config={"networks": {"a": {"IPAddress": ""}}},
            expected_output=None,
        ),
        TestCase(
            name="returns None when container has no networks",
            config={"networks": {}},
            expected_output=None,
        ),
    ],
)
def test_get_container_ip(
    test_case: TestCase, adapter: DockerClientAdapter, client: MagicMock
) -> None:
    container = make_container(
        attrs={"NetworkSettings": {"Networks": test_case.config["networks"]}}
    )
    client.containers.get.return_value = container

    assert adapter.get_container_ip("abc", "net") == test_case.expected_output
    container.reload.assert_called_once_with()


def test_get_container_ip_none_when_lookup_fails(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.containers.get.side_effect = Exception("gone")

    assert adapter.get_container_ip("abc", "net") is None


def test_list_containers_maps_fields_and_passes_filters(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    first = make_container(
        id="c1", labels={"job": "a"}, status="running", attrs={"Created": "2025-01-01"}
    )
    first.name = "first"
    second = make_container(id="c2", labels={}, status="exited", attrs={})
    second.name = "second"
    client.containers.list.return_value = [first, second]
    filters = {"label": ["job=a"]}

    result = adapter.list_containers(filters=filters)

    client.containers.list.assert_called_once_with(all=True, filters=filters)
    assert result == [
        {
            "id": "c1",
            "name": "first",
            "labels": {"job": "a"},
            "status": "running",
            "created": "2025-01-01",
        },
        {"id": "c2", "name": "second", "labels": {}, "status": "exited", "created": ""},
    ]


def test_list_containers_returns_empty_list_on_error(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.containers.list.side_effect = Exception("boom")

    assert adapter.list_containers() == []


def test_get_network_returns_summary(adapter: DockerClientAdapter, client: MagicMock) -> None:
    network = MagicMock(id="n1", attrs={"Labels": {"k": "v"}})
    network.name = "net"
    client.networks.get.return_value = network

    assert adapter.get_network("net") == {"id": "n1", "name": "net", "labels": {"k": "v"}}


def test_get_network_returns_none_when_lookup_fails(
    adapter: DockerClientAdapter, client: MagicMock
) -> None:
    client.networks.get.side_effect = Exception("not found")

    assert adapter.get_network("net") is None


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="dict result returns StatusCode",
            config={"result": {"StatusCode": 7}},
            expected_output=7,
        ),
        TestCase(
            name="dict result without StatusCode defaults to 0",
            config={"result": {}},
            expected_output=0,
        ),
        TestCase(
            name="non-dict result is coerced to int",
            config={"result": "4"},
            expected_output=4,
        ),
        TestCase(
            name="timeout error is raised as TimeoutError",
            expected_status=FAILED,
            config={"error": Exception("Read timeout expired")},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="other errors propagate unchanged",
            expected_status=FAILED,
            config={"error": ValueError("bad id")},
            expected_error=ValueError,
        ),
    ],
)
def test_wait_for_container(
    test_case: TestCase, adapter: DockerClientAdapter, client: MagicMock
) -> None:
    container = make_container()
    client.containers.get.return_value = container

    if test_case.expected_status == SUCCESS:
        container.wait.return_value = test_case.config["result"]
        assert adapter.wait_for_container("abc", timeout=5) == test_case.expected_output
        container.wait.assert_called_once_with(timeout=5)
    else:
        container.wait.side_effect = test_case.config["error"]
        with pytest.raises(test_case.expected_error):
            adapter.wait_for_container("abc", timeout=5)
