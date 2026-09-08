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

"""Tests for PodmanClientAdapter."""

from unittest.mock import Mock

from kubeflow.trainer.backends.container.adapters.podman import PodmanClientAdapter


def test_list_containers_without_filters():
    """Test that list_containers() with no filters lists all containers."""
    adapter = PodmanClientAdapter.__new__(PodmanClientAdapter)
    container = Mock()
    container.id = "abc123"
    container.name = "job-0"
    container.labels = {"app": "trainjob"}
    container.status = "running"
    container.attrs = {"Created": "2025-01-01T00:00:00Z"}
    adapter.client = Mock()
    adapter.client.containers.list.return_value = [container]

    result = adapter.list_containers()

    adapter.client.containers.list.assert_called_once_with(all=True, filters={})
    assert result == [
        {
            "id": "abc123",
            "name": "job-0",
            "labels": {"app": "trainjob"},
            "status": "running",
            "created": "2025-01-01T00:00:00Z",
        }
    ]


def test_list_containers_with_filters():
    """Test that a single-value label filter is collapsed and forwarded to Podman."""
    adapter = PodmanClientAdapter.__new__(PodmanClientAdapter)
    container = Mock()
    container.id = "abc123"
    container.name = "job-0"
    container.labels = {"app": "trainjob"}
    container.status = "running"
    container.attrs = {"Created": "2025-01-01T00:00:00Z"}
    adapter.client = Mock()
    adapter.client.containers.list.return_value = [container]

    result = adapter.list_containers(filters={"label": ["app=trainjob"]})

    adapter.client.containers.list.assert_called_once_with(
        all=True, filters={"label": "app=trainjob"}
    )
    assert result == [
        {
            "id": "abc123",
            "name": "job-0",
            "labels": {"app": "trainjob"},
            "status": "running",
            "created": "2025-01-01T00:00:00Z",
        }
    ]
