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
Container client adapters for Docker and Podman.

This module implements the adapter pattern to abstract away differences between
Docker and Podman APIs, allowing the backend to work with either runtime through
a common interface.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Optional


class BaseContainerClientAdapter(abc.ABC):
    """
    Abstract adapter interface for container clients.

    This adapter abstracts the container runtime API, allowing the backend
    to work with Docker and Podman through a unified interface.
    """

    @abc.abstractmethod
    def ping(self):
        """Test the connection to the container runtime."""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_network(
        self,
        name: str,
        labels: dict[str, str],
    ) -> str:
        """
        Create a container network.

        Args:
            name: Network name
            labels: Labels to attach to the network

        Returns:
            Network ID or name
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_network(self, network_id: str):
        """Delete a network."""
        raise NotImplementedError()

    @abc.abstractmethod
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
        """
        Create and start a container.

        Args:
            image: Container image
            command: Command to run
            name: Container name
            network_id: Network to attach to
            environment: Environment variables
            labels: Container labels
            volumes: Volume mounts
            working_dir: Working directory

        Returns:
            Container ID
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_container(self, container_id: str):
        """Get container object by ID."""
        raise NotImplementedError()

    @abc.abstractmethod
    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def stop_container(self, container_id: str, timeout: int = 10):
        """Stop a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_container(self, container_id: str, force: bool = True):
        """Remove a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def pull_image(self, image: str):
        """Pull an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def image_exists(self, image: str) -> bool:
        """Check if an image exists locally."""
        raise NotImplementedError()

    @abc.abstractmethod
    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        """
        Run a short-lived container and return its output.

        Args:
            image: Container image
            command: Command to run

        Returns:
            Container output as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def container_status(self, container_id: str) -> tuple[str, Optional[int]]:
        """
        Get container status.

        Returns:
            Tuple of (status_string, exit_code)
            Status strings: "running", "created", "exited", etc.
            Exit code is None if container hasn't exited
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_container_ip(self, container_id: str, network_id: str) -> Optional[str]:
        """
        Get container's IP address on a specific network.

        Args:
            container_id: Container ID
            network_id: Network name or ID

        Returns:
            IP address string or None if not found
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list_containers(self, filters: Optional[dict[str, list[str]]] = None) -> list[dict]:
        """
        List containers, optionally filtered by labels.

        Args:
            filters: Dictionary of filters (e.g., {"label": ["key=value"]})

        Returns:
            List of container info dictionaries with keys:
            - id: Container ID
            - name: Container name
            - labels: Dictionary of labels
            - status: Container status
            - created: Creation timestamp
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_network(self, network_id: str) -> Optional[dict]:
        """
        Get network information by ID or name.

        Args:
            network_id: Network ID or name

        Returns:
            Dictionary with network info including labels, or None if not found
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_container(self, container_id: str, timeout: Optional[int] = None) -> int:
        """
        Wait for a container to exit and return its exit code.

        This is a blocking call that waits until the container stops.

        Args:
            container_id: Container ID
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            Container exit code

        Raises:
            TimeoutError: If timeout is reached before container exits
        """
        raise NotImplementedError()
