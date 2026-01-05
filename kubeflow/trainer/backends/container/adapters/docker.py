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
Docker client adapter implementation.

This module provides the DockerClientAdapter class that implements the
BaseContainerClientAdapter interface for Docker runtime.
"""

from collections.abc import Iterator
from typing import Optional

from kubeflow.trainer.backends.container.adapters.base import BaseContainerClientAdapter


class DockerClientAdapter(BaseContainerClientAdapter):
    """Adapter for Docker client."""

    def __init__(self, host: Optional[str] = None):
        """
        Initialize Docker client.

        Args:
            host: Docker host URL, or None to use environment defaults
        """
        try:
            import docker  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'docker' Python package is not installed. Install with extras: "
                "pip install kubeflow[docker]"
            ) from e

        if host:
            self.client = docker.DockerClient(base_url=host)
        else:
            self.client = docker.from_env()

        self._runtime_type = "docker"

    def ping(self):
        """Test connection to Docker daemon."""
        self.client.ping()

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        """Create a Docker network."""
        try:
            self.client.networks.get(name)
            return name
        except Exception:
            pass

        self.client.networks.create(
            name=name,
            check_duplicate=True,
            labels=labels,
        )
        return name

    def delete_network(self, network_id: str):
        """Delete Docker network."""
        try:
            net = self.client.networks.get(network_id)
            net.remove()
        except Exception:
            pass

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
        """Create and start a Docker container."""
        container = self.client.containers.run(
            image=image,
            command=tuple(command),
            name=name,
            detach=True,
            working_dir=working_dir,
            network=network_id,
            environment=environment,
            labels=labels,
            volumes=volumes,
            auto_remove=False,
        )
        return container.id

    def get_container(self, container_id: str):
        """Get Docker container by ID."""
        return self.client.containers.get(container_id)

    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from Docker container."""
        container = self.get_container(container_id)
        logs = container.logs(stream=bool(follow), follow=bool(follow))
        if follow:
            for chunk in logs:
                if isinstance(chunk, bytes):
                    yield chunk.decode("utf-8", errors="ignore")
                else:
                    yield str(chunk)
        else:
            if isinstance(logs, bytes):
                yield logs.decode("utf-8", errors="ignore")
            else:
                yield str(logs)

    def stop_container(self, container_id: str, timeout: int = 10):
        """Stop Docker container."""
        container = self.get_container(container_id)
        container.stop(timeout=timeout)

    def remove_container(self, container_id: str, force: bool = True):
        """Remove Docker container."""
        container = self.get_container(container_id)
        container.remove(force=force)

    def pull_image(self, image: str):
        """Pull Docker image."""
        self.client.images.pull(image)

    def image_exists(self, image: str) -> bool:
        """Check if Docker image exists locally."""
        try:
            self.client.images.get(image)
            return True
        except Exception:
            return False

    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        """Run a short-lived Docker container and return output."""
        try:
            output = self.client.containers.run(
                image=image,
                command=tuple(command),
                detach=False,
                remove=True,
            )
            if isinstance(output, (bytes, bytearray)):
                return output.decode("utf-8", errors="ignore")
            return str(output)
        except Exception as e:
            raise RuntimeError(f"One-off container failed to run: {e}") from e

    def container_status(self, container_id: str) -> tuple[str, Optional[int]]:
        """Get Docker container status."""
        try:
            container = self.get_container(container_id)
            status = container.status
            # Get exit code if container has exited
            exit_code = None
            if status == "exited":
                inspect = container.attrs if hasattr(container, "attrs") else container.inspect()
                exit_code = inspect.get("State", {}).get("ExitCode")
            return (status, exit_code)
        except Exception:
            return ("unknown", None)

    def get_container_ip(self, container_id: str, network_id: str) -> Optional[str]:
        """Get container's IP address on a specific network."""
        try:
            container = self.get_container(container_id)
            # Refresh container info
            container.reload()
            # Get network settings
            networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})

            # Try to find the network by exact name or ID
            if network_id in networks:
                return networks[network_id].get("IPAddress")

            # Fallback: return first available IP
            for _net_name, net_info in networks.items():
                ip = net_info.get("IPAddress")
                if ip:
                    return ip

            return None
        except Exception:
            return None

    def list_containers(self, filters: Optional[dict[str, list[str]]] = None) -> list[dict]:
        """List Docker containers with optional filters."""
        try:
            containers = self.client.containers.list(all=True, filters=filters)
            result = []
            for c in containers:
                result.append(
                    {
                        "id": c.id,
                        "name": c.name,
                        "labels": c.labels,
                        "status": c.status,
                        "created": c.attrs.get("Created", ""),
                    }
                )
            return result
        except Exception:
            return []

    def get_network(self, network_id: str) -> Optional[dict]:
        """Get Docker network information."""
        try:
            network = self.client.networks.get(network_id)
            return {
                "id": network.id,
                "name": network.name,
                "labels": network.attrs.get("Labels", {}),
            }
        except Exception:
            return None

    def wait_for_container(self, container_id: str, timeout: Optional[int] = None) -> int:
        """
        Wait for a Docker container to exit and return its exit code.

        Args:
            container_id: Container ID
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            Container exit code

        Raises:
            TimeoutError: If timeout is reached before container exits
        """
        try:
            container = self.get_container(container_id)
            result = container.wait(timeout=timeout)
            # Docker wait() returns a dict with 'StatusCode' key
            if isinstance(result, dict):
                return result.get("StatusCode", 0)
            return int(result)
        except Exception as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(
                    f"Container {container_id} did not exit within {timeout} seconds"
                ) from e
            raise
