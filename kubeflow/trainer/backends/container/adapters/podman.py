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
Podman client adapter implementation.

This module provides the PodmanClientAdapter class that implements the
BaseContainerClientAdapter interface for Podman runtime.

Key differences from Docker:
- Uses DNS-enabled bridge networks for better container name resolution
- GPU support via CDI (Container Device Interface) instead of NVIDIA Container Toolkit
- Slightly different API for some operations (e.g., container.create + start pattern)
"""

from collections.abc import Iterator
from typing import Optional

from kubeflow.trainer.backends.container.adapters.base import BaseContainerClientAdapter


class PodmanClientAdapter(BaseContainerClientAdapter):
    """Adapter for Podman client."""

    def __init__(self, host: Optional[str] = None):
        """
        Initialize Podman client.

        Args:
            host: Podman host URL, or None to use environment defaults
        """
        try:
            import podman  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'podman' Python package is not installed. Install with extras: "
                "pip install kubeflow[podman]"
            ) from e

        if host:
            self.client = podman.PodmanClient(base_url=host)
        else:
            self.client = podman.PodmanClient()

        self._runtime_type = "podman"

    def ping(self):
        """Test connection to Podman."""
        self.client.ping()

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        """Create a Podman network with DNS enabled."""
        try:
            self.client.networks.get(name)
            return name
        except Exception:
            pass

        self.client.networks.create(
            name=name,
            driver="bridge",
            dns_enabled=True,
            labels=labels,
        )
        return name

    def delete_network(self, network_id: str):
        """Delete Podman network."""
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
        """Create and start a Podman container."""
        container = self.client.containers.run(
            image=image,
            command=command,
            name=name,
            network=network_id,
            working_dir=working_dir,
            environment=environment,
            labels=labels,
            volumes=volumes,
            detach=True,
            remove=False,
        )
        return container.id

    def get_container(self, container_id: str):
        """Get Podman container by ID."""
        return self.client.containers.get(container_id)

    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from Podman container."""
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
        """Stop Podman container."""
        container = self.get_container(container_id)
        container.stop(timeout=timeout)

    def remove_container(self, container_id: str, force: bool = True):
        """Remove Podman container."""
        container = self.get_container(container_id)
        container.remove(force=force)

    def pull_image(self, image: str):
        """Pull Podman image."""
        self.client.images.pull(image)

    def image_exists(self, image: str) -> bool:
        """Check if Podman image exists locally."""
        try:
            self.client.images.get(image)
            return True
        except Exception:
            return False

    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        """Run a short-lived Podman container and return output."""
        try:
            container = self.client.containers.create(
                image=image,
                command=command,
                detach=False,
                remove=True,
            )
            container.start()
            container.wait()
            logs = container.logs()

            if isinstance(logs, (bytes, bytearray)):
                return logs.decode("utf-8", errors="ignore")
            return str(logs)
        except Exception as e:
            raise RuntimeError(f"One-off container failed to run: {e}") from e

    def container_status(self, container_id: str) -> tuple[str, Optional[int]]:
        """Get Podman container status."""
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
            # Get container inspect data
            inspect = container.attrs if hasattr(container, "attrs") else container.inspect()

            # Get network settings - Podman structure is similar to Docker
            networks = inspect.get("NetworkSettings", {}).get("Networks", {})

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
        """List Podman containers with optional filters."""
        # Work-around for https://github.com/containers/podman-py/issues/542
        for k, v in filters.items():
            if len(v) == 1:
                filters[k] = v[0]
        try:
            containers = self.client.containers.list(all=True, filters=filters)
            result = []
            for c in containers:
                # The container status needs to be reloaded when the container
                # is retrieved from a list.
                # See: https://github.com/containers/podman-py/issues/446
                c.reload()
                inspect = c.attrs if hasattr(c, "attrs") else c.inspect()
                labels = (
                    c.labels
                    if hasattr(c, "labels")
                    else inspect.get("Config", {}).get("Labels", {})
                )
                result.append(
                    {
                        "id": c.id,
                        "name": c.name,
                        "labels": labels,
                        "status": c.status,
                        "created": inspect.get("Created", ""),
                    }
                )
            return result
        except Exception:
            return []

    def get_network(self, network_id: str) -> Optional[dict]:
        """Get Podman network information."""
        try:
            network = self.client.networks.get(network_id)
            inspect = network.attrs if hasattr(network, "attrs") else network.inspect()
            return {
                "id": inspect.get("ID", network_id),
                "name": inspect.get("Name", network_id),
                # The case is important for labels
                "labels": inspect.get("labels", {}),
            }
        except Exception:
            return None
