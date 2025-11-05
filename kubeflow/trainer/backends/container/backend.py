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
ContainerBackend
----------------

Unified local execution backend for `CustomTrainer` jobs using containers.

This backend automatically detects and uses either Docker or Podman.
It provides a single interface regardless of the underlying container runtime.

Key behaviors:
- Auto-detection: Tries Docker first, then Podman. Can be overridden via config.
- Multi-node jobs: one container per node connected via a per-job network.
- Entry script generation: we serialize the user's training function and embed it
  inline in the container command using a heredoc (no file I/O on the host). The
  script is created inside the container at /tmp/train.py and invoked using
  `torchrun` (preferred) or `python` as a fallback.
- Runtimes: we use `config/training_runtimes` to define runtime images and
  characteristics (e.g., torch). Defaults to `torch-distributed` if no runtime
  is provided.
- Image pulling: controlled via `pull_policy` and performed automatically if
  needed.
- Logs and lifecycle: streaming logs and deletion semantics similar to the
  Docker/Podman backends, but with automatic runtime detection.
"""

from collections.abc import Iterator
from datetime import datetime
import logging
import os
import random
import shutil
import string
from typing import Optional, Union
import uuid

from kubeflow.trainer.backends.base import RuntimeBackend
from kubeflow.trainer.backends.container import utils as container_utils
from kubeflow.trainer.backends.container.adapters.base import (
    BaseContainerClientAdapter,
)
from kubeflow.trainer.backends.container.adapters.docker import DockerClientAdapter
from kubeflow.trainer.backends.container.adapters.podman import PodmanClientAdapter
from kubeflow.trainer.backends.container.runtime_loader import (
    get_training_runtime_from_sources,
    list_training_runtimes_from_sources,
)
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class ContainerBackend(RuntimeBackend):
    """
    Unified container backend that auto-detects Docker or Podman.

    This backend uses the adapter pattern to abstract away differences between
    Docker and Podman, providing a single consistent interface.
    """

    def __init__(self, cfg: ContainerBackendConfig):
        self.cfg = cfg
        self.label_prefix = "trainer.kubeflow.org"

        # Initialize the container client adapter
        self._adapter = self._create_adapter()

    def _get_common_socket_locations(self, runtime_name: str) -> list[Optional[str]]:
        """
        Get common socket locations to try for the given runtime.

        Args:
            runtime_name: "docker" or "podman"

        Returns:
            List of socket URLs to try, including None (for default)
        """
        import os
        from pathlib import Path

        locations = [self.cfg.container_host] if self.cfg.container_host else []

        if runtime_name == "docker":
            # Common Docker socket locations
            colima_sock = Path.home() / ".colima/default/docker.sock"
            if colima_sock.exists():
                locations.append(f"unix://{colima_sock}")
            # Standard Docker socket
            locations.append(None)  # Use docker.from_env() default

        elif runtime_name == "podman":
            # Common Podman socket locations on macOS
            uid = os.getuid() if hasattr(os, "getuid") else None
            if uid:
                user_sock = f"/run/user/{uid}/podman/podman.sock"
                if Path(user_sock).exists():
                    locations.append(f"unix://{user_sock}")
            # Standard Podman socket
            locations.append(None)  # Use PodmanClient() default

        # Remove duplicates while preserving order
        seen = set()
        unique_locations = []
        for loc in locations:
            if loc not in seen:
                unique_locations.append(loc)
                seen.add(loc)

        return unique_locations

    def _create_adapter(self) -> BaseContainerClientAdapter:
        """
        Create the appropriate container client adapter.

        Tries Docker first, then Podman if Docker fails, unless a specific
        runtime is requested in the config. Automatically tries common socket
        locations (e.g., Colima for Docker on macOS, user socket for Podman).

        Raises RuntimeError if neither Docker nor Podman are available.
        """
        runtime_map = {
            "docker": DockerClientAdapter,
            "podman": PodmanClientAdapter,
        }

        # Determine which runtimes to try
        runtimes_to_try = (
            [self.cfg.container_runtime] if self.cfg.container_runtime else ["docker", "podman"]
        )

        attempted_connections = []
        last_error = None

        for runtime_name in runtimes_to_try:
            if runtime_name not in runtime_map:
                continue

            # Try common socket locations for this runtime
            socket_locations = self._get_common_socket_locations(runtime_name)

            for host in socket_locations:
                try:
                    adapter = runtime_map[runtime_name](host)
                    adapter.ping()
                    host_display = host or "default"
                    logger.debug(
                        f"Using {runtime_name} as container runtime (host: {host_display})"
                    )
                    return adapter
                except Exception as e:
                    host_str = host or "default"
                    logger.debug(f"{runtime_name} initialization failed at {host_str}: {e}")
                    attempted_connections.append(f"{runtime_name} at {host_str}")
                    last_error = e

        # Build helpful error message
        import platform

        system = platform.system()

        attempted = ", ".join(attempted_connections)
        error_msg = f"Could not connect to Docker or Podman (tried: {attempted}).\n"

        if system == "Darwin":  # macOS
            error_msg += (
                "Ensure Docker/Podman is running "
                "(e.g., 'colima start' or 'podman machine start').\n"
            )
        else:
            error_msg += "Ensure Docker/Podman is installed and running.\n"

        error_msg += (
            "To specify a custom socket: ContainerBackendConfig(container_host='unix:///path/to/socket')\n"
            "Or use LocalProcessBackendConfig for non-containerized execution."
        )

        raise RuntimeError(error_msg) from last_error

    @property
    def _runtime_type(self) -> str:
        """Get the runtime type for debugging/logging."""
        return self._adapter._runtime_type

    # ---- Runtime APIs ----
    def list_runtimes(self) -> list[types.Runtime]:
        return list_training_runtimes_from_sources(self.cfg.runtime_source.sources)

    def get_runtime(self, name: str) -> types.Runtime:
        return get_training_runtime_from_sources(name, self.cfg.runtime_source.sources)

    def get_runtime_packages(self, runtime: types.Runtime):
        """
        Spawn a short-lived container to report Python version, pip list, and nvidia-smi.
        """
        container_utils.maybe_pull_image(self._adapter, runtime.trainer.image, self.cfg.pull_policy)

        command = [
            "bash",
            "-lc",
            "python -c \"import sys; print(f'Python: {sys.version}')\" && "
            "(pip list || echo 'pip not found') && "
            "(nvidia-smi || echo 'nvidia-smi not found')",
        ]

        logs = self._adapter.run_oneoff_container(image=runtime.trainer.image, command=command)
        print(logs)

    def train(
        self,
        runtime: Optional[types.Runtime] = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[
            Union[types.CustomTrainer, types.CustomTrainerContainer, types.BuiltinTrainer]
        ] = None,
        options: Optional[list] = None,
    ) -> str:
        if runtime is None:
            runtime = self.get_runtime("torch-distributed")

        # Process options to extract configuration
        name = None
        if options:
            job_spec = {}
            for option in options:
                option(job_spec, trainer, self)

            metadata_section = job_spec.get("metadata", {})
            name = metadata_section.get("name")

        if not isinstance(trainer, types.CustomTrainer):
            raise ValueError(f"{self.__class__.__name__} supports only CustomTrainer in v1")

        # Generate train job name if not provided via options
        trainjob_name = name or (
            random.choice(string.ascii_lowercase)
            + uuid.uuid4().hex[: constants.JOB_NAME_UUID_LENGTH]
        )

        logger.debug(f"Starting training job: {trainjob_name}")
        try:
            # Create per-job working directory on host (for outputs, checkpoints, etc.)
            workdir = container_utils.create_workdir(trainjob_name)
            logger.debug(f"Created working directory: {workdir}")

            # Generate training script code (inline, not written to disk)
            training_script_code = container_utils.get_training_script_code(trainer)
            logger.debug("Generated training script code")

            # Resolve image and pull if needed
            logger.debug(f"Using image: {runtime.trainer.image}")

            container_utils.maybe_pull_image(
                self._adapter, runtime.trainer.image, self.cfg.pull_policy
            )
            logger.debug(f"Image ready: {runtime.trainer.image}")

            # Build base environment
            env = container_utils.build_environment(trainer)

            # Construct pre-run command to install packages
            pre_install_cmd = container_utils.build_pip_install_cmd(trainer)

            # Create network for multi-node communication
            num_nodes = trainer.num_nodes or runtime.trainer.num_nodes or 1
            logger.debug(f"Creating network for {num_nodes} nodes")

            # Determine number of processes per node from GPU count
            # For GPU training: spawn one process per GPU for optimal utilization
            # For CPU training: use single process (PyTorch parallelizes internally via threads)
            nproc_per_node = 1  # Default for CPU training
            if trainer.resources_per_node and "gpu" in trainer.resources_per_node:
                try:
                    nproc_per_node = int(trainer.resources_per_node["gpu"])
                    logger.debug(f"Using {nproc_per_node} processes per node (1 per GPU)")
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid GPU count in resources_per_node: "
                        f"{trainer.resources_per_node['gpu']}, defaulting to 1 process per node"
                    )
            else:
                logger.debug("No GPU specified, using 1 process per node")

            network_id = self._adapter.create_network(
                name=f"{trainjob_name}-net",
                labels={
                    f"{self.label_prefix}/trainjob-name": trainjob_name,
                    f"{self.label_prefix}/runtime-name": runtime.name,
                    f"{self.label_prefix}/workdir": workdir,
                },
            )
            logger.debug(f"Created network: {network_id}")

            # Create N containers (one per node)
            container_ids: list[str] = []
            master_container_id = None
            master_ip = None

            for rank in range(num_nodes):
                container_name = f"{trainjob_name}-node-{rank}"

                # Get master address and port for torchrun
                master_port = 29500

                # For Podman: use IP address to avoid DNS timing issues
                # For Docker: use hostname (DNS is reliable)
                if rank == 0:
                    # Master node - will be created first
                    master_addr = f"{trainjob_name}-node-0"
                else:
                    # Worker nodes - determine master address based on runtime
                    if self._runtime_type == "podman" and master_ip:
                        master_addr = master_ip
                        logger.debug(f"Using master IP address for Podman: {master_ip}")
                    else:
                        master_addr = f"{trainjob_name}-node-0"
                        logger.debug(f"Using master hostname: {master_addr}")

                # Prefer torchrun; fall back to python if torchrun is unavailable
                # For worker nodes, wait for master to be reachable before starting torchrun
                wait_for_master = ""
                if rank > 0:
                    wait_for_master = (
                        f"echo 'Waiting for master node {master_addr}:{master_port}...'; "
                        f"for i in {{1..60}}; do "
                        f"  if timeout 1 bash -c 'cat < /dev/null > "
                        f"/dev/tcp/{master_addr}/{master_port}' 2>/dev/null; then "
                        f"    echo 'Master node is reachable'; break; "
                        f"  fi; "
                        f"  if [ $i -eq 60 ]; then "
                        f"echo 'Timeout waiting for master node'; exit 1; fi; "
                        f"  sleep 2; "
                        f"done; "
                    )

                # Embed training script inline using heredoc (no file I/O on host)
                entry_cmd = (
                    f"{pre_install_cmd}"
                    f"{wait_for_master}"
                    f"cat > /tmp/train.py << 'TRAINING_SCRIPT_EOF'\n"
                    f"{training_script_code}\n"
                    f"TRAINING_SCRIPT_EOF\n"
                    "if command -v torchrun >/dev/null 2>&1; then "
                    f"  torchrun --nproc_per_node={nproc_per_node} --nnodes={num_nodes} "
                    f"  --node-rank={rank} --rdzv-backend=c10d "
                    f"  --rdzv-endpoint={master_addr}:{master_port} "
                    f"  /tmp/train.py; "
                    "else "
                    f"  python /tmp/train.py; "
                    "fi"
                )

                full_cmd = ["bash", "-lc", entry_cmd]

                labels = {
                    f"{self.label_prefix}/trainjob-name": trainjob_name,
                    f"{self.label_prefix}/step": f"node-{rank}",
                    f"{self.label_prefix}/network-id": network_id,
                }

                volumes = {
                    workdir: {
                        "bind": constants.WORKSPACE_PATH,
                        "mode": "rw",
                    }
                }

                logger.debug(f"Creating container {rank}/{num_nodes}: {container_name}")

                container_id = self._adapter.create_and_start_container(
                    image=runtime.trainer.image,
                    command=full_cmd,
                    name=container_name,
                    network_id=network_id,
                    environment=env,
                    labels=labels,
                    volumes=volumes,
                    working_dir=constants.WORKSPACE_PATH,
                )

                logger.debug(f"Started container {container_name} (ID: {container_id[:12]})")
                container_ids.append(container_id)

                # If this is the master node and we're using Podman, get its IP address
                if rank == 0:
                    master_container_id = container_id
                    if self._runtime_type == "podman":
                        # Get master IP for worker nodes to use
                        master_ip = self._adapter.get_container_ip(master_container_id, network_id)
                        if master_ip:
                            logger.debug(f"Master node IP address: {master_ip}")
                        else:
                            logger.warning(
                                "Could not retrieve master IP address. "
                                "Worker nodes will fall back to DNS resolution."
                            )

            logger.debug(
                f"Training job {trainjob_name} created successfully with "
                f"{len(container_ids)} container(s)"
            )
            return trainjob_name

        except Exception as e:
            # Clean up on failure
            logger.error(f"Failed to create training job {trainjob_name}: {e}")
            logger.exception("Full traceback:")

            # Try to clean up any resources that were created
            from contextlib import suppress

            try:
                # Stop and remove any containers that were created
                if "container_ids" in locals():
                    for container_id in container_ids:
                        with suppress(Exception):
                            self._adapter.stop_container(container_id, timeout=5)
                            self._adapter.remove_container(container_id, force=True)

                # Remove network if it was created
                if "network_id" in locals():
                    with suppress(Exception):
                        self._adapter.delete_network(network_id)

                # Remove working directory if it was created
                if "workdir" in locals() and os.path.isdir(workdir):
                    shutil.rmtree(workdir, ignore_errors=True)

            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")

            # Re-raise the original exception
            raise

    def _get_job_containers(self, name: str) -> list[dict]:
        """
        Get containers for a specific training job.

        Args:
            name: Name of the training job

        Returns:
            List of container dictionaries for this job

        Raises:
            ValueError: If no containers found for the job
        """
        filters = {"label": [f"{self.label_prefix}/trainjob-name={name}"]}
        containers = self._adapter.list_containers(filters=filters)

        if not containers:
            raise ValueError(f"No TrainJob with name {name}")

        return containers

    def __get_trainjob_from_containers(
        self, job_name: str, containers: list[dict]
    ) -> types.TrainJob:
        """
        Build a TrainJob object from a list of containers.

        Args:
            job_name: Name of the training job
            containers: List of container dictionaries for this job

        Returns:
            TrainJob object

        Raises:
            ValueError: If network metadata is missing or runtime not found
        """
        if not containers:
            raise ValueError(f"No containers found for TrainJob {job_name}")

        # Get metadata from network
        network_id = containers[0]["labels"].get(f"{self.label_prefix}/network-id")
        if not network_id:
            raise ValueError(f"TrainJob {job_name} is missing network metadata")

        network_info = self._adapter.get_network(network_id)
        if not network_info:
            raise ValueError(f"TrainJob {job_name} network not found")

        network_labels = network_info.get("labels", {})
        runtime_name = network_labels.get(f"{self.label_prefix}/runtime-name")

        # Get runtime object
        try:
            job_runtime = self.get_runtime(runtime_name) if runtime_name else None
        except Exception as e:
            raise ValueError(f"Runtime {runtime_name} not found for job {job_name}") from e

        if not job_runtime:
            raise ValueError(f"Runtime {runtime_name} not found for job {job_name}")

        # Parse creation timestamp from first container
        created_str = containers[0].get("created", "")
        try:
            from dateutil import parser

            creation_timestamp = parser.isoparse(created_str)
        except Exception:
            creation_timestamp = datetime.now()

        # Build steps from containers
        steps = []
        for container in sorted(containers, key=lambda c: c["name"]):
            step_name = container["labels"].get(f"{self.label_prefix}/step", "")
            steps.append(
                types.Step(
                    name=step_name,
                    pod_name=container["name"],
                    status=container_utils.get_container_status(self._adapter, container["id"]),
                )
            )

        # Get num_nodes from container count
        num_nodes = len(containers)

        return types.TrainJob(
            name=job_name,
            creation_timestamp=creation_timestamp,
            runtime=job_runtime,
            steps=steps,
            num_nodes=num_nodes,
            status=container_utils.aggregate_container_statuses(self._adapter, containers),
        )

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> list[types.TrainJob]:
        """List all training jobs by querying container runtime."""
        # Get all containers with our label prefix
        filters = {"label": [f"{self.label_prefix}/trainjob-name"]}
        containers = self._adapter.list_containers(filters=filters)

        # Group containers by job name
        jobs_map: dict[str, list[dict]] = {}
        for container in containers:
            job_name = container["labels"].get(f"{self.label_prefix}/trainjob-name")
            if job_name:
                if job_name not in jobs_map:
                    jobs_map[job_name] = []
                jobs_map[job_name].append(container)

        result: list[types.TrainJob] = []
        for job_name, job_containers in jobs_map.items():
            # Skip jobs with no containers
            if not job_containers:
                continue

            # Filter by runtime if specified
            if runtime:
                network_id = job_containers[0]["labels"].get(f"{self.label_prefix}/network-id")
                if network_id:
                    network_info = self._adapter.get_network(network_id)
                    if network_info:
                        network_labels = network_info.get("labels", {})
                        runtime_name = network_labels.get(f"{self.label_prefix}/runtime-name")
                        if runtime_name != runtime.name:
                            continue

            # Build TrainJob from containers
            try:
                result.append(self.__get_trainjob_from_containers(job_name, job_containers))
            except Exception as e:
                logger.warning(f"Failed to get TrainJob {job_name}: {e}")
                continue

        return result

    def get_job(self, name: str) -> types.TrainJob:
        """Get a specific training job by querying container runtime."""
        containers = self._get_job_containers(name)
        return self.__get_trainjob_from_containers(name, containers)

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        """Get logs for a training job by querying container runtime."""
        containers = self._get_job_containers(name)

        want_all = step == constants.NODE + "-0"
        for container in sorted(containers, key=lambda c: c["name"]):
            container_step = container["labels"].get(f"{self.label_prefix}/step", "")
            if not want_all and container_step != step:
                continue
            try:
                yield from self._adapter.container_logs(container["id"], follow)
            except Exception as e:
                logger.warning(f"Failed to get logs for {container['name']}: {e}")
                yield f"Error getting logs: {e}\n"

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJob:
        import time

        end = time.time() + timeout
        while time.time() < end:
            tj = self.get_job(name)
            logger.debug(f"TrainJob {name}, status {tj.status}")
            if tj.status in status:
                return tj
            if constants.TRAINJOB_FAILED not in status and tj.status == constants.TRAINJOB_FAILED:
                raise RuntimeError(f"TrainJob {name} is Failed")
            time.sleep(polling_interval)
        raise TimeoutError(f"Timeout waiting for TrainJob {name} to reach status: {status}")

    def delete_job(self, name: str):
        """Delete a training job by querying container runtime."""
        containers = self._get_job_containers(name)

        # Get network_id and workdir from labels
        network_id = containers[0]["labels"].get(f"{self.label_prefix}/network-id")

        # Get workdir from network labels
        workdir_host = None
        if network_id:
            network_info = self._adapter.get_network(network_id)
            if network_info:
                network_labels = network_info.get("labels", {})
                workdir_host = network_labels.get(f"{self.label_prefix}/workdir")

        # Stop containers and remove
        from contextlib import suppress

        for container in containers:
            with suppress(Exception):
                self._adapter.stop_container(container["id"], timeout=10)
            with suppress(Exception):
                self._adapter.remove_container(container["id"], force=True)

        # Remove network (best-effort)
        if network_id:
            with suppress(Exception):
                self._adapter.delete_network(network_id)

        # Remove working directory if configured
        if self.cfg.auto_remove and workdir_host and os.path.isdir(workdir_host):
            shutil.rmtree(workdir_host, ignore_errors=True)
