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
Utility functions for the Container backend.
"""

import logging
import os
from pathlib import Path

from kubeflow.common.constants import UNKNOWN
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


def create_workdir(job_name: str) -> str:
    """
    Create per-job working directory on host.

    Working directories are created under ~/.kubeflow/trainer/containers/<job_name>

    Args:
        job_name: Name of the training job.

    Returns:
        Absolute path to the working directory.
    """
    home_base = Path.home() / ".kubeflow" / "trainer" / "containers"
    home_base.mkdir(parents=True, exist_ok=True)
    workdir = str((home_base / f"{job_name}").resolve())
    os.makedirs(workdir, exist_ok=True)
    return workdir


def get_training_script_code(trainer: types.CustomTrainer) -> str:
    """
    Generate the training script code from the trainer function.

    This extracts the function source and appends a function call,
    similar to how the Kubernetes backend handles training scripts.

    Args:
        trainer: CustomTrainer configuration.

    Returns:
        Complete Python code as a string to execute.
    """
    import inspect
    import textwrap

    code = inspect.getsource(trainer.func)
    code = textwrap.dedent(code)
    if trainer.func_args is None:
        code += f"\n{trainer.func.__name__}()\n"
    else:
        code += f"\n{trainer.func.__name__}(**{trainer.func_args})\n"
    return code


def build_environment(trainer: types.CustomTrainer) -> dict[str, str]:
    """
    Build environment variables for containers.

    Args:
        trainer: CustomTrainer configuration.

    Returns:
        Dictionary of environment variables.
    """
    return dict(trainer.env or {})


def build_pip_install_cmd(trainer: types.CustomTrainer) -> str:
    """
    Build pip install command for packages.

    Args:
        trainer: CustomTrainer configuration.

    Returns:
        Pip install command string (empty if no packages to install).
    """
    pkgs = trainer.packages_to_install or []
    if not pkgs:
        return ""

    index_urls = trainer.pip_index_urls or list(constants.DEFAULT_PIP_INDEX_URLS)
    main_idx = index_urls[0]
    extras = " ".join(f"--extra-index-url {u}" for u in index_urls[1:])
    quoted = " ".join(f'"{p}"' for p in pkgs)
    return (
        "PIP_DISABLE_PIP_VERSION_CHECK=1 pip install --no-warn-script-location "
        f"--index-url {main_idx} {extras} {quoted} && "
    )


def container_status_to_trainjob_status(status: str, exit_code: int) -> str:
    """
    Convert container status to TrainJob status.

    Args:
        status: Container status (e.g., "running", "exited", "created").
        exit_code: Container exit code.

    Returns:
        TrainJob status constant.
    """
    if status == "running":
        return constants.TRAINJOB_RUNNING
    if status == "created":
        return constants.TRAINJOB_CREATED
    if status == "exited":
        # Exit code 0 -> complete, else failed
        return constants.TRAINJOB_COMPLETE if exit_code == 0 else constants.TRAINJOB_FAILED
    return UNKNOWN


def aggregate_status_from_containers(container_statuses: list[str]) -> str:
    """
    Aggregate status from multiple container statuses.

    Args:
        container_statuses: List of container status strings.

    Returns:
        Aggregated TrainJob status.
    """
    if constants.TRAINJOB_FAILED in container_statuses:
        return constants.TRAINJOB_FAILED
    if constants.TRAINJOB_RUNNING in container_statuses:
        return constants.TRAINJOB_RUNNING
    if all(s == constants.TRAINJOB_COMPLETE for s in container_statuses if s != UNKNOWN):
        return constants.TRAINJOB_COMPLETE
    if any(s == constants.TRAINJOB_CREATED for s in container_statuses):
        return constants.TRAINJOB_CREATED
    return UNKNOWN


def maybe_pull_image(adapter, image: str, pull_policy: str):
    """
    Pull image based on pull policy.

    Args:
        adapter: Container client adapter (DockerClientAdapter or PodmanClientAdapter).
        image: Container image name.
        pull_policy: Pull policy ("IfNotPresent", "Always", or "Never").

    Raises:
        RuntimeError: If image is not found or pull fails.
    """
    policy = pull_policy.lower()
    try:
        if policy == "never":
            if not adapter.image_exists(image):
                raise RuntimeError(f"Image '{image}' not found locally and pull policy is Never")
            return
        if policy == "always":
            logger.debug(f"Pulling image (Always): {image}")
            adapter.pull_image(image)
            return
        # IfNotPresent
        if not adapter.image_exists(image):
            logger.debug(f"Pulling image (IfNotPresent): {image}")
            adapter.pull_image(image)
    except Exception as e:
        raise RuntimeError(f"Failed to ensure image '{image}': {e}") from e


def get_container_status(adapter, container_id: str) -> str:
    """
    Get the TrainJob status of a container.

    Args:
        adapter: Container client adapter (DockerClientAdapter or PodmanClientAdapter).
        container_id: Container ID.

    Returns:
        TrainJob status constant.
    """
    try:
        status, exit_code = adapter.container_status(container_id)
        return container_status_to_trainjob_status(status, exit_code)
    except Exception:
        return UNKNOWN


def aggregate_container_statuses(adapter, containers: list[dict]) -> str:
    """
    Aggregate TrainJob status from container info dicts.

    Args:
        adapter: Container client adapter (DockerClientAdapter or PodmanClientAdapter).
        containers: List of container info dicts with 'id' key.

    Returns:
        Aggregated TrainJob status.
    """
    statuses = [get_container_status(adapter, c["id"]) for c in containers]
    return aggregate_status_from_containers(statuses)
