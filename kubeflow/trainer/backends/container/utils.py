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


def build_initializer_command(initializer: types.BaseInitializer, init_type: str) -> list[str]:
    """
    Build the command for an initializer container.

    Args:
        initializer: Dataset or model initializer configuration.
        init_type: Type of initializer ("dataset" or "model").

    Returns:
        Command list for the initializer container.

    Raises:
        ValueError: If the initializer type is not supported.
    """
    # Use the training-operator initializer script
    # The initializer script is expected to be available in the image
    if isinstance(initializer, (types.S3DatasetInitializer, types.S3ModelInitializer)):
        python_cmd = "python -m kubeflow.storage_initializer.s3 "
    elif isinstance(
        initializer, (types.HuggingFaceDatasetInitializer, types.HuggingFaceModelInitializer)
    ):
        python_cmd = "python -m kubeflow.storage_initializer.hugging_face "
    elif isinstance(initializer, types.DataCacheInitializer):
        python_cmd = "python -m kubeflow.storage_initializer.datacache "
    else:
        raise ValueError(
            f"Unsupported initializer type: {type(initializer).__name__}. "
            "Supported types: HuggingFaceDatasetInitializer, HuggingFaceModelInitializer, "
            "S3DatasetInitializer, S3ModelInitializer, DataCacheInitializer"
        )

    return ["bash", "-c", python_cmd]


def build_initializer_env(initializer: types.BaseInitializer, init_type: str) -> dict[str, str]:
    """
    Build environment variables for an initializer container.

    Args:
        initializer: Dataset or model initializer configuration.
        init_type: Type of initializer ("dataset" or "model").

    Returns:
        Dictionary of environment variables.
    """
    env = {
        "STORAGE_URI": initializer.storage_uri,
    }

    # Set the output path based on initializer type
    if init_type == "dataset":
        env["OUTPUT_PATH"] = constants.DATASET_PATH
    else:  # model
        env["OUTPUT_PATH"] = constants.MODEL_PATH

    # Add optional fields based on initializer type
    if isinstance(
        initializer, (types.HuggingFaceDatasetInitializer, types.HuggingFaceModelInitializer)
    ):
        if initializer.access_token:
            env["ACCESS_TOKEN"] = initializer.access_token
        if hasattr(initializer, "ignore_patterns") and initializer.ignore_patterns:
            env["IGNORE_PATTERNS"] = ",".join(initializer.ignore_patterns)

    elif isinstance(initializer, (types.S3DatasetInitializer, types.S3ModelInitializer)):
        if initializer.endpoint:
            env["ENDPOINT"] = initializer.endpoint
        if initializer.access_key_id:
            env["ACCESS_KEY_ID"] = initializer.access_key_id
        if initializer.secret_access_key:
            env["SECRET_ACCESS_KEY"] = initializer.secret_access_key
        if initializer.region:
            env["REGION"] = initializer.region
        if initializer.role_arn:
            env["ROLE_ARN"] = initializer.role_arn
        if hasattr(initializer, "ignore_patterns") and initializer.ignore_patterns:
            env["IGNORE_PATTERNS"] = ",".join(initializer.ignore_patterns)

    elif isinstance(initializer, types.DataCacheInitializer):
        env["CLUSTER_SIZE"] = str(initializer.num_data_nodes + 1)
        env["METADATA_LOC"] = initializer.metadata_loc
        if initializer.head_cpu:
            env["HEAD_CPU"] = initializer.head_cpu
        if initializer.head_mem:
            env["HEAD_MEM"] = initializer.head_mem
        if initializer.worker_cpu:
            env["WORKER_CPU"] = initializer.worker_cpu
        if initializer.worker_mem:
            env["WORKER_MEM"] = initializer.worker_mem
        if initializer.iam_role:
            env["IAM_ROLE"] = initializer.iam_role

    return env


def get_initializer_image(config) -> str:
    """
    Get the container image for initializers from backend config.

    Args:
        config: ContainerBackendConfig with initializer_image setting.

    Returns:
        Container image name for initializers.
    """
    return config.initializer_image
