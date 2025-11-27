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
Types and configuration for the Container backend.

This backend enables local hyperparameter optimization using Docker or Podman
containers without requiring a Kubernetes cluster. It orchestrates trial execution
by delegating to the TrainerClient's ContainerBackend for individual training runs.

Configuration options:
 - pull_policy: Controls image pulling. Supported values: "IfNotPresent",
   "Always", "Never". The default is "IfNotPresent".
 - auto_remove: Whether to remove containers and networks when jobs are deleted.
   Defaults to True.
 - container_host: Optional override for connecting to a remote/local container
   daemon. By default, auto-detects from environment or uses system defaults.
 - container_runtime: Force use of a specific container runtime ("docker" or "podman").
   If not set, auto-detects based on availability (tries Docker first, then Podman).
 - storage_path: Local directory to store optimization state and results.
   Defaults to "~/.kubeflow/optimizer". The path will be expanded to handle ~.
 - max_parallel_trials: Maximum number of trials to run in parallel.
   Defaults to 1 (sequential execution). Set higher for faster optimization
   if system resources permit.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ContainerBackendConfig(BaseModel):
    """Configuration for container-based optimizer backend.

    This configuration enables local hyperparameter optimization using Docker or
    Podman containers. The backend orchestrates trials by using the TrainerClient's
    ContainerBackend to execute individual training runs.

    Examples:
        Basic usage with defaults:
        ```python
        from kubeflow.optimizer import OptimizerClient, ContainerBackendConfig

        client = OptimizerClient(backend_config=ContainerBackendConfig())
        ```

        Custom configuration:
        ```python
        config = ContainerBackendConfig(
            pull_policy="Always",
            container_runtime="podman",
            storage_path="/tmp/optimizer-jobs",
            max_parallel_trials=3,
        )
        client = OptimizerClient(backend_config=config)
        ```

    Attributes:
        pull_policy: When to pull container images. Options: "IfNotPresent" (pull
            only if image doesn't exist locally), "Always" (always pull latest),
            "Never" (never pull, use local only). Defaults to "IfNotPresent".
        auto_remove: Whether to automatically remove containers and networks when
            optimization jobs are deleted. Defaults to True.
        container_host: Optional custom container daemon connection URL. If not
            provided, uses environment variables or system defaults. Example:
            "unix:///var/run/docker.sock" for Docker or "unix:///run/podman/podman.sock"
            for Podman.
        container_runtime: Force specific container runtime. Options: "docker" or
            "podman". If not set, automatically detects available runtime (tries
            Docker first, then Podman).
        storage_path: Directory path for storing optimization job state, trial
            results, and metadata. Defaults to "~/.kubeflow/optimizer". The tilde
            (~) is automatically expanded to the user's home directory.
        max_parallel_trials: Maximum number of trials to execute concurrently.
            Defaults to 1 (sequential execution). Higher values can speed up
            optimization but require more system resources (CPU, memory, GPU).
            Consider your system's capacity when setting this value.
    """

    pull_policy: Literal["IfNotPresent", "Always", "Never"] = Field(
        default="IfNotPresent",
        description="When to pull container images: IfNotPresent, Always, or Never",
    )

    auto_remove: bool = Field(
        default=True,
        description="Whether to remove containers and networks when jobs are deleted",
    )

    container_host: Optional[str] = Field(
        default=None,
        description="Optional custom container daemon connection URL",
    )

    container_runtime: Optional[Literal["docker", "podman"]] = Field(
        default=None,
        description="Force specific container runtime (docker or podman)",
    )

    storage_path: str = Field(
        default="~/.kubeflow/optimizer",
        description="Local directory to store optimization state and results",
    )

    max_parallel_trials: int = Field(
        default=1,
        description="Maximum number of trials to run in parallel",
        ge=1,
    )

    @field_validator("pull_policy")
    @classmethod
    def validate_pull_policy(cls, v: str) -> str:
        """Validate pull_policy is one of the allowed values."""
        allowed = {"IfNotPresent", "Always", "Never"}
        if v not in allowed:
            raise ValueError(f"pull_policy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("max_parallel_trials")
    @classmethod
    def validate_max_parallel_trials(cls, v: int) -> int:
        """Validate max_parallel_trials is positive."""
        if v < 1:
            raise ValueError(f"max_parallel_trials must be >= 1, got {v}")
        return v

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, v: str) -> str:
        """Validate storage_path is not empty."""
        if not v or not v.strip():
            raise ValueError("storage_path cannot be empty")
        return v.strip()

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
