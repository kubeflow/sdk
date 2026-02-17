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
Types and configuration for the unified Container backend.

This backend automatically detects and uses either Docker or Podman.
It provides a single interface for container-based execution regardless
of the underlying runtime.

Configuration options:
 - pull_policy: Controls image pulling. Supported values: "IfNotPresent",
   "Always", "Never". The default is "IfNotPresent".
 - auto_remove: Whether to remove containers and networks when jobs are deleted.
   Defaults to True.
 - container_host: Optional override for connecting to a remote/local container
   daemon. By default, auto-detects from environment or uses system defaults.
   For Docker: uses DOCKER_HOST or default socket.
   For Podman: uses CONTAINER_HOST or default socket.
 - container_runtime: Force use of a specific container runtime ("docker" or "podman").
   If not set, auto-detects based on availability (tries Docker first, then Podman).
 - runtime_source: Configuration for training runtime sources using URL schemes.
   Supports github://, https://, http://, file://, and absolute paths.
   Built-in runtimes packaged with kubeflow-trainer are used as default fallback.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TrainingRuntimeSource(BaseModel):
    """Configuration for training runtime sources using URL schemes."""

    sources: list[str] = Field(
        default_factory=lambda: ["github://kubeflow/trainer"],
        description=(
            "Runtime sources with URL schemes (checked in priority order):\n"
            "  - github://owner/repo[/path] - GitHub repository\n"
            "  - https://url or http://url - HTTP(S) endpoint\n"
            "  - file:///path or /absolute/path - Local filesystem\n"
            "If a runtime is not found in configured sources, built-in runtimes "
            "packaged with kubeflow-trainer are used as default."
        ),
    )


class ContainerBackendConfig(BaseModel):
    pull_policy: str = Field(default="IfNotPresent")
    auto_remove: bool = Field(default=True)
    container_host: str | None = Field(default=None)
    container_runtime: Literal["docker", "podman"] | None = Field(default=None)
    runtime_source: TrainingRuntimeSource = Field(
        default_factory=TrainingRuntimeSource,
        description="Configuration for training runtime sources",
    )
    dataset_initializer_image: str = Field(
        default="ghcr.io/kubeflow/trainer/dataset-initializer:latest",
        description="Container image for dataset initializers",
    )
    model_initializer_image: str = Field(
        default="ghcr.io/kubeflow/trainer/model-initializer:latest",
        description="Container image for model initializers",
    )
    initializer_timeout: int = Field(
        default=600,
        description="Timeout in seconds for initializer containers (default 10 minutes)",
    )
