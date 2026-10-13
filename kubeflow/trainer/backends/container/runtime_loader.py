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
Runtime loader for container backends (Docker, Podman).

Training runtime definitions are loaded from the sources configured in
`TrainingRuntimeSource.sources`, in priority order. Each source is a URL with a
scheme: `github://owner/repo[/path]`, `https://`, `http://`, `file://`, or an
absolute path. The first runtime found for a given name wins.

Any runtime name not provided by a configured source falls back to a built-in
definition derived from `constants.DEFAULT_FRAMEWORK_IMAGES`.

GitHub sources are fetched over the network and memoized for the lifetime of the
process, so resolving runtimes repeatedly (for example, once per job in
`list_jobs`) does not re-fetch them.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import yaml

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types as base_types

logger = logging.getLogger(__name__)

__all__ = [
    "get_training_runtime_from_sources",
    "list_training_runtimes_from_sources",
]


def _load_runtime_from_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def _discover_github_runtime_files(
    owner: str = "kubeflow",
    repo: str = "trainer",
    branch: str = "master",
    path: str = "manifests/base/runtimes",
) -> list[str]:
    """
    Discover available runtime YAML files from GitHub repository.

    Fetches the directory listing from GitHub and extracts .yaml filenames,
    excluding kustomization.yaml and other non-runtime files.

    Args:
        owner: GitHub repository owner (default: "kubeflow")
        repo: GitHub repository name (default: "trainer")
        branch: Git branch name (default: "master")
        path: Path to runtimes directory (default: "manifests/base/runtimes")

    Returns:
        List of YAML filenames (e.g., ['torch_distributed.yaml', ...])
        Returns empty list if discovery fails.
    """
    tree_url = f"https://github.com/{owner}/{repo}/tree/{branch}/{path}"
    try:
        logger.debug(f"Discovering runtimes from GitHub: {tree_url}")
        with urllib.request.urlopen(tree_url, timeout=5) as response:
            html_content = response.read().decode("utf-8")

        # Parse HTML to find .yaml files
        # Look for .yaml filenames in the HTML content
        import re

        # Pattern to match .yaml files in the HTML
        # Matches word characters, hyphens, underscores followed by .yaml
        pattern = r"([\w-]+\.yaml)"
        matches = re.findall(pattern, html_content)

        # Filter out kustomization.yaml, config files, and duplicates
        # Keep only runtime files (typically named *_distributed.yaml or similar)
        runtime_files = []
        seen = set()
        exclude_files = {"kustomization.yaml", "golangci.yaml", "pre-commit-config.yaml"}

        for match in matches:
            filename = match
            if filename not in seen and filename not in exclude_files:
                runtime_files.append(filename)
                seen.add(filename)

        logger.debug(f"Discovered {len(runtime_files)} runtime files: {runtime_files}")
        return runtime_files

    except Exception as e:
        logger.debug(f"Failed to discover GitHub runtime files from {tree_url}: {e}")
        return []


def _fetch_runtime_from_github(
    runtime_file: str,
    owner: str = "kubeflow",
    repo: str = "trainer",
    branch: str = "master",
    path: str = "manifests/base/runtimes",
) -> dict[str, Any] | None:
    """
    Fetch a runtime YAML from GitHub.

    Args:
        runtime_file: YAML filename to fetch
        owner: GitHub repository owner (default: "kubeflow")
        repo: GitHub repository name (default: "trainer")
        branch: Git branch name (default: "master")
        path: Path to runtimes directory (default: "manifests/base/runtimes")

    Returns None if fetch fails (network error, timeout, etc.)
    """
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{runtime_file}"
    try:
        logger.debug(f"Fetching runtime from GitHub: {url}")
        with urllib.request.urlopen(url, timeout=5) as response:
            content = response.read().decode("utf-8")
            data = yaml.safe_load(content)
            logger.debug(f"Successfully fetched {runtime_file} from GitHub")
            return data
    except (urllib.error.URLError, TimeoutError, Exception) as e:
        logger.debug(f"Failed to fetch {runtime_file} from GitHub: {e}")
        return None


def _create_default_runtimes() -> list[base_types.Runtime]:
    """
    Create default Runtime objects from DEFAULT_FRAMEWORK_IMAGES constant.

    Returns:
        List of default Runtime objects for each framework.
    """
    default_runtimes = []

    for framework, image in constants.DEFAULT_FRAMEWORK_IMAGES.items():
        runtime = base_types.Runtime(
            name=f"{framework}-distributed",
            trainer=base_types.RuntimeTrainer(
                trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                framework=framework,
                num_nodes=1,
                image=image,
            ),
            kind=base_types.RuntimeKind.TRAINING_RUNTIME,
        )
        default_runtimes.append(runtime)
        logger.debug(f"Created default runtime: {runtime.name} with image {image}")

    return default_runtimes


def _parse_runtime_yaml(data: dict[str, Any], source: str = "unknown") -> base_types.Runtime:
    """
    Parse a runtime YAML dict into a Runtime object.

    Args:
        data: The YAML data as a dictionary
        source: Source of the YAML (for error messages)

    Returns:
        Runtime object

    Raises:
        ValueError: If the YAML is malformed or missing required fields
    """
    # Require CRD-like schema strictly. Accept both ClusterTrainingRuntime
    # and TrainingRuntime kinds.
    if not (
        data.get("kind") in {"ClusterTrainingRuntime", "TrainingRuntime"} and data.get("metadata")
    ):
        raise ValueError(
            f"Runtime YAML from {source} must be a ClusterTrainingRuntime CRD-shaped document"
        )

    name = data["metadata"].get("name")
    if not name:
        raise ValueError(f"Runtime YAML from {source} missing metadata.name")

    labels = data["metadata"].get("labels", {})
    framework = labels.get("trainer.kubeflow.org/framework")
    if not framework:
        raise ValueError(
            f"Runtime {name} from {source} must set "
            f"metadata.labels['trainer.kubeflow.org/framework']"
        )

    spec = data.get("spec", {})
    ml_policy = spec.get("mlPolicy", {})
    num_nodes = int(ml_policy.get("numNodes", 1))

    # Validate presence of a 'node' replicated job with a container image
    templ = spec.get("template", {}).get("spec", {})
    replicated = templ.get("replicatedJobs", [])
    node_jobs = [j for j in replicated if j.get("name") == "node"]
    if not node_jobs:
        raise ValueError(
            f"Runtime {name} from {source} must define replicatedJobs with a 'node' entry"
        )
    node_spec = node_jobs[0].get("template", {}).get("spec", {}).get("template", {}).get("spec", {})
    containers = node_spec.get("containers", [])
    if not containers:
        raise ValueError(f"Runtime {name} from {source} 'node' must specify at least one container")

    # Extract the container image from the container named 'node', or fallback to first container
    image = None
    for container in containers:
        if container.get("name") == "node" and container.get("image"):
            image = container.get("image")
            break

    # Fallback to first container with an image if no 'node' container found
    if not image:
        for container in containers:
            if container.get("image"):
                image = container.get("image")
                break

    if not image:
        raise ValueError(
            f"Runtime {name} from {source} 'node' must specify an image in at least one container"
        )

    return base_types.Runtime(
        name=name,
        trainer=base_types.RuntimeTrainer(
            trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
            framework=framework,
            num_nodes=num_nodes,
            image=image,
        ),
        kind=base_types.RuntimeKind.TRAINING_RUNTIME,
    )


def _parse_source_url(source: str) -> tuple[str, str]:
    """
    Parse a source URL to determine its type and path.

    Args:
        source: Source URL with scheme (github://, https://, file://, or absolute path)

    Returns:
        Tuple of (source_type, path) where source_type is one of:
        'github', 'http', 'https', 'file'
    """
    if source.startswith("github://"):
        return ("github", source[9:])  # Remove 'github://'
    elif source.startswith("https://"):
        return ("https", source)
    elif source.startswith("http://"):
        return ("http", source)
    elif source.startswith("file://"):
        return ("file", source[7:])  # Remove 'file://'
    elif source.startswith("/"):
        # Absolute path without file:// prefix
        return ("file", source)
    else:
        raise ValueError(
            f"Unsupported source URL scheme: {source}. "
            f"Supported: github://, https://, http://, file://, or absolute paths"
        )


@functools.cache
def _load_from_github_url(github_path: str) -> tuple[base_types.Runtime, ...]:
    """
    Load runtimes from a GitHub URL (github://owner/repo[/path]).

    Memoized for the lifetime of the process: runtimes are resolved once per job in
    `list_jobs`, and re-fetching them per job costs a directory listing plus one
    request per runtime file against unauthenticated GitHub. Restart the process to
    pick up upstream runtime changes.

    Args:
        github_path: Path after github:// (e.g., "kubeflow/trainer" or "myorg/myrepo")

    Returns:
        Tuple of Runtime objects loaded from GitHub
    """
    runtimes = []
    runtime_names_seen = set()

    # Parse the GitHub path
    # Format: owner/repo[/path/to/runtimes]
    parts = github_path.split("/")
    if len(parts) < 2:
        logger.warning(f"Invalid GitHub path format: {github_path}. Expected owner/repo[/path]")
        return ()

    owner = parts[0]
    repo = parts[1]
    # Custom path if provided (default to manifests/base/runtimes)
    custom_path = "/".join(parts[2:]) if len(parts) > 2 else "manifests/base/runtimes"

    # Discover runtime files from the specified GitHub repo
    logger.debug(f"Loading runtimes from GitHub: {owner}/{repo}/{custom_path}")
    github_runtime_files = _discover_github_runtime_files(owner=owner, repo=repo, path=custom_path)

    for runtime_file in github_runtime_files:
        try:
            data = _fetch_runtime_from_github(
                runtime_file, owner=owner, repo=repo, path=custom_path
            )
            if data is not None:
                runtime = _parse_runtime_yaml(data, source=f"github://{github_path}/{runtime_file}")
                if runtime.name not in runtime_names_seen:
                    runtimes.append(runtime)
                    runtime_names_seen.add(runtime.name)
                    logger.debug(f"Loaded runtime from GitHub: {runtime.name}")
        except Exception as e:
            logger.debug(f"Failed to parse GitHub runtime {runtime_file}: {e}")

    return tuple(runtimes)


def _load_from_http_url(url: str) -> list[base_types.Runtime]:
    """
    Load runtimes from an HTTP(S) URL.

    Args:
        url: HTTP(S) URL to a runtime YAML file or directory listing

    Returns:
        List of Runtime objects loaded from HTTP(S)
    """
    runtimes = []

    try:
        import urllib.request

        logger.debug(f"Fetching runtime from HTTP: {url}")
        with urllib.request.urlopen(url, timeout=5) as response:
            content = response.read().decode("utf-8")
            import yaml

            data = yaml.safe_load(content)
            runtime = _parse_runtime_yaml(data, source=url)
            runtimes.append(runtime)
            logger.debug(f"Loaded runtime from HTTP: {runtime.name}")
    except Exception as e:
        logger.debug(f"Failed to load runtime from HTTP {url}: {e}")

    return runtimes


def _load_from_filesystem(path: str) -> list[base_types.Runtime]:
    """
    Load runtimes from local filesystem path.

    Args:
        path: Local filesystem path to a directory or YAML file

    Returns:
        List of Runtime objects loaded from filesystem
    """
    from pathlib import Path

    runtimes = []
    runtime_path = Path(path).expanduser()

    try:
        if runtime_path.is_dir():
            # Load all YAML files from directory
            for yaml_file in sorted(runtime_path.glob("*.yaml")):
                try:
                    data = _load_runtime_from_yaml(yaml_file)
                    runtime = _parse_runtime_yaml(data, source=str(yaml_file))
                    runtimes.append(runtime)
                    logger.debug(f"Loaded runtime from file: {runtime.name}")
                except Exception as e:
                    logger.warning(f"Failed to load runtime from {yaml_file}: {e}")
        elif runtime_path.is_file():
            # Load single YAML file
            data = _load_runtime_from_yaml(runtime_path)
            runtime = _parse_runtime_yaml(data, source=str(runtime_path))
            runtimes.append(runtime)
            logger.debug(f"Loaded runtime from file: {runtime.name}")
        else:
            logger.warning(f"Path does not exist: {runtime_path}")
    except Exception as e:
        logger.warning(f"Failed to load runtimes from {path}: {e}")

    return runtimes


def list_training_runtimes_from_sources(sources: list[str]) -> list[base_types.Runtime]:
    """
    List all available training runtimes from configured sources.

    Args:
        sources: List of source URLs with schemes (github://, https://, http://, file://, or paths)

    Returns:
        List of Runtime objects (built-in runtimes used as default if not found in sources)
    """
    runtimes: list[base_types.Runtime] = []
    runtime_names_seen = set()

    # Load from each configured source in priority order
    for source in sources:
        try:
            source_type, source_path = _parse_source_url(source)

            if source_type == "github":
                source_runtimes = _load_from_github_url(source_path)
            elif source_type in ("http", "https"):
                source_runtimes = _load_from_http_url(source)
            elif source_type == "file":
                source_runtimes = _load_from_filesystem(source_path)
            else:
                logger.warning(f"Unsupported source type: {source_type}")
                continue

            # Add runtimes, skipping duplicates
            for runtime in source_runtimes:
                if runtime.name not in runtime_names_seen:
                    runtimes.append(runtime)
                    runtime_names_seen.add(runtime.name)
        except Exception as e:
            logger.debug(f"Failed to load from source {source}: {e}")

    # Fallback to default runtimes from constants if not found in sources
    for default_runtime in _create_default_runtimes():
        if default_runtime.name not in runtime_names_seen:
            runtimes.append(default_runtime)
            runtime_names_seen.add(default_runtime.name)

    return runtimes


def get_training_runtime_from_sources(name: str, sources: list[str]) -> base_types.Runtime:
    """
    Get a specific training runtime by name from configured sources.

    Args:
        name: The name of the runtime to get
        sources: List of source URLs with schemes

    Returns:
        Runtime object

    Raises:
        ValueError: If the runtime is not found
    """
    for rt in list_training_runtimes_from_sources(sources):
        if rt.name == name:
            return rt
    raise ValueError(
        f"Runtime '{name}' not found. Available runtimes: "
        f"{[rt.name for rt in list_training_runtimes_from_sources(sources)]}"
    )
