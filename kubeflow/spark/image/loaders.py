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

"""Image loaders for loading Docker images into Kubernetes clusters.

Each loader abstracts the cluster-specific CLI command needed to make a locally
built Docker image available to cluster nodes. Choose the loader that matches
your local cluster type:

- KindImageLoader  → Kind clusters  (kind load docker-image)
- MinikubeImageLoader → Minikube     (minikube image load)      [not yet implemented]
- RegistryImageLoader → Any cluster  (docker push to registry)  [not yet implemented]

Example:
    loader = KindImageLoader(cluster_name="spark-cluster")
    loader.load("my-spark-app:v1.0")
"""

import abc
import logging
import subprocess

logger = logging.getLogger(__name__)


class ImageLoader(abc.ABC):
    """Abstract base class for loading Docker images into a Kubernetes cluster.

    Implementations wrap the cluster-specific CLI tool (kind, k3d, minikube, etc.)
    so that the rest of the SDK stays decoupled from the cluster type.
    """

    @abc.abstractmethod
    def load(self, image: str) -> None:
        """Load a Docker image into the target cluster.

        The image must already exist in the local Docker daemon before calling
        this method.

        Args:
            image: Image name and tag (e.g., "my-spark-app:latest").

        Raises:
            RuntimeError: If the load command fails or the CLI tool is not found.
        """
        raise NotImplementedError()


class KindImageLoader(ImageLoader):
    """Load a Docker image into a Kind cluster via `kind load docker-image`.

    Args:
        cluster_name: Name of the Kind cluster. If None, Kind uses the current
            kubectl context cluster (typically named "kind").
    """

    def __init__(self, cluster_name: str | None = None):
        """Initialize KindImageLoader.

        Args:
            cluster_name: Name of the Kind cluster, or None for the default cluster.
        """
        self.cluster_name = cluster_name

    def load(self, image: str) -> None:
        """Load image into the Kind cluster.

        Args:
            image: Image name and tag (e.g., "my-spark-app:latest").

        Raises:
            RuntimeError: If `kind` is not installed, or if the load command fails.
        """
        cmd = ["kind", "load", "docker-image", image]
        if self.cluster_name:
            cmd += ["--name", self.cluster_name]

        logger.info("Loading image '%s' into Kind cluster (cmd: %s)", image, " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.info("kind load output: %s", result.stdout.strip())
        except FileNotFoundError as e:
            raise RuntimeError(
                "The 'kind' CLI was not found. "
                "Install it from: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
            ) from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip() if e.stderr else "(no stderr)"
            raise RuntimeError(
                f"Failed to load image '{image}' into Kind cluster: {stderr}"
            ) from e

class MinikubeImageLoader(ImageLoader):
    """Load a Docker image into a Minikube cluster via `minikube image load`.

    Args:
        profile: Minikube profile name. If None, uses the default profile.
    """

    def __init__(self, profile: str | None = None):
        self.profile = profile

    def load(self, image: str) -> None:
        raise NotImplementedError(
            "MinikubeImageLoader is not yet implemented. "
            "To load manually: minikube image load "
            + image
            + (" -p " + self.profile if self.profile else "")
        )


class RegistryImageLoader(ImageLoader):
    """Push a Docker image to a container registry so the cluster can pull it.
    Args:
        registry: Registry host (e.g., "gcr.io/my-project").

    """

    def __init__(self, registry: str):
        self.registry = registry

    def load(self, image: str) -> None:
        raise NotImplementedError(
            "RegistryImageLoader is not yet implemented. "
            f"To push manually: docker tag {image} {self.registry}/{image} "
            f"&& docker push {self.registry}/{image}"
        )
