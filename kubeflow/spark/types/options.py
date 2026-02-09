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

"""Options for advanced Spark configuration (KEP-107 lines 180-192).

The options pattern provides extensibility for advanced Kubernetes configurations
without polluting the main API. Future option types can be added without breaking changes.

This follows the same callable pattern as kubeflow.trainer.options for SDK consistency.
"""

from dataclasses import dataclass
from typing import Any

from kubeflow.spark.backends.base import RuntimeBackend


@dataclass
class Labels:
    """Add Kubernetes labels to Spark resources (.metadata.labels).

    Labels are key-value pairs attached to Kubernetes resources for organization,
    selection, and grouping.

    Supported backends:
        - Kubernetes

    Args:
        labels: Dictionary of label key-value pairs.

    Example:
        options = [Labels({"app": "spark", "team": "data-eng"})]
        spark = client.connect(..., options=options)
    """

    labels: dict[str, str]

    def __call__(self, crd: dict[str, Any], backend: RuntimeBackend) -> None:
        """Apply labels to the CRD specification.

        Args:
            crd: CRD specification dictionary to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support labels.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Labels option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        metadata = crd.setdefault("metadata", {})
        labels = metadata.setdefault("labels", {})
        labels.update(self.labels)


@dataclass
class Annotations:
    """Add Kubernetes annotations to Spark resources (.metadata.annotations).

    Annotations store non-identifying metadata that can be used by tools,
    libraries, or for documentation purposes.

    Supported backends:
        - Kubernetes

    Args:
        annotations: Dictionary of annotation key-value pairs.

    Example:
        options = [
            Annotations({
                "description": "Daily ETL pipeline",
                "owner": "data-team@company.com"
            })
        ]
        spark = client.connect(..., options=options)
    """

    annotations: dict[str, str]

    def __call__(self, crd: dict[str, Any], backend: RuntimeBackend) -> None:
        """Apply annotations to the CRD specification.

        Args:
            crd: CRD specification dictionary to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support annotations.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Annotations option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        metadata = crd.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        annotations.update(self.annotations)


@dataclass
class PodTemplateOverride:
    """Override pod template specifications for driver or executors.

    Provides full control over Kubernetes pod specifications for advanced use cases
    like custom volumes, init containers, sidecars, or security contexts.

    Supported backends:
        - Kubernetes

    Args:
        role: Target role ("driver" or "executor").
        template: Pod template specification dict.

    Example:
        options = [
            PodTemplateOverride(
                role="executor",
                template={
                    "spec": {
                        "securityContext": {
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        }
                    }
                }
            )
        ]
        spark = client.connect(..., options=options)

    Warning:
        Pod template overrides can conflict with SDK-managed settings.
        Use with caution and test thoroughly.
    """

    role: str  # "driver" or "executor"
    template: dict[str, Any]

    def __call__(self, crd: dict[str, Any], backend: RuntimeBackend) -> None:
        """Apply pod template override to the CRD specification.

        Args:
            crd: CRD specification dictionary to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support pod template overrides or invalid role.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"PodTemplateOverride option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        if self.role == "driver":
            spec_key = "server"
        elif self.role == "executor":
            spec_key = "executor"
        else:
            raise ValueError(f"Invalid role '{self.role}'. Must be 'driver' or 'executor'.")

        spec = crd.setdefault("spec", {})
        role_spec = spec.setdefault(spec_key, {})
        template = role_spec.setdefault("template", {})

        # Deep merge template
        self._deep_merge(template, self.template)

    @staticmethod
    def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                PodTemplateOverride._deep_merge(target[key], value)
            else:
                target[key] = value


@dataclass
class NodeSelector:
    """Add node selector constraints to Spark pods.

    Node selectors constrain pod scheduling to nodes with matching labels.
    Applied to both driver and executor pods.

    Supported backends:
        - Kubernetes

    Args:
        selectors: Dictionary of node label key-value pairs.

    Example:
        options = [
            NodeSelector({"node-type": "spark", "gpu": "true"})
        ]
        spark = client.connect(..., options=options)
    """

    selectors: dict[str, str]

    def __call__(self, crd: dict[str, Any], backend: RuntimeBackend) -> None:
        """Apply node selector constraints to the CRD specification.

        Args:
            crd: CRD specification dictionary to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support node selectors.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"NodeSelector option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        spec = crd.setdefault("spec", {})

        for role in ["server", "executor"]:
            role_spec = spec.setdefault(role, {})
            template = role_spec.setdefault("template", {})
            pod_spec = template.setdefault("spec", {})
            node_selector = pod_spec.setdefault("nodeSelector", {})
            node_selector.update(self.selectors)


@dataclass
class Toleration:
    """Add toleration to Spark pods for node taints.

    Tolerations allow pods to schedule onto nodes with matching taints.
    Applied to both driver and executor pods.

    Supported backends:
        - Kubernetes

    Args:
        key: Taint key to tolerate.
        operator: Operator (Equal or Exists).
        value: Taint value (if operator is Equal).
        effect: Taint effect (NoSchedule, PreferNoSchedule, or NoExecute).

    Example:
        options = [
            Toleration(
                key="spark-workload",
                operator="Equal",
                value="true",
                effect="NoSchedule"
            )
        ]
        spark = client.connect(..., options=options)
    """

    key: str
    operator: str = "Equal"
    value: str = ""
    effect: str = "NoSchedule"

    def __call__(self, crd: dict[str, Any], backend: RuntimeBackend) -> None:
        """Apply toleration to the CRD specification.

        Args:
            crd: CRD specification dictionary to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support tolerations.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Toleration option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        toleration = {
            "key": self.key,
            "operator": self.operator,
            "effect": self.effect,
        }
        if self.value:
            toleration["value"] = self.value

        spec = crd.setdefault("spec", {})

        for role in ["server", "executor"]:
            role_spec = spec.setdefault(role, {})
            template = role_spec.setdefault("template", {})
            pod_spec = template.setdefault("spec", {})
            tolerations = pod_spec.setdefault("tolerations", [])
            tolerations.append(toleration)


@dataclass
class Name:
    """Set a custom name for the SparkConnect session.

    This option sets the session name which becomes the Kubernetes resource name.
    If not provided, a name will be auto-generated with format: spark-connect-{uuid}

    The session name must follow DNS-1123 subdomain rules:
    - Lowercase alphanumeric characters, '-', or '.'
    - Start and end with alphanumeric character
    - Maximum 253 characters

    Supported backends:
        - Kubernetes

    Args:
        name: Custom name for the session. Must be a valid Kubernetes resource name.

    Example:
        ```python
        from kubeflow.spark import SparkClient
        from kubeflow.spark.types.options import Name

        client = SparkClient()

        # With explicit name
        spark = client.connect(options=[Name("my-custom-session")])

        # Auto-generated name
        spark = client.connect()  # Creates "spark-connect-a1b2c3d4"
        ```

    Note:
        This option is extracted early in the backend flow before CRD building,
        unlike other options which modify the CRD after it's built.
    """

    name: str

    def __call__(self, crd: dict[str, Any], backend: RuntimeBackend) -> None:
        """Apply custom name to CRD metadata.

        Note: This method exists for interface consistency but is not typically
        called, as the name is extracted earlier in the backend flow.

        Args:
            crd: CRD specification dictionary to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support custom names.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Name option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        metadata = crd.setdefault("metadata", {})
        metadata["name"] = self.name
