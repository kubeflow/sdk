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
import math
import re
from typing import Any

from kubeflow_spark_api import models

from kubeflow.spark.backends.base import RuntimeBackend
from kubeflow.spark.backends.kubernetes.constants import (
    DEFAULT_DRIVER_CPU,
    DEFAULT_DRIVER_MEMORY,
    DEFAULT_EXECUTOR_CPU,
    DEFAULT_EXECUTOR_MEMORY,
    DEFAULT_NUM_EXECUTORS,
)
from kubeflow.spark.types.types import Driver as BaseDriver, Executor as BaseExecutor

SparkResource = models.SparkV1alpha1SparkConnect | models.SparkV1beta2SparkApplication


# NOTE:
# This helper intentionally mirrors the Kubernetes backend implementation.
# It is duplicated here to avoid a circular import between the options module
# and backend utilities.
def _convert_kubernetes_memory_to_spark(memory: str) -> str:
    """Convert Kubernetes-style memory values to Spark-compatible memory values.

    Spark accepts integer memory values with JVM suffixes (k, m, g, t, p).
    Kubernetes quantities may contain fractional values (for example, ``1.5Gi``),
    so fractional values are converted to an equivalent MiB value.

    Args:
        memory: Memory value using Kubernetes or Spark notation.

    Returns:
        Memory value formatted using Spark-compatible units. If the input format
        is not recognized, the original value is returned.
    """
    if not memory or not memory[-1].isalpha():
        return memory

    match = re.match(
        r"^(\d+(?:\.\d+)?)\s*([KMGTPE]i?|[kmgtp]b?)$",
        memory,
        re.IGNORECASE,
    )
    if not match:
        return memory

    coefficient, suffix = match.group(1), (match.group(2) or "").lower()

    exponent_by_suffix = {
        "ki": 10,
        "k": 10,
        "kb": 10,
        "mi": 20,
        "m": 20,
        "mb": 20,
        "gi": 30,
        "g": 30,
        "gb": 30,
        "ti": 40,
        "t": 40,
        "tb": 40,
        "pi": 50,
        "p": 50,
        "pb": 50,
        "ei": 60,
    }

    if suffix not in exponent_by_suffix:
        return memory

    exponent = exponent_by_suffix[suffix]

    spark_suffix = {10: "k", 20: "m", 30: "g", 40: "t", 50: "p"}.get(exponent)
    if "." not in coefficient and spark_suffix is not None:
        return coefficient + spark_suffix

    total_bytes = math.ceil(float(coefficient) * (2**exponent))
    return f"{math.ceil(total_bytes / (2**20))}m"


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

    def __call__(self, resource: SparkResource, backend: RuntimeBackend) -> None:
        """Apply labels to the Spark resource.

        Args:
            resource: Spark resource to modify.
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

        if resource.metadata.labels is None:
            resource.metadata.labels = {}
        resource.metadata.labels.update(self.labels)


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

    def __call__(self, resource: SparkResource, backend: RuntimeBackend) -> None:
        """Apply annotations to the Spark resource.

        Args:
            resource: Spark resource to modify.
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

        if resource.metadata.annotations is None:
            resource.metadata.annotations = {}
        resource.metadata.annotations.update(self.annotations)


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
                            "fsGroup": 1000,
                        }
                    }
                },
            )
        ]
        spark = client.connect(..., options=options)

    Warning:
        Pod template overrides can conflict with SDK-managed settings.
        Use with caution and test thoroughly.
    """

    role: str  # "driver" or "executor"
    template: dict[str, Any]

    def __call__(
        self,
        resource: SparkResource,
        backend: RuntimeBackend,
    ) -> None:
        """Apply pod template override to the SparkConnect model.

        Args:
            resource: Spark resource to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support pod template overrides,
                the resource is not SparkConnect, or the role is invalid.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"PodTemplateOverride option is not compatible with "
                f"{type(backend).__name__}. Supported backends: KubernetesBackend"
            )

        if not isinstance(resource, models.SparkV1alpha1SparkConnect):
            raise ValueError("PodTemplateOverride is currently supported only for SparkConnect.")

        if self.role == "driver":
            role_spec = resource.spec.server
        elif self.role == "executor":
            role_spec = resource.spec.executor
        else:
            raise ValueError(f"Invalid role '{self.role}'. Must be 'driver' or 'executor'.")

        # Get or create template
        if role_spec.template is None:
            role_spec.template = models.IoK8sApiCoreV1PodTemplateSpec()

        # Convert existing template to dict, merge, and convert back
        existing_dict = role_spec.template.to_dict() if role_spec.template else {}
        self._deep_merge(existing_dict, self.template)

        # Ensure spec.containers exists (required by PodSpec validation)
        if (
            "spec" in existing_dict
            and existing_dict["spec"] is not None
            and (
                "containers" not in existing_dict["spec"]
                or existing_dict["spec"]["containers"] is None
            )
        ):
            existing_dict["spec"]["containers"] = []

        role_spec.template = models.IoK8sApiCoreV1PodTemplateSpec.from_dict(existing_dict)

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

    def __call__(self, resource: SparkResource, backend: RuntimeBackend) -> None:
        """Apply node selector constraints to the Spark resource.

        Args:
            resource: Spark resource to modify.
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

        # Apply to both server and executor
        if isinstance(resource, models.SparkV1alpha1SparkConnect):
            role_specs = [
                resource.spec.server,
                resource.spec.executor,
            ]

            for role_spec in role_specs:
                if role_spec.template is None:
                    role_spec.template = models.IoK8sApiCoreV1PodTemplateSpec()

                if role_spec.template.spec is None:
                    role_spec.template.spec = models.IoK8sApiCoreV1PodSpec(containers=[])

                if role_spec.template.spec.node_selector is None:
                    role_spec.template.spec.node_selector = {}

                role_spec.template.spec.node_selector.update(self.selectors)

        else:
            role_specs = [
                resource.spec.driver,
                resource.spec.executor,
            ]

            for role_spec in role_specs:
                if role_spec.node_selector is None:
                    role_spec.node_selector = {}

                role_spec.node_selector.update(self.selectors)


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

    def __call__(self, resource: SparkResource, backend: RuntimeBackend) -> None:
        """Apply toleration to the Spark resource.

        Args:
            resource: Spark resource to modify.
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

        # Create toleration model
        toleration = models.IoK8sApiCoreV1Toleration(
            key=self.key,
            operator=self.operator,
            effect=self.effect,
            value=self.value if self.value else None,
        )

        # Apply to both server and executor
        if isinstance(resource, models.SparkV1alpha1SparkConnect):
            role_specs = [
                resource.spec.server,
                resource.spec.executor,
            ]

            for role_spec in role_specs:
                if role_spec.template is None:
                    role_spec.template = models.IoK8sApiCoreV1PodTemplateSpec()

                if role_spec.template.spec is None:
                    role_spec.template.spec = models.IoK8sApiCoreV1PodSpec(containers=[])

                if role_spec.template.spec.tolerations is None:
                    role_spec.template.spec.tolerations = []

                role_spec.template.spec.tolerations.append(toleration)

        else:
            role_specs = [
                resource.spec.driver,
                resource.spec.executor,
            ]

            for role_spec in role_specs:
                if role_spec.tolerations is None:
                    role_spec.tolerations = []

                role_spec.tolerations.append(toleration)


@dataclass
class Name:
    """Set a custom name for the Spark resource.

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

    def __call__(self, resource: SparkResource, backend: RuntimeBackend) -> None:
        """Apply custom name to the Spark resource metadata.

        Note: This method exists for interface consistency but is not typically
        called, as the name is extracted earlier in the backend flow.

        Args:
            resource: Spark resource to modify.
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

        resource.metadata.name = self.name


@dataclass
class DriverOption(BaseDriver):
    """Configure the SparkApplication driver.

    This option customizes the driver configuration for Spark batch jobs.

    Supported backends:
        - Kubernetes

    Args:
        image: Custom container image for the driver.
        resources: Resource requirements as a dictionary.
        java_options: JVM options for the driver.
        service_account: Kubernetes service account for the driver.
    """

    def __call__(
        self,
        resource: SparkResource,
        backend: RuntimeBackend,
    ) -> None:
        """Apply driver configuration to a SparkApplication.

        Args:
            resource: Spark resource to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support driver configuration.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Driver option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        if not isinstance(resource, models.SparkV1beta2SparkApplication):
            return

        resources = self.resources or {}

        cores = int(resources.get("cpu", DEFAULT_DRIVER_CPU))
        memory = _convert_kubernetes_memory_to_spark(resources.get("memory", DEFAULT_DRIVER_MEMORY))

        resource.spec.driver.cores = cores
        resource.spec.driver.memory = memory

        if self.service_account is not None:
            resource.spec.driver.service_account = self.service_account

        if self.java_options is not None:
            resource.spec.driver.java_options = self.java_options

        if self.image is not None:
            resource.spec.image = self.image


@dataclass
class ExecutorOption(BaseExecutor):
    """Configure the SparkApplication executors.

    This option customizes the executor configuration for Spark batch jobs.

    Supported backends:
        - Kubernetes

    Args:
        num_instances: Number of executor instances.
        resources_per_executor: Resource requirements for each executor.
        java_options: JVM options for executors.
    """

    def __call__(
        self,
        resource: SparkResource,
        backend: RuntimeBackend,
    ) -> None:
        """Apply executor configuration to a SparkApplication.

        Args:
            resource: Spark resource to modify.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support executor configuration.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Executor option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        if not isinstance(resource, models.SparkV1beta2SparkApplication):
            return

        resources = self.resources_per_executor or {}

        instances = self.num_instances if self.num_instances is not None else DEFAULT_NUM_EXECUTORS
        cores = int(resources.get("cpu", DEFAULT_EXECUTOR_CPU))
        memory = _convert_kubernetes_memory_to_spark(
            resources.get("memory", DEFAULT_EXECUTOR_MEMORY)
        )

        resource.spec.executor.instances = instances
        resource.spec.executor.cores = cores
        resource.spec.executor.memory = memory

        if self.java_options is not None:
            resource.spec.executor.java_options = self.java_options
