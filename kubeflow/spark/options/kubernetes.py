# Copyright The Kubeflow Authors.
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

from kubeflow_spark_api import models

from kubeflow.spark.backends.base import RuntimeBackend

SparkResource = models.SparkV1alpha1SparkConnect | models.SparkV1beta2SparkApplication


@dataclass
class Labels:
    """Add Kubernetes labels to Spark resources (.metadata.labels).

    Labels are key-value pairs attached to Kubernetes resources for organization,
    selection, and grouping.

    Supported backends:
        - Kubernetes

    Args:
        labels: Dictionary of label key-value pairs.

    Example::

        options = [
            Labels(
                {
                    "app": "spark",
                    "team": "data-eng",
                }
            ),
        ]
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

    Example::

        options = [
            Annotations(
                {
                    "description": "Daily ETL pipeline",
                    "owner": "data-team@company.com",
                }
            ),
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
class NodeSelector:
    """Add node selector constraints to Spark pods.

    Node selectors constrain pod scheduling to nodes with matching labels.
    Applied to both driver and executor pods.

    Supported backends:
        - Kubernetes

    Args:
        selectors: Dictionary of node label key-value pairs.

    Example::

        options = [
            NodeSelector(
                {
                    "node-type": "spark",
                    "gpu": "true",
                }
            ),
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

            TypeError: If the resource is not a supported Spark resource.
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

        elif isinstance(resource, models.SparkV1beta2SparkApplication):
            role_specs = [
                resource.spec.driver,
                resource.spec.executor,
            ]

            for role_spec in role_specs:
                if role_spec.node_selector is None:
                    role_spec.node_selector = {}

                role_spec.node_selector.update(self.selectors)
        else:
            raise TypeError(f"Unsupported Spark resource type: {type(resource).__name__}")


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

    Example::

        options = [
            Toleration(
                key="spark-workload",
                operator="Equal",
                value="true",
                effect="NoSchedule",
            ),
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

            TypeError: If the resource is not a supported Spark resource.
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

        elif isinstance(resource, models.SparkV1beta2SparkApplication):
            role_specs = [
                resource.spec.driver,
                resource.spec.executor,
            ]

            for role_spec in role_specs:
                if role_spec.tolerations is None:
                    role_spec.tolerations = []

                role_spec.tolerations.append(toleration)
        else:
            raise TypeError(f"Unsupported Spark resource type: {type(resource).__name__}")

@dataclass
class AddConnectJar:
    """Add the Spark Connect Maven JAR to spark.jars.

    Supported backends:
        - Kubernetes

    Args:
        enabled: Whether to add the Spark Connect JAR.
    """

    enabled: bool = True

    def __call__(
        self,
        spark_connect: models.SparkV1alpha1SparkConnect,
        backend: RuntimeBackend,
    ) -> None:
        """Apply the Spark Connect JAR configuration."""
        from kubeflow.spark.backends.kubernetes import constants
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"AddConnectJar option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )
        if not self.enabled:
            return
        spark_version = (spark_connect.spec.spark_version or constants.DEFAULT_SPARK_VERSION)
        connect_jar_url = (
            f"https://repo1.maven.org/maven2/org/apache/spark/"
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}/"
            f"{spark_version}/"
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-"
            f"{spark_version}.jar"
        )
        if spark_connect.spec.spark_conf is None:
            spark_connect.spec.spark_conf = {}

        existing_jars = spark_connect.spec.spark_conf.get("spark.jars", "").strip()

        spark_connect.spec.spark_conf["spark.jars"] = (
            f"{connect_jar_url},{existing_jars}"
            if existing_jars
            else connect_jar_url
        )
@dataclass
class Name:
    """Set a custom name for the Spark resource.

    This option sets the Kubernetes resource name.

    If not provided, a name is automatically generated:
    - Spark Connect sessions: `spark-connect-{uuid}`
    - Spark batch jobs: `spark-job-{uuid}`

    The session name must follow DNS-1123 subdomain rules:
    - Lowercase alphanumeric characters, '-', or '.'
    - Start and end with alphanumeric character
    - Maximum 253 characters

    Supported backends:
        - Kubernetes

    Args:
        name: Custom name for the session. Must be a valid Kubernetes resource name.

    Example::

        from kubeflow.spark import SparkClient
        from kubeflow.spark.options import Name

        client = SparkClient()

        # With explicit name
        spark = client.connect(options=[Name("my-custom-session")])

        # Auto-generated name
        spark = client.connect()  # Creates "spark-connect-a1b2c3d4"
    """

    name: str

    def __call__(self, resource: SparkResource, backend: RuntimeBackend) -> None:
        """Apply custom name to the Spark resource metadata.

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
