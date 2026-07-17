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

"""Types for Kubeflow Spark SDK."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SparkConnectState(str, Enum):
    """State of a SparkConnect session."""

    PROVISIONING = "Provisioning"
    READY = "Ready"
    RUNNING = "Running"  # Operator may set this when server is up; treated as ready
    NOT_READY = "NotReady"
    FAILED = "Failed"


@dataclass
class SparkConnectInfo:
    """Information about a SparkConnect session.

    Args:
        name: Name of the SparkConnect session.
        namespace: Kubernetes namespace. Included in SparkConnectInfo for standalone usage
            and passing info between components without requiring SparkClient context.
        state: Current state of the session.
        driver_pod_name: Name of the driver pod.
        pod_ip: IP address of the server pod.
        service_name: Name of the Kubernetes service.
        creation_timestamp: Timestamp when the session was created.
    """

    name: str
    namespace: str
    state: SparkConnectState
    driver_pod_name: str | None = None
    pod_ip: str | None = None
    service_name: str | None = None
    creation_timestamp: datetime | None = None


@dataclass
class ExistingSession:
    """Connect to an existing Spark Connect server.

    Args:
        base_url: Spark Connect URL (for example ``sc://server:15002``).
        token: Optional authentication token for the existing server.
    """

    base_url: str
    token: str | None = None


@dataclass
class NewSession:
    """Create a new Spark Connect session.

    Args:
        num_executors: Number of executor instances.
        resources_per_executor: Resource requirements per executor as a dict.
            Format: ``{"cpu": "5", "memory": "10Gi"}``.
        spark_conf: Spark configuration dictionary.
    """

    num_executors: int | None = None
    resources_per_executor: dict[str, str] | None = None
    spark_conf: dict[str, str] | None = None


@dataclass
class Driver:
    """Driver configuration for Spark Connect session.

    The Driver configuration allows fine-grained control over the Spark driver pod.
    All fields are optional, with sensible defaults applied by the backend.

    Prefer passing ``Driver(...)`` through ``connect(..., options=[...])`` so the
    create-mode API stays aligned with other Spark options.

    Args:
        image: Custom container image for the driver.
        resources: Resource requirements as dict (e.g., {"cpu": "2", "memory": "4Gi"}).
        java_options: JVM options for the driver (e.g., "-Xmx4g -XX:+UseG1GC").
        service_account: Kubernetes service account name for RBAC.

    Example:
        spark = client.connect(
            session=NewSession(),
            options=[
                Driver(
                    resources={"cpu": "4", "memory": "8Gi"},
                    service_account="spark-driver-prod",
                )
            ],
        )

    Note:
        The resources dict is extensible - any valid Kubernetes resource name is supported.
        This design allows future resource types without API changes.
    """

    image: str | None = None
    resources: dict[str, str] | None = None
    java_options: str | None = None
    service_account: str | None = None

    def __call__(self, spark_connect: Any, backend: Any) -> None:
        """Apply driver configuration to a SparkConnect model.

        Only Driver-owned fields are updated in place so options applied earlier
        (e.g. NodeSelector, Toleration, PodTemplateOverride) are preserved rather
        than discarded by rebuilding the server spec from scratch.
        """
        from kubeflow_spark_api import models

        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend
        from kubeflow.spark.backends.kubernetes.utils import _memory_kubernetes_to_spark

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Driver option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        if spark_connect.spec is None:
            raise ValueError("SparkConnect spec must be set before applying Driver")

        server_spec = spark_connect.spec.server
        if self.resources:
            if "cpu" in self.resources:
                server_spec.cores = int(self.resources["cpu"])
            if "memory" in self.resources:
                server_spec.memory = _memory_kubernetes_to_spark(self.resources["memory"])

        if self.service_account:
            if server_spec.template is None:
                server_spec.template = models.IoK8sApiCoreV1PodTemplateSpec()
            if server_spec.template.spec is None:
                # PodSpec requires containers field (can be empty list)
                server_spec.template.spec = models.IoK8sApiCoreV1PodSpec(containers=[])
            server_spec.template.spec.service_account_name = self.service_account

        if self.image:
            spark_connect.spec.image = self.image


@dataclass
class Executor:
    """Executor configuration for Spark Connect session.

    The Executor configuration controls the worker pods that execute Spark tasks.
    All fields are optional, with sensible defaults applied by the backend.

    Prefer passing ``Executor(...)`` through ``connect(..., options=[...])`` so the
    create-mode API stays aligned with other Spark options.

    Args:
        num_instances: Number of executor instances (pods).
        resources_per_executor: Resource requirements per executor as dict
            (e.g., {"cpu": "4", "memory": "8Gi"}).
        java_options: JVM options for executors (e.g., "-Xmx28g -XX:+UseG1GC").

    Example:
        spark = client.connect(
            session=NewSession(),
            options=[
                Executor(
                    num_instances=20,
                    resources_per_executor={
                        "cpu": "8",
                        "memory": "32Gi",
                        "nvidia.com/gpu": "2",
                    },
                )
            ],
        )

    Note:
        The resources_per_executor dict is extensible - any valid Kubernetes resource
        name is supported. This design allows future resource types without API changes.
    """

    num_instances: int | None = None
    resources_per_executor: dict[str, str] | None = None
    java_options: str | None = None

    def __call__(self, spark_connect: Any, backend: Any) -> None:
        """Apply executor configuration to a SparkConnect model.

        Only explicitly set Executor fields are merged into the existing executor
        spec so values already applied by ``NewSession``/legacy simple arguments or
        earlier template options (e.g. NodeSelector, Toleration) are preserved.
        """
        from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend
        from kubeflow.spark.backends.kubernetes.utils import _memory_kubernetes_to_spark

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Executor option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        if spark_connect.spec is None:
            raise ValueError("SparkConnect spec must be set before applying Executor")

        executor_spec = spark_connect.spec.executor
        if self.num_instances:
            executor_spec.instances = self.num_instances

        if self.resources_per_executor:
            if "cpu" in self.resources_per_executor:
                executor_spec.cores = int(self.resources_per_executor["cpu"])
            if "memory" in self.resources_per_executor:
                executor_spec.memory = _memory_kubernetes_to_spark(
                    self.resources_per_executor["memory"]
                )


class SparkJobStatus(str, Enum):
    """State of a Spark batch job."""

    CREATED = "Created"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

    @classmethod
    def from_operator_state(
        cls,
        raw_state: str | None,
    ) -> "SparkJobStatus":
        """Map a SparkApplication state to a SparkJobStatus.

        Args:
            raw_state: SparkApplication ``applicationState.state`` value.

        Returns:
            Corresponding SparkJobStatus.

        Note:
            Unknown SparkApplication states default to FAILED so newly
            introduced operator states are handled conservatively.
        """
        normalized_state = (raw_state or "").upper()

        status = _SDK_STATE_BY_OPERATOR_STATE.get(normalized_state)

        if status is None:
            logger.warning("Unknown SparkApplication state '%s'. Defaulting to FAILED.", raw_state)
            return cls.FAILED

        return status


_SDK_STATE_BY_OPERATOR_STATE: dict[str, SparkJobStatus] = {
    state: sdk_status
    for sdk_status, operator_states in {
        SparkJobStatus.CREATED: (
            "",
            "SUBMITTED",
        ),
        SparkJobStatus.RUNNING: (
            "RUNNING",
            "SUCCEEDING",
            "SUSPENDING",
            "SUSPENDED",
            "RESUMING",
        ),
        SparkJobStatus.COMPLETED: ("COMPLETED",),
        SparkJobStatus.FAILED: (
            "FAILED",
            "SUBMISSION_FAILED",
            "FAILING",
            "PENDING_RERUN",
            "INVALIDATING",
            "UNKNOWN",
        ),
    }.items()
    for state in operator_states
}


@dataclass
class SparkJob:
    """Information about a Spark batch job.

    Args:
        name: Name of the SparkApplication.
        namespace: Kubernetes namespace containing the SparkApplication.
            Included in SparkJob for standalone usage and passing job information
            between components without requiring SparkClient context.
        status: Current state of the Spark batch job.
        creation_timestamp: Timestamp when the SparkApplication was created.
        num_executors: Number of configured Spark executor instances.
        driver_pod_name: Name of the Spark driver pod, if available.
    """

    name: str
    namespace: str
    status: SparkJobStatus | None = None
    creation_timestamp: datetime | None = None
    num_executors: int | None = None
    driver_pod_name: str | None = None


@dataclass
class FileJob:
    """Spark application referenced by a local or remote file source.

    Args:
        file_source: Path or URI of the Spark application.
            Supports local paths available to the Spark cluster as well as
            remote URIs such as s3a://, gs://, hdfs:// and https://.
        args: Optional command-line arguments passed to the application.
    """

    file_source: str
    args: list[str] | None = None


@dataclass
class FuncJob:
    """Function-based Spark application.

    The provided function must be self-contained. Any required imports
    should be placed inside the function body. Module-level globals,
    closures, and decorated functions are not supported.

    Args:
        func: Python function executed as a Spark batch job.
        func_args: Optional keyword arguments passed to the function.
    """

    func: Callable
    func_args: dict[str, Any] | None = None
