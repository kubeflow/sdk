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


class SparkApplicationType(str, Enum):
    """SparkApplication ``spec.type`` values from the Spark Operator.

    Mirrors ``SparkApplicationType`` in spark-operator ``api/v1beta2``. Generated
    ``kubeflow_spark_api`` models currently expose these as plain strings; this enum
    provides typed SDK usage and a single place to detect drift with upstream.
    """

    JAVA = "Java"
    SCALA = "Scala"
    PYTHON = "Python"
    R = "R"


class SparkApplicationDeployMode(str, Enum):
    """SparkApplication ``spec.mode`` values from the Spark Operator.

    Mirrors ``DeployMode`` in spark-operator ``api/v1beta2``.
    """

    CLUSTER = "cluster"
    CLIENT = "client"
    IN_CLUSTER_CLIENT = "in-cluster-client"


class SparkApplicationState(str, Enum):
    """SparkApplication ``status.applicationState.state`` values from the Spark Operator.

    Mirrors ``ApplicationStateType`` in spark-operator ``api/v1beta2``. Generated
    Python models currently type this field as ``StrictStr``; prefer this enum in the
    SDK until ``kubeflow_spark_api`` exports an equivalent enum.
    """

    NEW = ""
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"
    PENDING_RERUN = "PENDING_RERUN"
    INVALIDATING = "INVALIDATING"
    SUCCEEDING = "SUCCEEDING"
    FAILING = "FAILING"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    RESUMING = "RESUMING"
    UNKNOWN = "UNKNOWN"


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
class Driver:
    """Driver configuration for Spark Connect session.

    The Driver configuration allows fine-grained control over the Spark driver pod.
    All fields are optional, with sensible defaults applied by the backend.

    Args:
        image: Custom container image for the driver.
        resources: Resource requirements as dict (e.g., {"cpu": "2", "memory": "4Gi"}).
        java_options: JVM options for the driver (e.g., "-Xmx4g -XX:+UseG1GC").
        service_account: Kubernetes service account name for RBAC.

    Example:
        driver = Driver(
            resources={"cpu": "4", "memory": "8Gi"},
            service_account="spark-driver-prod"
        )

    Note:
        The resources dict is extensible - any valid Kubernetes resource name is supported.
        This design allows future resource types without API changes.
    """

    image: str | None = None
    resources: dict[str, str] | None = None
    java_options: str | None = None
    service_account: str | None = None


@dataclass
class Executor:
    """Executor configuration for Spark Connect session.

    The Executor configuration controls the worker pods that execute Spark tasks.
    All fields are optional, with sensible defaults applied by the backend.

    Args:
        num_instances: Number of executor instances (pods).
        resources_per_executor: Resource requirements per executor as dict
            (e.g., {"cpu": "4", "memory": "8Gi"}).
        java_options: JVM options for executors (e.g., "-Xmx28g -XX:+UseG1GC").

    Example:
        executor = Executor(
            num_instances=20,
            resources_per_executor={"cpu": "8", "memory": "32Gi"}
        )

    Note:
        The resources_per_executor dict is extensible - any valid Kubernetes resource
        name is supported. This design allows future resource types without API changes.
    """

    num_instances: int | None = None
    resources_per_executor: dict[str, str] | None = None
    java_options: str | None = None


class SparkJobStatus(str, Enum):
    """State of a Spark batch job."""

    CREATED = "Created"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

    @classmethod
    def from_operator_state(
        cls,
        raw_state: str | SparkApplicationState | None,
    ) -> "SparkJobStatus":
        """Map a SparkApplication state to a SparkJobStatus.

        Args:
            raw_state: SparkApplication ``applicationState.state`` value. Accepts the
                operator string, a :class:`SparkApplicationState` value, or ``None``.

        Returns:
            Corresponding SparkJobStatus.

        Note:
            Unknown SparkApplication states default to FAILED so newly
            introduced operator states are handled conservatively.
        """
        if isinstance(raw_state, SparkApplicationState):
            operator_state = raw_state
        else:
            normalized_state = raw_state or SparkApplicationState.NEW.value
            try:
                operator_state = SparkApplicationState(normalized_state)
            except ValueError:
                try:
                    operator_state = SparkApplicationState(normalized_state.upper())
                except ValueError:
                    logger.warning(
                        "Unknown SparkApplication state '%s'. Defaulting to FAILED.",
                        raw_state,
                    )
                    return cls.FAILED

        status = _SDK_STATE_BY_OPERATOR_STATE.get(operator_state)
        if status is None:
            logger.warning(
                "Unmapped SparkApplication state '%s'. Defaulting to FAILED.",
                operator_state,
            )
            return cls.FAILED

        return status


_SDK_STATE_BY_OPERATOR_STATE: dict[SparkApplicationState, SparkJobStatus] = {
    state: sdk_status
    for sdk_status, operator_states in {
        SparkJobStatus.CREATED: (
            SparkApplicationState.NEW,
            SparkApplicationState.SUBMITTED,
        ),
        SparkJobStatus.RUNNING: (
            SparkApplicationState.RUNNING,
            SparkApplicationState.SUCCEEDING,
            SparkApplicationState.SUSPENDING,
            SparkApplicationState.SUSPENDED,
            SparkApplicationState.RESUMING,
        ),
        SparkJobStatus.COMPLETED: (SparkApplicationState.COMPLETED,),
        SparkJobStatus.FAILED: (
            SparkApplicationState.FAILED,
            SparkApplicationState.SUBMISSION_FAILED,
            SparkApplicationState.FAILING,
            SparkApplicationState.PENDING_RERUN,
            SparkApplicationState.INVALIDATING,
            SparkApplicationState.UNKNOWN,
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

    Args:
        func: Python function executed as a Spark batch job.
        func_args: Optional keyword arguments passed to the function.
    """

    func: Callable
    func_args: dict[str, Any] | None = None
