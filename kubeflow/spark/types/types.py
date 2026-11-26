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

import kubeflow.common.constants as common_constants

logger = logging.getLogger(__name__)


class SparkConnectState(str, Enum):
    """State of a SparkConnect session."""

    PROVISIONING = "Provisioning"
    READY = "Ready"
    NOT_READY = "NotReady"
    FAILED = "Failed"


@dataclass
class SparkConnectAuth:
    """Authentication configuration for a Spark Connect session.

    Supports three mechanisms that cover the ways a Spark Connect client
    can authenticate:

    * **Bearer token** – sent via ``spark.connect.authenticate.token``.
    * **Extra gRPC headers** – arbitrary key/value pairs appended to the
      Spark Connect URL as ``;key=value`` parameters, which PySpark
      forwards as gRPC metadata.  Useful for gateways and proxies that
      expect their own header (an API key, an identity header, etc.).
    * **Custom channel builder** – a ``ChannelBuilder`` instance passed
      to ``SparkSession.builder.channelBuilder()`` for mTLS client
      certificates, tokens that expire and need refreshing, or gRPC
      interceptors.

    Args:
        token: Bearer token string.
        headers: Extra gRPC metadata headers as ``{name: value}`` dict.
        channel_builder: A PySpark ``ChannelBuilder`` (or compatible
            object) for full control over the gRPC channel.

    Raises:
        ValueError: If ``channel_builder`` is combined with ``token``
            or ``headers``, since the channel builder owns the entire
            channel and those options would be silently ignored.

    Example::

        from kubeflow.spark import SparkClient, SparkConnectAuth

        client = SparkClient()

        # Bearer token
        client.connect(
            base_url="sc://spark.example:443",
            auth=SparkConnectAuth(token="my-token"),
        )

        # Gateway / proxy header
        client.connect(
            base_url="sc://gateway:443",
            auth=SparkConnectAuth(headers={"x-api-key": "key-123"}),
        )

        # Custom channel builder (mTLS, token refresh, interceptors)
        client.connect(
            base_url="sc://spark.example:443",
            auth=SparkConnectAuth(channel_builder=my_builder),
        )
    """

    token: str | None = None
    headers: dict[str, str] | None = None
    channel_builder: Any | None = None

    def __post_init__(self) -> None:
        if self.channel_builder is not None and (self.token or self.headers):
            raise ValueError(
                "channel_builder cannot be combined with token or headers; "
                "the channel builder owns the entire gRPC channel"
            )


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
    state: str = common_constants.UNKNOWN
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

    Example::

        driver = Driver(
            resources={
                "cpu": "4",
                "memory": "8Gi",
            },
            service_account="spark-driver-prod",
        )

    Note:
        The resources dict is extensible - any valid Kubernetes resource name is supported.
        This design allows future resource types without API changes.
    """

    image: str | None = None
    resources: dict | None = None
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

    Example::

        executor = Executor(
            num_instances=20,
            resources_per_executor={
                "cpu": "8",
                "memory": "32Gi",
            },
        )

    Note:
        The resources_per_executor dict is extensible - any valid Kubernetes resource
        name is supported. This design allows future resource types without API changes.
    """

    num_instances: int | None = None
    resources_per_executor: dict | None = None
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
    status: str = common_constants.UNKNOWN
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
