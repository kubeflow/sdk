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

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


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
        pod_name: Name of the server pod.
        pod_ip: IP address of the server pod.
        service_name: Name of the Kubernetes service.
        creation_timestamp: Timestamp when the session was created.
    """

    name: str
    namespace: str
    state: SparkConnectState
    pod_name: Optional[str] = None
    pod_ip: Optional[str] = None
    service_name: Optional[str] = None
    creation_timestamp: Optional[datetime] = None


@dataclass
class Driver:
    """Driver configuration for Spark Connect session (KEP-107 lines 165-170).

    The Driver configuration allows fine-grained control over the Spark driver pod.
    All fields are optional, with sensible defaults applied by the backend.

    Args:
        image: Custom container image for the driver.
        resources: Resource requirements as dict (e.g., {"cpu": "2", "memory": "4Gi"}).
            Supports arbitrary Kubernetes resources including GPUs (nvidia.com/gpu).
        java_options: JVM options for the driver (e.g., "-Xmx4g -XX:+UseG1GC").
        service_account: Kubernetes service account name for RBAC.

    Example:
        driver = Driver(
            resources={"cpu": "4", "memory": "8Gi"},
            service_account="spark-driver-prod"
        )

    Note:
        The resources dict is extensible - any valid Kubernetes resource name is supported.
        This design allows GPU support and future resource types without API changes.
    """

    image: Optional[str] = None
    resources: Optional[dict[str, str]] = None
    java_options: Optional[str] = None
    service_account: Optional[str] = None


@dataclass
class Executor:
    """Executor configuration for Spark Connect session (KEP-107 lines 172-177).

    The Executor configuration controls the worker pods that execute Spark tasks.
    All fields are optional, with sensible defaults applied by the backend.

    Args:
        num_instances: Number of executor instances (pods).
        resources_per_executor: Resource requirements per executor as dict
            (e.g., {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"}).
            Supports arbitrary Kubernetes resources for future extensibility.
        java_options: JVM options for executors (e.g., "-Xmx28g -XX:+UseG1GC").

    Example:
        executor = Executor(
            num_instances=20,
            resources_per_executor={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "2"}
        )

    Note:
        The resources_per_executor dict is extensible - any valid Kubernetes resource
        name is supported. This design allows GPU support, custom devices, and future
        resource types without API changes.
    """

    num_instances: Optional[int] = None
    resources_per_executor: Optional[dict[str, str]] = None
    java_options: Optional[str] = None
