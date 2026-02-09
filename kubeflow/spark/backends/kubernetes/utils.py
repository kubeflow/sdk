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

"""Utility functions for Kubernetes Spark backend."""

import contextlib
from datetime import datetime
import re
from typing import Any, Optional
from urllib.parse import urlparse
import uuid

from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.types.types import Driver, Executor, SparkConnectInfo, SparkConnectState

# Type alias for backend to avoid circular imports
if False:  # TYPE_CHECKING equivalent without import
    pass


def generate_session_name() -> str:
    """Generate a unique session name.

    Returns:
        Session name in format: spark-connect-{uuid}.
    """
    short_uuid = str(uuid.uuid4())[:8]
    return f"{constants.SESSION_NAME_PREFIX}-{short_uuid}"


def validate_spark_connect_url(url: str) -> bool:
    """Validate a Spark Connect URL.

    Args:
        url: URL to validate (e.g., "sc://host:15002").

    Returns:
        True if valid.

    Raises:
        ValueError: If URL is invalid.
    """
    parsed = urlparse(url)
    if parsed.scheme != "sc":
        raise ValueError(f"Invalid scheme '{parsed.scheme}'. Expected 'sc://'")
    if not parsed.port:
        raise ValueError("Port is required in Spark Connect URL")
    return True


def _memory_kubernetes_to_spark(memory: str) -> str:
    """Convert Kubernetes-style memory (e.g. 4Gi, 512Mi) to Spark/JVM style (4g, 512m).

    SparkSubmit expects JVM memory suffixes (k, m, g, t); Kubernetes uses Ki, Mi, Gi, Ti.
    """
    if not memory or not memory[-1].isalpha():
        return memory
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGTPE]i?|k|m|g|t|kb|mb|gb|tb)?$", memory, re.IGNORECASE)
    if not m:
        return memory
    num, suffix = m.group(1), (m.group(2) or "").lower()
    k8s_to_spark = {"ki": "k", "mi": "m", "gi": "g", "ti": "t", "pi": "p", "ei": "e"}
    spark_suffix = k8s_to_spark.get(suffix, suffix.rstrip("b") if suffix else "")
    return num + spark_suffix


def build_service_url(info: SparkConnectInfo) -> str:
    """Build Spark Connect URL from session info.

    Args:
        info: SparkConnectInfo with service details.

    Returns:
        Spark Connect URL (e.g., "sc://service-name:15002").
    """
    service = info.service_name or f"{info.name}-svc"
    return f"sc://{service}.{info.namespace}.svc.cluster.local:{constants.SPARK_CONNECT_PORT}"


def build_spark_connect_crd(
    name: str,
    namespace: str,
    spark_version: Optional[str] = None,
    num_executors: Optional[int] = None,
    resources_per_executor: Optional[dict[str, str]] = None,
    spark_conf: Optional[dict[str, str]] = None,
    driver: Optional[Driver] = None,
    executor: Optional[Executor] = None,
    options: Optional[list] = None,
    backend: Optional[Any] = None,
) -> dict[str, Any]:
    """Build SparkConnect CRD manifest (KEP-107 compliant).

    Precedence rules:
    - Executor instances: executor.num_instances > num_executors > default
    - Executor resources: executor.resources_per_executor > resources_per_executor
    - Driver resources: driver.resources (only source)
    - Image: driver.image > default

    Args:
        name: Session name.
        namespace: Kubernetes namespace.
        spark_version: Spark version (default: 3.4.1).
        num_executors: Number of executor instances (simple mode).
        resources_per_executor: Resource requirements per executor (simple mode).
        spark_conf: Spark configuration properties.
        driver: Driver configuration (advanced mode).
        executor: Executor configuration (advanced mode).
        options: List of configuration options (advanced mode).
        backend: Backend instance for option validation.

    Returns:
        SparkConnect CRD as dictionary.
    """
    spark_version = spark_version or constants.DEFAULT_SPARK_VERSION

    # Precedence: executor.num_instances > num_executors > default
    executor_spec: dict[str, Any] = {}
    if executor and executor.num_instances is not None:
        executor_spec["instances"] = executor.num_instances
    elif num_executors is not None:
        executor_spec["instances"] = num_executors
    else:
        executor_spec["instances"] = constants.DEFAULT_NUM_EXECUTORS

    # Precedence: executor.resources_per_executor > resources_per_executor
    resource_dict = None
    if executor and executor.resources_per_executor:
        resource_dict = executor.resources_per_executor
    elif resources_per_executor:
        resource_dict = resources_per_executor

    if resource_dict:
        if "cpu" in resource_dict:
            executor_spec["cores"] = int(resource_dict["cpu"])
        if "memory" in resource_dict:
            executor_spec["memory"] = _memory_kubernetes_to_spark(resource_dict["memory"])
    if "cores" not in executor_spec:
        executor_spec["cores"] = constants.DEFAULT_EXECUTOR_CPU
    if "memory" not in executor_spec:
        executor_spec["memory"] = _memory_kubernetes_to_spark(constants.DEFAULT_EXECUTOR_MEMORY)

    server_spec: dict[str, Any] = {}
    if driver:
        if driver.resources:
            if "cpu" in driver.resources:
                server_spec["cores"] = int(driver.resources["cpu"])
            if "memory" in driver.resources:
                server_spec["memory"] = _memory_kubernetes_to_spark(driver.resources["memory"])

        if driver.service_account:
            if "template" not in server_spec:
                server_spec["template"] = {"spec": {}}
            server_spec["template"]["spec"]["serviceAccountName"] = driver.service_account
    if "cores" not in server_spec:
        server_spec["cores"] = constants.DEFAULT_DRIVER_CPU
    if "memory" not in server_spec:
        server_spec["memory"] = _memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY)

    crd: dict[str, Any] = {
        "apiVersion": f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}",
        "kind": constants.SPARK_CONNECT_KIND,
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "sparkVersion": spark_version,
            "server": server_spec,
            "executor": executor_spec,
        },
    }

    # Precedence: driver.image > default
    if driver and driver.image:
        crd["spec"]["image"] = driver.image
    else:
        crd["spec"]["image"] = constants.DEFAULT_SPARK_IMAGE

    # Use direct JAR URL to avoid Ivy cache (container may not have writable ~/.ivy2)
    connect_jar_url = (
        f"https://repo1.maven.org/maven2/org/apache/spark/"
        f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}/{spark_version}/"
        f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-{spark_version}.jar"
    )
    # Server listens on all interfaces so port-forward and in-cluster access work (Spark Connect config)
    base_conf: dict[str, str] = {
        "spark.jars": connect_jar_url,
        "spark.connect.grpc.binding.address": "0.0.0.0",
    }
    if spark_conf:
        existing_jars = spark_conf.get("spark.jars", "").strip()
        if existing_jars:
            base_conf["spark.jars"] = f"{connect_jar_url},{existing_jars}"
        for k, v in spark_conf.items():
            if k != "spark.jars":
                base_conf[k] = v
    crd["spec"]["sparkConf"] = base_conf

    # Apply options - extensibility without API changes (callable pattern)
    if options and backend is not None:
        for option in options:
            if callable(option):
                option(crd, backend)

    return crd


def parse_spark_connect_status(crd_response: dict[str, Any]) -> SparkConnectInfo:
    """Parse SparkConnect CRD response into SparkConnectInfo.

    Args:
        crd_response: Raw CRD response from Kubernetes API.

    Returns:
        SparkConnectInfo with parsed status.
    """
    metadata = crd_response.get("metadata", {})
    status = crd_response.get("status", {})
    server_status = status.get("server", {})

    state_str = status.get("state", "")
    try:
        state = SparkConnectState(state_str) if state_str else SparkConnectState.PROVISIONING
    except ValueError:
        state = SparkConnectState.PROVISIONING

    # Parse creation timestamp
    creation_timestamp = None
    creation_ts = metadata.get("creationTimestamp")
    if creation_ts:
        with contextlib.suppress(ValueError, AttributeError):
            creation_timestamp = datetime.fromisoformat(creation_ts.replace("Z", "+00:00"))

    return SparkConnectInfo(
        name=metadata.get("name", ""),
        namespace=metadata.get("namespace", ""),
        state=state,
        pod_name=server_status.get("podName"),
        pod_ip=server_status.get("podIp"),
        service_name=server_status.get("serviceName"),
        creation_timestamp=creation_timestamp,
    )
