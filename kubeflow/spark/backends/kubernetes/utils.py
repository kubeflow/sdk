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

import hashlib
import logging
import os
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import urlparse
import uuid

from kubeflow_spark_api import models

from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.types.types import (
    Driver,
    Executor,
    SparkConnectInfo,
    SparkConnectState,
    SparkJob,
    SparkJobStatus,
)

logger = logging.getLogger(__name__)


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


def get_server_spec_from_driver(
    driver: Driver | None = None,
) -> models.SparkV1alpha1ServerSpec:
    """Convert SDK Driver to API ServerSpec.

    Args:
        driver: SDK Driver configuration.

    Returns:
        API ServerSpec model.
    """
    cores = constants.DEFAULT_DRIVER_CPU
    memory = _memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY)
    template = None

    if driver:
        if driver.resources:
            if "cpu" in driver.resources:
                cores = int(driver.resources["cpu"])
            if "memory" in driver.resources:
                memory = _memory_kubernetes_to_spark(driver.resources["memory"])

        if driver.service_account:
            # PodSpec requires containers field (can be empty list)
            template = models.IoK8sApiCoreV1PodTemplateSpec(
                spec=models.IoK8sApiCoreV1PodSpec(
                    containers=[],
                    service_account_name=driver.service_account,
                )
            )

    return models.SparkV1alpha1ServerSpec(
        cores=cores,
        memory=memory,
        template=template,
    )


def get_executor_spec_from_executor(
    executor: Executor | None = None,
    num_executors: int | None = None,
    resources_per_executor: dict[str, str] | None = None,
) -> models.SparkV1alpha1ExecutorSpec:
    """Convert SDK Executor to API ExecutorSpec.

    Precedence rules:
    - Instances: executor.num_instances > num_executors > default
    - Resources: executor.resources_per_executor > resources_per_executor

    Args:
        executor: SDK Executor configuration.
        num_executors: Simple mode number of executors.
        resources_per_executor: Simple mode resource requirements.

    Returns:
        API ExecutorSpec model.
    """
    # Determine number of instances
    if executor and executor.num_instances:
        instances = executor.num_instances
    elif num_executors:
        instances = num_executors
    else:
        instances = constants.DEFAULT_NUM_EXECUTORS

    # Determine resource dict
    resource_dict = None
    if executor and executor.resources_per_executor:
        resource_dict = executor.resources_per_executor
    elif resources_per_executor:
        resource_dict = resources_per_executor

    # Extract cores and memory
    cores = constants.DEFAULT_EXECUTOR_CPU
    memory = _memory_kubernetes_to_spark(constants.DEFAULT_EXECUTOR_MEMORY)

    if resource_dict:
        if "cpu" in resource_dict:
            cores = int(resource_dict["cpu"])
        if "memory" in resource_dict:
            memory = _memory_kubernetes_to_spark(resource_dict["memory"])

    return models.SparkV1alpha1ExecutorSpec(
        instances=instances,
        cores=cores,
        memory=memory,
    )


def build_spark_connect_cr(
    name: str,
    namespace: str,
    spark_version: str | None = None,
    num_executors: int | None = None,
    resources_per_executor: dict[str, str] | None = None,
    spark_conf: dict[str, str] | None = None,
    driver: Driver | None = None,
    executor: Executor | None = None,
    options: list | None = None,
    backend: Any | None = None,
) -> models.SparkV1alpha1SparkConnect:
    """Build SparkConnect CR using typed API models (KEP-107 compliant).

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
        SparkConnect CR as typed Pydantic model.
    """
    spark_version = spark_version or constants.DEFAULT_SPARK_VERSION

    # Build server spec using conversion function
    server_spec = get_server_spec_from_driver(driver)

    # Build executor spec using conversion function
    executor_spec = get_executor_spec_from_executor(executor, num_executors, resources_per_executor)

    # Determine image (driver.image > default)
    image = driver.image if driver and driver.image else constants.DEFAULT_SPARK_IMAGE

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

    # Build the typed SparkConnect model
    spark_connect = models.SparkV1alpha1SparkConnect(
        api_version=f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}",
        kind=constants.SPARK_CONNECT_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
        ),
        spec=models.SparkV1alpha1SparkConnectSpec(
            spark_version=spark_version,
            image=image,
            server=server_spec,
            executor=executor_spec,
            spark_conf=base_conf,
        ),
    )

    # Apply options - extensibility without API changes (callable pattern)
    if options and backend is not None:
        for option in options:
            if callable(option):
                option(spark_connect, backend)

    return spark_connect


def get_spark_connect_info_from_cr(
    spark_connect_cr: models.SparkV1alpha1SparkConnect,
) -> SparkConnectInfo:
    """Convert API SparkConnect model to SDK SparkConnectInfo.

    Args:
        spark_connect_cr: API SparkConnect model.

    Returns:
        SDK SparkConnectInfo dataclass.

    Raises:
        ValueError: If the CR is invalid.
    """
    if not (spark_connect_cr.metadata and spark_connect_cr.metadata.name):
        raise ValueError(f"SparkConnect CR is invalid: {spark_connect_cr}")

    # Parse state
    state = SparkConnectState.PROVISIONING
    if spark_connect_cr.status and spark_connect_cr.status.state:
        try:
            state = SparkConnectState(spark_connect_cr.status.state)
        except ValueError:
            state = SparkConnectState.PROVISIONING

    # Extract server status
    server_status = None
    if spark_connect_cr.status and spark_connect_cr.status.server:
        server_status = spark_connect_cr.status.server

    return SparkConnectInfo(
        name=spark_connect_cr.metadata.name,
        namespace=spark_connect_cr.metadata.namespace,
        state=state,
        pod_name=server_status.pod_name if server_status else None,
        pod_ip=server_status.pod_ip if server_status else None,
        service_name=server_status.service_name if server_status else None,
        creation_timestamp=spark_connect_cr.metadata.creation_timestamp,
    )

def generate_job_name() -> str:
    """Generate a unique batch job name.

    Returns:
        Job name in format: spark-job-{uuid}.
    """
    short_uuid = str(uuid.uuid4())[:8]
    return f"{constants.JOB_NAME_PREFIX}-{short_uuid}"


def _is_local_main_file(main_file: str) -> bool:
    """Return True if main_file is a local filesystem path (not a remote URI)."""
    return not any(main_file.startswith(scheme) for scheme in constants.REMOTE_URI_SCHEMES)


def _build_job_image(main_file: str, base_image: str) -> str:
    """Build a Docker image containing the user's Python script.

    The image tag is derived from the SHA-256 hash of the script content so that
    identical scripts produce the same tag.

    Dockerfile generated:
        FROM <base_image>
        COPY <script_name> <JOB_SCRIPT_MOUNT_PATH>/<script_name>

    Args:
        main_file: Local path to the Python script.
        base_image: Base Spark image to build FROM.

    Returns:
        Built image tag (e.g., "spark-job-abc12345:latest").

    Raises:
        FileNotFoundError: If main_file does not exist.
        ImportError: If the docker Python package is not installed.
        RuntimeError: If the Docker build fails.
    """
    try:
        import docker as docker_sdk  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The 'docker' Python package is required to build Spark job images. "
            "Install it with: pip install kubeflow[docker]"
        ) from e

    if not os.path.isfile(main_file):
        raise FileNotFoundError(f"main_file not found: {main_file}")

    # Derive deterministic image tag from file content hash
    with open(main_file, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:8]
    image_tag = f"{constants.JOB_IMAGE_PREFIX}-{content_hash}:latest"

    try:
        docker_sdk.from_env().images.get(image_tag)
        logger.info("Image '%s' already exists for same file content, reusing it", image_tag)
        return image_tag
    except Exception:
        pass  # Image not found locally, build it

    script_name = os.path.basename(main_file)
    dockerfile_content = (
        f"FROM {base_image}\n"
        f"COPY {script_name} {constants.JOB_SCRIPT_MOUNT_PATH}/{script_name}\n"
    )

    tmpdir = tempfile.mkdtemp(prefix="spark-job-build-")
    try:
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        shutil.copy2(main_file, os.path.join(tmpdir, script_name))

        client = docker_sdk.from_env()
        _, build_logs = client.images.build(path=tmpdir, tag=image_tag, rm=True)

    except docker_sdk.errors.BuildError as e:
        raise RuntimeError(f"Docker build failed for '{main_file}': {e}") from e
    except docker_sdk.errors.DockerException as e:
        raise RuntimeError(
            f"Docker daemon error while building image for '{main_file}': {e}. "
            "Is Docker running?"
        ) from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return image_tag


def build_spark_application_cr(
    name: str,
    namespace: str,
    main_file: str,
    image: str,
    arguments: list[str] | None = None,
    spark_conf: dict[str, str] | None = None,
) -> models.SparkV1beta2SparkApplication:
    """Build a SparkApplication CR using typed API models.

    Args:
        name: Job name (used as the Kubernetes resource name).
        namespace: Kubernetes namespace.
        main_file: Path to the main Python file (local:// or remote URI).
        image: Docker image for driver and executors.
        arguments: Optional list of arguments passed to the application.
        spark_conf: Optional Spark configuration properties.

    Returns:
        SparkV1beta2SparkApplication model ready for submission.
    """
    driver_spec = models.SparkV1beta2DriverSpec(
        cores=constants.DEFAULT_DRIVER_CPU,
        memory=_memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY),
    )
    executor_spec = models.SparkV1beta2ExecutorSpec(
        instances=constants.DEFAULT_NUM_EXECUTORS,
        cores=constants.DEFAULT_EXECUTOR_CPU,
        memory=_memory_kubernetes_to_spark(constants.DEFAULT_EXECUTOR_MEMORY),
    )

    return models.SparkV1beta2SparkApplication(
        api_version=f"{constants.SPARK_APPLICATION_GROUP}/{constants.SPARK_APPLICATION_VERSION}",
        kind=constants.SPARK_APPLICATION_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
        ),
        spec=models.SparkV1beta2SparkApplicationSpec(
            type=constants.SPARK_APPLICATION_TYPE_PYTHON,
            image=image,
            spark_version=constants.DEFAULT_SPARK_VERSION,
            main_application_file=main_file,
            arguments=arguments or None,
            spark_conf=spark_conf or None,
            driver=driver_spec,
            executor=executor_spec,
        ),
    )


def get_spark_application_info_from_cr(
    cr: models.SparkV1beta2SparkApplication,
) -> SparkJob:
    """Convert API SparkApplication model to SDK SparkJob.

    Args:
        cr: SparkV1beta2SparkApplication model from the Kubernetes API.

    Returns:
        SDK SparkJob dataclass.
    """
    if not (cr.metadata and cr.metadata.name):
        raise ValueError(f"SparkApplication CR is invalid: {cr}")

    status = SparkJobStatus.NEW
    error_message = None
    if cr.status and cr.status.application_state:
        raw_state = cr.status.application_state.state or ""
        try:
            status = SparkJobStatus(raw_state)
        except ValueError:
            status = SparkJobStatus.UNKNOWN
        error_message = cr.status.application_state.error_message

    driver_pod_name = None
    if cr.status and cr.status.driver_info:
        driver_pod_name = cr.status.driver_info.pod_name

    return SparkJob(
        name=cr.metadata.name,
        namespace=cr.metadata.namespace or "",
        status=status,
        driver_pod_name=driver_pod_name,
        error_message=error_message,
    )
