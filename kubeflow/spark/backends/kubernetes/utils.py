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

from collections.abc import Callable, Iterator
import inspect
import logging
import math
import multiprocessing
import os
import re
import textwrap
from typing import Any
from urllib.parse import urlparse
import uuid

from kubeflow_spark_api import models
from kubernetes import client

from kubeflow.common import constants as common_constants
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

# ----------------------------------------------------------------------
# Shared utility functions
# ----------------------------------------------------------------------


def read_pod_logs(
    core_api: client.CoreV1Api,
    namespace: str,
    pod_name: str,
    follow: bool = False,
) -> Iterator[str]:
    """Read logs from a Kubernetes pod.

    Args:
        core_api: Kubernetes CoreV1Api client.
        namespace: Kubernetes namespace.
        pod_name: Name of the pod.
        follow: Whether to stream logs continuously.

    Yields:
        Log lines from the pod.

    Raises:
        TimeoutError: If retrieving pod logs times out.
        RuntimeError: If pod logs cannot be retrieved.
    """
    try:
        if follow:
            thread = core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                follow=True,
                _preload_content=False,
                async_req=True,
            )

            resp = thread.get(common_constants.DEFAULT_TIMEOUT)

            for line in resp.stream():
                yield line.decode("utf-8").rstrip("\n")
        else:
            thread = core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                async_req=True,
            )

            logs = thread.get(common_constants.DEFAULT_TIMEOUT)

            for line in logs.split("\n"):
                yield line

    except multiprocessing.TimeoutError as e:
        raise TimeoutError("Timeout while retrieving pod logs.") from e

    except Exception as e:
        raise RuntimeError("Failed to retrieve pod logs.") from e


def _resolve_driver_resources(
    driver: Driver | None = None,
) -> tuple[int, str]:
    """Resolve driver resource configuration.

    Args:
        driver: Driver configuration.

    Returns:
        Tuple of (cores, memory).

    Raises:
        ValueError:
            If the configured CPU or memory values are invalid.
    """

    cores = constants.DEFAULT_DRIVER_CPU
    memory = _memory_kubernetes_to_spark(constants.DEFAULT_DRIVER_MEMORY)

    if driver and driver.resources:
        if "cpu" in driver.resources:
            cores = _validate_cpu_value(driver.resources["cpu"])

        if "memory" in driver.resources:
            memory = _memory_kubernetes_to_spark(
                driver.resources["memory"],
            )

    return cores, memory


def _resolve_executor_resources(
    executor: Executor | None = None,
    num_executors: int | None = None,
    resources_per_executor: dict[str, str] | None = None,
) -> tuple[int, int, str]:
    """Resolve executor configuration.

    Args:
        executor:
            Executor configuration.

        num_executors:
            Number of executor instances.

        resources_per_executor:
            Resource requirements.

    Returns:
        Tuple containing ``(instances, cores, memory)``.

    Raises:
        ValueError:
            If the configured CPU or memory values are invalid.
    """

    if executor and executor.num_instances is not None:
        instances = executor.num_instances
    elif num_executors is not None:
        instances = num_executors
    else:
        instances = constants.DEFAULT_NUM_EXECUTORS

    resource_dict = None

    if executor and executor.resources_per_executor:
        resource_dict = executor.resources_per_executor
    elif resources_per_executor:
        resource_dict = resources_per_executor

    cores = constants.DEFAULT_EXECUTOR_CPU
    memory = _memory_kubernetes_to_spark(
        constants.DEFAULT_EXECUTOR_MEMORY,
    )

    if resource_dict:
        if "cpu" in resource_dict:
            cores = _validate_cpu_value(resource_dict["cpu"])

        if "memory" in resource_dict:
            memory = _memory_kubernetes_to_spark(
                resource_dict["memory"],
            )

    return instances, cores, memory


def _memory_kubernetes_to_spark(memory: str) -> str:
    """Convert Kubernetes-style memory to Spark-compatible memory.

    Spark accepts integer memory values with JVM suffixes (k, m, g, t).
    Kubernetes quantities may contain fractional values (e.g. 1.5Gi), so
    these are converted to an absolute MiB value.

    Args:
        memory: Memory value using Kubernetes or Spark notation.

    Returns:
        Memory value formatted using Spark-compatible units.
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


def _validate_cpu_value(cpu: str | int | None) -> int:
    """Validate and normalize CPU cores value.

    Args:
        cpu: CPU value provided by user.

    Returns:
        Integer CPU core value.

    Raises:
        ValueError: If CPU value is invalid.
    """
    if cpu is None:
        raise ValueError("CPU value cannot be None")

    if isinstance(cpu, int):
        cores = float(cpu)

    elif isinstance(cpu, str):
        cpu = cpu.strip()

        if not cpu:
            raise ValueError("CPU value cannot be empty")

        if cpu.endswith("m"):
            milli_cpu = cpu[:-1]

            if "." in milli_cpu:
                raise ValueError(
                    f"Invalid CPU value '{cpu}'. Decimal milli-CPU values are not supported."
                )

            cores = int(milli_cpu) / 1000

        else:
            cores = float(cpu)

    else:
        raise ValueError(f"Invalid CPU type '{type(cpu)}'. Expected str or int.")

    if not math.isfinite(cores) or cores <= 0:
        raise ValueError(f"Invalid CPU value: {cpu!r}")

    cores = math.ceil(cores)

    if cores > 1024:
        raise ValueError("CPU cores value is unrealistically large")

    return cores


def apply_options(
    resource: models.SparkV1alpha1SparkConnect | models.SparkV1beta2SparkApplication,
    options: list | None,
    backend: Any | None = None,
) -> None:
    """Apply configuration options to a Spark resource.

    Args:
        resource:
            Spark resource to configure.

        options:
            List of configuration options.

        backend:
            Backend used for option validation.

        Raises:
            ValueError:
                If options are provided without a backend instance.

            TypeError:
                If an option is not callable.
    """
    if not options:
        return

    if backend is None:
        raise ValueError("A backend instance is required to apply Spark options.")

    for option in options:
        if not callable(option):
            raise TypeError(
                f"Invalid Spark option: {option!r}. Options must be callable, "
                "for example: Name, Labels, Annotations, NodeSelector or Toleration."
            )

        option(resource, backend)


def _validate_spark_conf(
    spark_conf: dict[str, str] | None,
) -> None:
    """Validate Spark configuration.

    Args:
        spark_conf:
            Spark configuration properties.

    Raises:
        ValueError:
            If ``spark_conf`` is not a dictionary of string keys and values.
    """
    if spark_conf is None:
        return

    if not isinstance(spark_conf, dict):
        raise ValueError("spark_conf must be a dictionary.")

    for key, value in spark_conf.items():
        if not isinstance(key, str):
            raise ValueError("All spark_conf keys must be strings.")

        if not isinstance(value, str):
            raise ValueError("All spark_conf values must be strings.")


# ----------------------------------------------------------------------
# Spark Connect session utility functions
# ----------------------------------------------------------------------


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
    if not parsed.hostname:
        raise ValueError("Host is required in Spark Connect URL")
    if not parsed.port:
        raise ValueError("Port is required in Spark Connect URL")
    return True


def build_service_url(info: SparkConnectInfo) -> str:
    """Build the in-cluster Spark Connect URL from session info.

    Relies on the operator-reported Service name. The operator records the
    actual Service name (default "<name>-server" or a user-customized name via
    .spec.server.service) in Status.Server.ServiceName, so this is correct for
    both default and customized Services. Callers are expected to have waited
    for readiness (wait_for_ready), by which point the operator has populated
    this field in the same atomic status write that set the Ready state.

    Args:
        info: SparkConnectInfo with service details.

    Returns:
        Spark Connect URL
        (e.g., "sc://my-session-server.default.svc.cluster.local:15002").

    Raises:
        RuntimeError: If ``info.service_name`` is not populated. An empty value
            means the session is not ready yet, not a name we should guess;
            callers should wait for readiness before building the URL.
    """
    if not info.service_name:
        raise RuntimeError(
            f"Cannot build Spark Connect URL for {info.namespace}/{info.name}: "
            "status.server.serviceName is not populated yet. The session is not "
            "ready to accept in-cluster connections."
        )
    return (
        f"sc://{info.service_name}.{info.namespace}.svc.cluster.local"
        f":{constants.SPARK_CONNECT_PORT}"
    )


def get_spark_connect_driver_spec(
    driver: Driver | None = None,
) -> models.SparkV1alpha1ServerSpec:
    """Convert SDK Driver to API ServerSpec.

    Args:
        driver: SDK Driver configuration.

    Returns:
        API ServerSpec model.

    Raises:
        ValueError:
            If the configured driver resources are invalid.
    """
    cores, memory = _resolve_driver_resources(driver)

    template = None

    if driver and driver.service_account:
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


def get_spark_connect_executor_spec(
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

    Raises:
        ValueError:
            If the configured executor resources are invalid.
    """
    instances, cores, memory = _resolve_executor_resources(
        executor,
        num_executors,
        resources_per_executor,
    )

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
        spark_version: Spark version (default: `constants.DEFAULT_SPARK_VERSION`).
        num_executors: Number of executor instances (simple mode).
        resources_per_executor: Resource requirements per executor (simple mode).
        spark_conf: Spark configuration properties.
        driver: Driver configuration (advanced mode).
        executor: Executor configuration (advanced mode).
        options: List of configuration options (advanced mode).
        backend: Backend instance for option validation.

    Returns:
        SparkConnect CR as typed Pydantic model.

    Raises:
        ValueError:
            If the provided driver or executor resource configuration is invalid.
    """
    _validate_spark_conf(spark_conf)

    spark_version = spark_version or constants.DEFAULT_SPARK_VERSION

    # Build server spec using conversion function
    server_spec = get_spark_connect_driver_spec(driver)

    # Build executor spec using conversion function
    executor_spec = get_spark_connect_executor_spec(executor, num_executors, resources_per_executor)

    # Determine image (driver.image > SPARK_E2E_IMAGE > default)
    default_image = os.environ.get(
        "SPARK_E2E_IMAGE",
        constants.DEFAULT_SPARK_IMAGE,
    )

    image = driver.image if driver and driver.image else default_image

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
    apply_options(
        spark_connect,
        options,
        backend,
    )

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
        driver_pod_name=server_status.pod_name if server_status else None,
        pod_ip=server_status.pod_ip if server_status else None,
        service_name=server_status.service_name if server_status else None,
        creation_timestamp=spark_connect_cr.metadata.creation_timestamp,
    )


# ----------------------------------------------------------------------
# Spark batch job utility functions
# ----------------------------------------------------------------------


def generate_job_name() -> str:
    """Generate a unique batch job name.

    Returns:
        Job name in format: spark-job-{uuid}.
    """
    short_uuid = str(uuid.uuid4())[:8]
    return f"{constants.JOB_NAME_PREFIX}-{short_uuid}"


def get_spark_job_driver_spec(
    driver: Driver | None = None,
) -> models.SparkV1beta2DriverSpec:
    """Build DriverSpec for SparkApplication.

    Returns:
        SparkApplication DriverSpec model.

    Raises:
        ValueError:
            If the default driver resource configuration is invalid.
    """
    cores, memory = _resolve_driver_resources(driver)

    return models.SparkV1beta2DriverSpec(
        cores=cores,
        memory=memory,
        service_account=constants.DEFAULT_SERVICE_ACCOUNT,
    )


def get_spark_job_executor_spec(
    num_executors: int | None = None,
    resources_per_executor: dict[str, str] | None = None,
) -> models.SparkV1beta2ExecutorSpec:
    """Build ExecutorSpec for SparkApplication.

    Args:
        num_executors: Number of executor instances.
        resources_per_executor: Resource requirements for each executor.

    Returns:
        SparkApplication ExecutorSpec model.

    Raises:
        ValueError:
            If the configured executor resources are invalid.
    """
    instances, cores, memory = _resolve_executor_resources(
        num_executors=num_executors,
        resources_per_executor=resources_per_executor,
    )

    return models.SparkV1beta2ExecutorSpec(
        instances=instances,
        cores=cores,
        memory=memory,
    )


def get_func_job_init_container(
    command: list[str],
) -> models.IoK8sApiCoreV1Container:
    """Build the initContainer for a function-based Spark job.

    The initContainer executes the provided command to reconstruct the
    generated Spark application in the shared volume before the Spark
    driver starts.

    Args:
        command:
            Container command that generates the Spark application script.

    Returns:
        Kubernetes initContainer specification.
    """
    return models.IoK8sApiCoreV1Container(
        name=constants.FUNC_JOB_INIT_CONTAINER_NAME,
        image=constants.DEFAULT_SPARK_IMAGE,
        command=command,
        volume_mounts=[
            models.IoK8sApiCoreV1VolumeMount(
                name=constants.FUNC_JOB_VOLUME_NAME,
                mount_path=constants.FUNC_JOB_SCRIPT_DIR,
            ),
        ],
    )


def get_command_using_spark_func(
    func: Callable,
    func_args: dict[str, Any] | None,
) -> list[str]:
    """Get the Spark command from the given function and parameters.

    The generated command contains the user-defined function followed by its
    invocation. The command is written by the SparkApplication initContainer
    to a Python script and executed by the Spark driver.

    The provided function must be self-contained because only its source code
    is extracted. Any required imports should be placed inside the function
    body. Module-level imports, globals, closures, and decorated functions
    are not supported.


    Args:
        func:
            Python function to execute.

        func_args:
            Keyword arguments passed to the function.

    Returns:
        Container command that generates the Spark application script.

    Raises:
        ValueError:
            If ``func`` is not callable.
    """
    if not callable(func):
        raise ValueError(f"Expected a callable function, got {type(func)}.")

    func_code = textwrap.dedent(
        inspect.getsource(func),
    )

    if func_args is None:
        func_call = f"{func.__name__}()"
    else:
        func_call = f"{func.__name__}(**{repr(func_args)})"

    func_code = f"{func_code}\n{func_call}\n"
    return [
        "bash",
        "-c",
        constants.FUNC_JOB_SCRIPT_TEMPLATE.format(
            func_code=func_code,
            func_file=(f"{constants.FUNC_JOB_SCRIPT_DIR}/{constants.FUNC_JOB_SCRIPT_NAME}"),
        ),
    ]


def get_spark_application_cr_from_file_job(
    name: str,
    namespace: str,
    main_file: str,
    arguments: list[str] | None = None,
    num_executors: int | None = None,
    resources_per_executor: dict[str, str] | None = None,
    options: list | None = None,
    backend: Any | None = None,
    spark_conf: dict[str, str] | None = None,
) -> models.SparkV1beta2SparkApplication:
    """Build a SparkApplication custom resource for a file-based Spark job.

    Args:
        name: Job name.
        namespace: Kubernetes namespace.
        main_file: Path or URI to the Spark application file.
        arguments: Command-line arguments passed to the Spark application.
        num_executors: Number of executor instances.
        resources_per_executor: Resource requirements for each executor.
        options: List of configuration options.
        backend: Backend instance used for option validation.
        spark_conf: Spark configuration properties.

    Returns:
        SparkApplication custom resource model.

    Raises:
        ValueError:
            If the executor resource configuration is invalid.
    """
    spark_application = models.SparkV1beta2SparkApplication(
        api_version=f"{constants.SPARK_APPLICATION_GROUP}/{constants.SPARK_APPLICATION_VERSION}",
        kind=constants.SPARK_APPLICATION_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
        ),
        spec=models.SparkV1beta2SparkApplicationSpec(
            spark_version=constants.DEFAULT_SPARK_VERSION,
            type="Python",
            mode="cluster",
            image=constants.DEFAULT_SPARK_IMAGE,
            main_application_file=main_file,
            arguments=arguments or None,
            driver=get_spark_job_driver_spec(),
            executor=get_spark_job_executor_spec(
                num_executors=num_executors,
                resources_per_executor=resources_per_executor,
            ),
            spark_conf=spark_conf or None,
        ),
    )

    apply_options(
        spark_application,
        options,
        backend,
    )

    return spark_application


def get_spark_application_cr_from_func_job(
    name: str,
    namespace: str,
    func: Callable,
    func_args: dict[str, Any] | None = None,
    num_executors: int | None = None,
    resources_per_executor: dict[str, str] | None = None,
    options: list | None = None,
    backend: Any | None = None,
    spark_conf: dict[str, str] | None = None,
) -> models.SparkV1beta2SparkApplication:
    """Build a SparkApplication custom resource for a function-based Spark job.

    Args:
        name: Job name.
        namespace: Kubernetes namespace.
        func: Python function to execute as the Spark application.
        func_args: Keyword arguments passed to the function.
        num_executors: Number of executor instances.
        resources_per_executor: Resource requirements for each executor.
        options: List of configuration options.
        backend: Backend instance used for option validation.
        spark_conf: Spark configuration properties.

    Returns:
        SparkApplication custom resource model.

    Raises:
        ValueError:
            If the provided function is invalid or the executor resource
            configuration is invalid.
    """
    _validate_spark_conf(spark_conf)

    command = get_command_using_spark_func(
        func=func,
        func_args=func_args,
    )

    spark_application = models.SparkV1beta2SparkApplication(
        api_version=f"{constants.SPARK_APPLICATION_GROUP}/{constants.SPARK_APPLICATION_VERSION}",
        kind=constants.SPARK_APPLICATION_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
        ),
        spec=models.SparkV1beta2SparkApplicationSpec(
            spark_version=constants.DEFAULT_SPARK_VERSION,
            type="Python",
            mode="cluster",
            image=constants.DEFAULT_SPARK_IMAGE,
            main_application_file=constants.FUNC_JOB_MAIN_FILE,
            driver=get_spark_job_driver_spec(),
            executor=get_spark_job_executor_spec(
                num_executors=num_executors,
                resources_per_executor=resources_per_executor,
            ),
            spark_conf=spark_conf or None,
        ),
    )

    spark_application.spec.volumes = [
        models.IoK8sApiCoreV1Volume(
            name=constants.FUNC_JOB_VOLUME_NAME,
            empty_dir=models.IoK8sApiCoreV1EmptyDirVolumeSource(),
        ),
    ]

    spark_application.spec.driver.init_containers = [
        get_func_job_init_container(command),
    ]

    spark_application.spec.driver.volume_mounts = [
        models.IoK8sApiCoreV1VolumeMount(
            name=constants.FUNC_JOB_VOLUME_NAME,
            mount_path=constants.FUNC_JOB_SCRIPT_DIR,
        ),
    ]

    apply_options(
        spark_application,
        options,
        backend,
    )

    return spark_application


def get_spark_application_info_from_cr(
    cr: models.SparkV1beta2SparkApplication,
) -> SparkJob:
    """Convert SparkApplication CR to SparkJob.

    Args:
        cr: SparkApplication custom resource.

    Returns:
        SparkJob information object.
    """
    if not (cr.metadata and cr.metadata.name):
        raise ValueError(f"SparkApplication CR is invalid: {cr}")

    status = SparkJobStatus.CREATED
    creation_timestamp = cr.metadata.creation_timestamp
    num_executors = None
    driver_pod_name = None

    if cr.spec and cr.spec.executor:
        num_executors = cr.spec.executor.instances

    if cr.status:
        if cr.status.application_state:
            status = SparkJobStatus.from_operator_state(
                cr.status.application_state.state,
            )

        if cr.status.driver_info:
            driver_pod_name = cr.status.driver_info.pod_name

    return SparkJob(
        name=cr.metadata.name,
        namespace=cr.metadata.namespace or "",
        status=status,
        creation_timestamp=creation_timestamp,
        num_executors=num_executors,
        driver_pod_name=driver_pod_name,
    )
