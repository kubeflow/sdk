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

"""Kubernetes Spark Operator backend implementation."""

from collections.abc import Iterator
from dataclasses import dataclass, field
import logging
import multiprocessing
import time
from typing import Any, Optional

from kubernetes import client, config as k8s_config, watch

from kubeflow.spark.backends.base import BatchSparkBackend
from kubeflow.spark.models import (
    ApplicationState,
    ApplicationStatus,
    SparkApplicationResponse,
)

logger = logging.getLogger(__name__)


# Constants for Spark Operator
SPARK_OPERATOR_API_GROUP = "sparkoperator.k8s.io"
SPARK_OPERATOR_API_VERSION = "v1beta2"
SPARK_APPLICATION_PLURAL = "sparkapplications"
SPARK_APPLICATION_KIND = "SparkApplication"
DEFAULT_TIMEOUT = 60  # seconds


@dataclass
class OperatorBackendConfig:
    """Configuration for Spark Operator backend.

    Attributes:
        namespace: Kubernetes namespace to use
        context: Kubernetes context name
        config_file: Path to kubeconfig file
        client_configuration: Custom Kubernetes client configuration
        service_account: Service account for Spark pods
        image_pull_policy: Image pull policy (IfNotPresent, Always, Never)
        default_spark_image: Default Docker image for Spark
        timeout: Default timeout for API operations in seconds
        enable_monitoring: Enable Prometheus monitoring
        enable_ui: Enable Spark UI service
    """

    namespace: Optional[str] = None
    context: Optional[str] = None
    config_file: Optional[str] = None
    client_configuration: Optional[client.Configuration] = None
    service_account: str = "spark-operator-spark"
    image_pull_policy: str = "IfNotPresent"
    default_spark_image: str = "gcr.io/spark-operator/spark-py"
    timeout: int = DEFAULT_TIMEOUT
    enable_monitoring: bool = True
    enable_ui: bool = True
    extra_labels: dict[str, str] = field(default_factory=dict)
    extra_annotations: dict[str, str] = field(default_factory=dict)


class OperatorBackend(BatchSparkBackend):
    """Kubernetes Spark Operator backend.

    This backend uses the Kubeflow Spark Operator to manage Spark applications
    on Kubernetes. It creates SparkApplication CRDs that the operator watches
    and converts into Kubernetes pods.

    Example:
        config = OperatorBackendConfig(namespace="spark-jobs")
        backend = OperatorBackend(config)
        response = backend.submit_application(
            app_name="my-spark-job",
            main_application_file="local:///app/main.py",
            ...
        )
    """

    def __init__(self, config: OperatorBackendConfig):
        """Initialize the Operator backend.

        Args:
            config: OperatorBackendConfig instance
        """
        self.config = config

        # Determine namespace
        if self.config.namespace is None:
            self.config.namespace = self._get_default_namespace()

        # Load Kubernetes configuration
        if self.config.client_configuration is None:
            if self.config.config_file or not self._is_running_in_k8s():
                k8s_config.load_kube_config(
                    config_file=self.config.config_file,
                    context=self.config.context,
                )
            else:
                k8s_config.load_incluster_config()

        # Initialize Kubernetes API clients
        k8s_client = client.ApiClient(self.config.client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)

        logger.info(f"Initialized OperatorBackend with namespace: {self.config.namespace}")

    def submit_application(
        self,
        app_name: str,
        main_application_file: str,
        spark_version: str = "3.5.0",
        app_type: str = "Python",
        driver_cores: int = 1,
        driver_memory: str = "1g",
        executor_cores: int = 1,
        executor_memory: str = "1g",
        num_executors: int = 2,
        queue: Optional[str] = None,
        arguments: Optional[list[str]] = None,
        python_version: str = "3",
        spark_conf: Optional[dict[str, str]] = None,
        hadoop_conf: Optional[dict[str, str]] = None,
        env_vars: Optional[dict[str, str]] = None,
        deps: Optional[dict[str, list[str]]] = None,
        **kwargs: Any,
    ) -> SparkApplicationResponse:
        """Submit a Spark application using Spark Operator.

        Creates a SparkApplication CRD in Kubernetes which the Spark Operator
        watches and converts into driver and executor pods.

        Args:
            app_name: Name of the application (must be DNS-compliant)
            main_application_file: Path to main application file
            spark_version: Spark version to use
            app_type: Application type (Python, Scala, Java, R)
            driver_cores: Number of cores for driver
            driver_memory: Memory for driver
            executor_cores: Number of cores per executor
            executor_memory: Memory per executor
            num_executors: Number of executors
            queue: Namespace to submit to (overrides config namespace)
            arguments: Application arguments
            python_version: Python version
            spark_conf: Spark configuration properties
            hadoop_conf: Hadoop configuration properties
            env_vars: Environment variables
            deps: Dependencies dict with keys: jars, pyFiles, files
            **kwargs: Additional parameters (volumes, node_selector, tolerations, etc.)

        Returns:
            SparkApplicationResponse with submission details

        Raises:
            ValueError: If required parameters are invalid
            RuntimeError: If submission fails
            TimeoutError: If submission times out
        """
        # Validate app_name is DNS-compliant
        if not self._is_valid_k8s_name(app_name):
            raise ValueError(
                f"app_name '{app_name}' must be DNS-compliant "
                "(lowercase alphanumeric characters, '-' or '.')"
            )

        # Determine target namespace
        target_namespace = queue if queue else self.config.namespace

        # Build SparkApplication CRD
        spark_app = self._build_spark_application_crd(
            app_name=app_name,
            main_application_file=main_application_file,
            spark_version=spark_version,
            app_type=app_type,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            num_executors=num_executors,
            arguments=arguments or [],
            python_version=python_version,
            spark_conf=spark_conf or {},
            hadoop_conf=hadoop_conf or {},
            env_vars=env_vars or {},
            deps=deps,
            **kwargs,
        )

        # Submit to Kubernetes
        try:
            thread = self.custom_api.create_namespaced_custom_object(
                group=SPARK_OPERATOR_API_GROUP,
                version=SPARK_OPERATOR_API_VERSION,
                namespace=target_namespace,
                plural=SPARK_APPLICATION_PLURAL,
                body=spark_app,
                async_req=True,
            )
            result = thread.get(self.config.timeout)

            logger.info(f"SparkApplication {target_namespace}/{app_name} created successfully")

            return SparkApplicationResponse(
                submission_id=app_name,
                app_name=app_name,
                status="SUBMITTED",
                message=f"SparkApplication created in namespace {target_namespace}",
            )

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout creating SparkApplication {target_namespace}/{app_name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create SparkApplication {target_namespace}/{app_name}: {e}"
            ) from e

    def get_job(self, submission_id: str) -> ApplicationStatus:
        """Get status of a Spark application.

        Args:
            submission_id: Name of the SparkApplication (same as app_name)

        Returns:
            ApplicationStatus with current status

        Raises:
            RuntimeError: If request fails
            TimeoutError: If request times out
        """
        try:
            thread = self.custom_api.get_namespaced_custom_object(
                group=SPARK_OPERATOR_API_GROUP,
                version=SPARK_OPERATOR_API_VERSION,
                namespace=self.config.namespace,
                plural=SPARK_APPLICATION_PLURAL,
                name=submission_id,
                async_req=True,
            )
            spark_app = thread.get(self.config.timeout)

            return self._parse_application_status(spark_app)

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout getting SparkApplication {self.config.namespace}/{submission_id}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get SparkApplication {self.config.namespace}/{submission_id}: {e}"
            ) from e

    def delete_job(self, submission_id: str) -> dict[str, Any]:
        """Delete a Spark application.

        Args:
            submission_id: Name of the SparkApplication to delete

        Returns:
            Dictionary with deletion response

        Raises:
            RuntimeError: If deletion fails
            TimeoutError: If deletion times out
        """
        try:
            thread = self.custom_api.delete_namespaced_custom_object(
                group=SPARK_OPERATOR_API_GROUP,
                version=SPARK_OPERATOR_API_VERSION,
                namespace=self.config.namespace,
                plural=SPARK_APPLICATION_PLURAL,
                name=submission_id,
                async_req=True,
            )
            result = thread.get(self.config.timeout)

            logger.info(f"SparkApplication {self.config.namespace}/{submission_id} deleted")

            return {
                "status": "deleted",
                "message": f"Application {submission_id} deleted",
            }

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout deleting SparkApplication {self.config.namespace}/{submission_id}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete SparkApplication {self.config.namespace}/{submission_id}: {e}"
            ) from e

    def get_job_logs(
        self,
        submission_id: str,
        executor_id: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get application logs from driver or executor pods.

        Args:
            submission_id: Name of the SparkApplication
            executor_id: Optional executor ID (e.g., "1", "2"). If None, returns driver logs
            follow: Whether to stream logs in real-time

        Yields:
            Log lines as strings

        Raises:
            RuntimeError: If request fails
        """
        # Determine pod name based on executor_id
        if executor_id:
            # Executor pod naming: <app-name>-<executor-id>
            pod_name = f"{submission_id}-{executor_id}"
            container_name = "executor"
        else:
            # Driver pod naming: <app-name>-driver
            pod_name = f"{submission_id}-driver"
            container_name = "spark-kubernetes-driver"

        try:
            if follow:
                # Stream logs in real-time
                log_stream = watch.Watch().stream(
                    self.core_api.read_namespaced_pod_log,
                    name=pod_name,
                    namespace=self.config.namespace,
                    container=container_name,
                    follow=True,
                )
                yield from log_stream
            else:
                # Get all logs at once
                logs = self.core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.config.namespace,
                    container=container_name,
                )
                yield from logs.splitlines()

        except client.exceptions.ApiException as e:
            if e.status == 404:
                logger.warning(f"Pod {pod_name} not found in namespace {self.config.namespace}")
                return
            elif e.status == 400 and (
                "waiting to start" in str(e.body) or "ContainerCreating" in str(e.body)
            ):
                # Pod exists but container is not ready yet
                # Check if it's a "waiting to start" error
                logger.warning(
                    f"Pod {pod_name} is not ready yet (ContainerCreating). "
                    "Wait for pod to be running before fetching logs."
                )
                return
            elif e.status == 400:
                # Otherwise, it's a different 400 error
                raise RuntimeError(
                    f"Failed to read logs for pod {self.config.namespace}/{pod_name}: {e}"
                ) from e
            raise RuntimeError(
                f"Failed to read logs for pod {self.config.namespace}/{pod_name}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to read logs for pod {self.config.namespace}/{pod_name}: {e}"
            ) from e

    def list_jobs(
        self,
        namespace: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> list[ApplicationStatus]:
        """List Spark applications.

        Args:
            namespace: Optional namespace filter (uses config namespace if None)
            labels: Optional label filters

        Returns:
            List of ApplicationStatus objects

        Raises:
            RuntimeError: If request fails
            TimeoutError: If request times out
        """
        target_namespace = namespace or self.config.namespace

        try:
            # Build label selector
            label_selector = None
            if labels:
                label_selector = ",".join([f"{k}={v}" for k, v in labels.items()])

            thread = self.custom_api.list_namespaced_custom_object(
                group=SPARK_OPERATOR_API_GROUP,
                version=SPARK_OPERATOR_API_VERSION,
                namespace=target_namespace,
                plural=SPARK_APPLICATION_PLURAL,
                label_selector=label_selector,
                async_req=True,
            )
            result = thread.get(self.config.timeout)

            applications = []
            for item in result.get("items", []):
                applications.append(self._parse_application_status(item))

            return applications

        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout listing SparkApplications in namespace {target_namespace}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list SparkApplications in namespace {target_namespace}: {e}"
            ) from e

    def wait_for_job_status(
        self,
        submission_id: str,
        timeout: int = 3600,
        polling_interval: int = 10,
    ) -> ApplicationStatus:
        """Wait for Spark application to complete.

        Args:
            submission_id: Name of the SparkApplication
            timeout: Maximum time to wait in seconds
            polling_interval: Polling interval in seconds

        Returns:
            Final ApplicationStatus

        Raises:
            TimeoutError: If application doesn't complete within timeout
            RuntimeError: If monitoring fails
        """
        start_time = time.time()

        while True:
            status = self.get_job(submission_id)

            # Check if application reached terminal state
            if status.state in [ApplicationState.COMPLETED, ApplicationState.FAILED]:
                return status

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Application {submission_id} did not complete within {timeout}s. "
                    f"Last status: {status.state.value}"
                )

            logger.debug(
                f"Application {submission_id} status: {status.state.value}. "
                f"Waiting {polling_interval}s... ({int(elapsed)}s elapsed)"
            )
            time.sleep(polling_interval)

    def _build_spark_application_crd(
        self,
        app_name: str,
        main_application_file: str,
        spark_version: str,
        app_type: str,
        driver_cores: int,
        driver_memory: str,
        executor_cores: int,
        executor_memory: str,
        num_executors: int,
        arguments: list[str],
        python_version: str,
        spark_conf: dict[str, str],
        hadoop_conf: dict[str, str],
        env_vars: dict[str, str],
        deps: Optional[dict[str, list[str]]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build SparkApplication CRD specification.

        Args:
            All parameters from submit_application
            **kwargs: Additional parameters like volumes, node_selector, etc.

        Returns:
            SparkApplication CRD dictionary
        """
        # Build base CRD structure
        spark_app: dict[str, Any] = {
            "apiVersion": f"{SPARK_OPERATOR_API_GROUP}/{SPARK_OPERATOR_API_VERSION}",
            "kind": SPARK_APPLICATION_KIND,
            "metadata": {
                "name": app_name,
                "labels": {
                    "app": app_name,
                    "version": spark_version,
                    **self.config.extra_labels,
                },
                "annotations": self.config.extra_annotations,
            },
            "spec": {
                "type": app_type,
                "mode": "cluster",
                "image": f"{self.config.default_spark_image}:{spark_version}",
                "imagePullPolicy": self.config.image_pull_policy,
                "mainApplicationFile": main_application_file,
                "sparkVersion": spark_version,
                "restartPolicy": self._build_restart_policy(kwargs.get("restart_policy")),
                "driver": {
                    "cores": driver_cores,
                    "memory": driver_memory,
                    "serviceAccount": self.config.service_account,
                    "labels": {"version": spark_version, "component": "driver"},
                },
                "executor": {
                    "cores": executor_cores,
                    "instances": num_executors,
                    "memory": executor_memory,
                    "labels": {"version": spark_version, "component": "executor"},
                },
            },
        }

        # Add optional fields
        if arguments:
            spark_app["spec"]["arguments"] = arguments

        # Add main class for Scala/Java applications
        if "main_class" in kwargs and kwargs["main_class"]:
            spark_app["spec"]["mainClass"] = kwargs["main_class"]

        if spark_conf:
            spark_app["spec"]["sparkConf"] = spark_conf

        if hadoop_conf:
            spark_app["spec"]["hadoopConf"] = hadoop_conf

        # Add environment variables
        if env_vars:
            env_list = [{"name": k, "value": v} for k, v in env_vars.items()]
            spark_app["spec"]["driver"]["env"] = env_list
            spark_app["spec"]["executor"]["env"] = env_list

        # Add dependencies
        if deps:
            spark_app["spec"]["deps"] = deps

        # Add Python version for Python apps
        if app_type == "Python":
            spark_app["spec"]["pythonVersion"] = python_version

        # Add monitoring if enabled
        if self.config.enable_monitoring:
            spark_app["spec"]["monitoring"] = {
                "exposeDriverMetrics": True,
                "exposeExecutorMetrics": True,
                "prometheus": {
                    "jmxExporterJar": "/prometheus/jmx_prometheus_javaagent-0.11.0.jar",
                    "port": 8090,
                },
            }

        # Add Spark UI service if enabled
        if self.config.enable_ui:
            spark_app["spec"]["sparkUIOptions"] = {
                "servicePort": 4040,
                "serviceType": "ClusterIP",  # Required for service creation
            }

        # Add volumes if specified
        if "volumes" in kwargs:
            spark_app["spec"]["volumes"] = kwargs["volumes"]
        if "driver_volume_mounts" in kwargs:
            spark_app["spec"]["driver"]["volumeMounts"] = kwargs["driver_volume_mounts"]
        if "executor_volume_mounts" in kwargs:
            spark_app["spec"]["executor"]["volumeMounts"] = kwargs["executor_volume_mounts"]

        # Add node selector if specified
        if "node_selector" in kwargs:
            spark_app["spec"]["driver"]["nodeSelector"] = kwargs["node_selector"]
            spark_app["spec"]["executor"]["nodeSelector"] = kwargs["node_selector"]

        # Add tolerations if specified
        if "tolerations" in kwargs:
            spark_app["spec"]["driver"]["tolerations"] = kwargs["tolerations"]
            spark_app["spec"]["executor"]["tolerations"] = kwargs["tolerations"]

        # Add resource limits if specified
        if "driver_limits" in kwargs:
            if "limits" not in spark_app["spec"]["driver"]:
                spark_app["spec"]["driver"]["limits"] = {}
            spark_app["spec"]["driver"]["limits"].update(kwargs["driver_limits"])

        if "executor_limits" in kwargs:
            if "limits" not in spark_app["spec"]["executor"]:
                spark_app["spec"]["executor"]["limits"] = {}
            spark_app["spec"]["executor"]["limits"].update(kwargs["executor_limits"])

        # Add dynamic allocation if specified
        if kwargs.get("enable_dynamic_allocation"):
            spark_app["spec"]["dynamicAllocation"] = {
                "enabled": True,
                "initialExecutors": kwargs.get("initial_executors", num_executors),
                "minExecutors": kwargs.get("min_executors", 1),
                "maxExecutors": kwargs.get("max_executors", num_executors * 2),
            }

        # Add time_to_live_seconds if specified
        if "time_to_live_seconds" in kwargs and kwargs["time_to_live_seconds"]:
            spark_app["spec"]["timeToLiveSeconds"] = kwargs["time_to_live_seconds"]

        # Add labels if specified
        if "labels" in kwargs and kwargs["labels"]:
            spark_app["metadata"]["labels"].update(kwargs["labels"])

        return spark_app

    def _build_restart_policy(self, restart_policy: Optional[Any]) -> dict[str, Any]:
        """Build restart policy dict from RestartPolicy object or default.

        Args:
            restart_policy: RestartPolicy object or None

        Returns:
            Restart policy dictionary
        """
        from kubeflow.spark.models import RestartPolicy, RestartPolicyType

        if restart_policy is None:
            return {"type": "Never"}

        # If it's already a RestartPolicy object
        if isinstance(restart_policy, RestartPolicy):
            policy_dict = {
                "type": restart_policy.type.value
                if isinstance(restart_policy.type, RestartPolicyType)
                else restart_policy.type
            }
            if restart_policy.on_failure_retries is not None:
                policy_dict["onFailureRetries"] = restart_policy.on_failure_retries
            if restart_policy.on_failure_retry_interval:
                policy_dict["onFailureRetryInterval"] = restart_policy.on_failure_retry_interval
            if restart_policy.on_submission_failure_retries is not None:
                policy_dict["onSubmissionFailureRetries"] = (
                    restart_policy.on_submission_failure_retries
                )
            if restart_policy.on_submission_failure_retry_interval:
                policy_dict["onSubmissionFailureRetryInterval"] = (
                    restart_policy.on_submission_failure_retry_interval
                )
            return policy_dict

        # If it's a string, use it as type
        if isinstance(restart_policy, str):
            return {"type": restart_policy}

        # Default
        return {"type": "Never"}

    def _parse_application_status(self, spark_app: dict[str, Any]) -> ApplicationStatus:
        """Parse SparkApplication CRD status into ApplicationStatus.

        Args:
            spark_app: SparkApplication CRD dictionary

        Returns:
            ApplicationStatus object
        """
        metadata = spark_app.get("metadata", {})
        status = spark_app.get("status", {})
        app_state_dict = status.get("applicationState", {})

        # Parse state
        state_str = app_state_dict.get("state", "UNKNOWN")
        try:
            state = ApplicationState(state_str)
        except ValueError:
            logger.warning(f"Unknown application state: {state_str}")
            state = ApplicationState.UNKNOWN

        return ApplicationStatus(
            submission_id=metadata.get("name", ""),
            app_id=status.get("sparkApplicationId"),
            app_name=metadata.get("name"),
            state=state,
            submission_time=status.get("submissionTime"),
            start_time=status.get("lastSubmissionAttemptTime"),
            completion_time=status.get("terminationTime"),
            driver_info=status.get("driverInfo"),
            executor_state=status.get("executorState"),
        )

    def _is_valid_k8s_name(self, name: str) -> bool:
        """Check if name is DNS-compliant for Kubernetes.

        Args:
            name: Name to validate

        Returns:
            True if valid, False otherwise
        """
        import re

        # Kubernetes resource names must be lowercase alphanumeric, '-' or '.'
        # and start/end with alphanumeric
        pattern = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
        return bool(re.match(pattern, name)) and len(name) <= 253

    def _get_default_namespace(self) -> str:
        """Get default Kubernetes namespace.

        Returns:
            Default namespace string
        """
        import os

        # Try to get from environment
        namespace = os.getenv("NAMESPACE")
        if namespace:
            return namespace

        # Try to read from service account
        try:
            with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
                return f.read().strip()
        except FileNotFoundError:
            pass

        # Default to "default"
        return "default"

    def _is_running_in_k8s(self) -> bool:
        """Check if running inside a Kubernetes cluster.

        Returns:
            True if running in cluster, False otherwise
        """
        import os

        return os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")

    def wait_for_pod_ready(
        self,
        submission_id: str,
        executor_id: Optional[str] = None,
        timeout: int = 300,
    ) -> bool:
        """Wait for driver or executor pod to be ready.

        Args:
            submission_id: Name of the SparkApplication
            executor_id: Optional executor ID. If None, waits for driver pod
            timeout: Maximum time to wait in seconds

        Returns:
            True if pod becomes ready, False if timeout

        Raises:
            RuntimeError: If pod check fails
        """
        # Determine pod name
        if executor_id:
            pod_name = f"{submission_id}-{executor_id}"
        else:
            pod_name = f"{submission_id}-driver"

        start_time = time.time()

        while True:
            try:
                pod = self.core_api.read_namespaced_pod(
                    name=pod_name, namespace=self.config.namespace
                )

                # Check if pod is running and container is ready
                if pod.status.phase == "Running" and pod.status.container_statuses:
                    # Check if containers are ready
                    for container_status in pod.status.container_statuses:
                        if container_status.ready:
                            logger.info(f"Pod {pod_name} is ready")
                            return True

                # Check if pod failed
                if pod.status.phase in ["Failed", "Unknown"]:
                    logger.warning(f"Pod {pod_name} is in {pod.status.phase} state")
                    return False

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning(
                        f"Timeout waiting for pod {pod_name} to be ready. "
                        f"Current phase: {pod.status.phase}"
                    )
                    return False

                # Wait before next check
                time.sleep(2)

            except client.exceptions.ApiException as e:
                if e.status == 404:
                    # Pod doesn't exist yet
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.warning(f"Timeout waiting for pod {pod_name} to be created")
                        return False
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"Failed to check pod {self.config.namespace}/{pod_name}: {e}"
                ) from e

    def close(self):
        """Close Kubernetes API client connections."""
        if hasattr(self, "custom_api") and self.custom_api.api_client:
            self.custom_api.api_client.close()
