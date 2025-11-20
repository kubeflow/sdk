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

"""Batch Spark client for managing Spark applications."""

from collections.abc import Iterator
from typing import Any, Optional, Union

from kubeflow.spark.base_client import BaseSparkClient
from kubeflow.spark.backends.base import BatchSparkBackend
from kubeflow.spark.backends.gateway import (
    GatewayBackend,
    GatewayBackendConfig,
)
from kubeflow.spark.backends.operator import (
    OperatorBackend,
    OperatorBackendConfig,
)
from kubeflow.spark.models import (
    ApplicationStatus,
    SparkApplicationResponse,
)


class BatchSparkClient(BaseSparkClient):
    """Client for managing batch Spark applications.

    This client provides a high-level API for submitting and managing batch
    Spark applications using either the Kubernetes Spark Operator or REST gateways.

    Supported backends:
    - **OperatorBackend**: Submits SparkApplication CRDs to Kubernetes (recommended)
    - **GatewayBackend**: Submits jobs via REST API to Spark gateways (Livy, etc.)

    Example with Operator Backend:
        ```python
        from kubeflow.spark import BatchSparkClient, OperatorBackendConfig

        # Initialize client
        config = OperatorBackendConfig(namespace="spark-jobs")
        client = BatchSparkClient(backend_config=config)

        # Submit application
        response = client.submit_application(
            app_name="my-etl-job",
            main_application_file="s3a://bucket/jobs/etl.py",
            driver_cores=2,
            driver_memory="4g",
            executor_cores=4,
            executor_memory="8g",
            num_executors=10,
        )

        # Wait for completion
        status = client.wait_for_completion(response.submission_id)
        print(f"Job completed with state: {status.state}")

        # Get logs
        for line in client.get_logs(response.submission_id):
            print(line)
        ```

    Example with Gateway Backend:
        ```python
        from kubeflow.spark import BatchSparkClient, GatewayBackendConfig

        config = GatewayBackendConfig(
            gateway_url="http://livy-gateway:8998",
            user="myuser"
        )
        client = BatchSparkClient(backend_config=config)
        ```

    Context Manager:
        ```python
        with BatchSparkClient(backend_config=config) as client:
            response = client.submit_application(...)
            # Cleanup happens automatically
        ```
    """

    def __init__(
        self,
        backend_config: Union[OperatorBackendConfig, GatewayBackendConfig, None] = None,
    ):
        """Initialize Batch Spark client.

        Args:
            backend_config: Backend configuration:
                          - OperatorBackendConfig: Kubernetes with Spark Operator (default)
                          - GatewayBackendConfig: REST API gateway

        Raises:
            ValueError: If invalid backend configuration provided
        """
        # Default to OperatorBackend
        if backend_config is None:
            backend_config = OperatorBackendConfig()

        # Initialize appropriate backend
        if isinstance(backend_config, OperatorBackendConfig):
            backend: BatchSparkBackend = OperatorBackend(backend_config)
        elif isinstance(backend_config, GatewayBackendConfig):
            backend = GatewayBackend(backend_config)
        else:
            raise ValueError(
                f"Invalid backend config type for BatchSparkClient: {type(backend_config)}. "
                "Expected OperatorBackendConfig or GatewayBackendConfig."
            )

        # Initialize base class
        super().__init__(backend)

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
        """Submit a Spark application for batch execution.

        Args:
            app_name: Name of the application (must be unique)
            main_application_file: Path to main application file
                                  Supported formats: local://, s3a://, http://, etc.
            spark_version: Spark version (default: "3.5.0")
            app_type: Application type: "Python", "Scala", "Java", "R" (default: "Python")
            driver_cores: Number of CPU cores for driver (default: 1)
            driver_memory: Memory for driver, e.g., "1g", "512m" (default: "1g")
            executor_cores: Number of CPU cores per executor (default: 1)
            executor_memory: Memory per executor, e.g., "1g", "2g" (default: "1g")
            num_executors: Number of executors (default: 2)
            queue: Queue/namespace for submission (backend-specific, optional)
            arguments: Command-line arguments for the main file (optional)
            python_version: Python version for PySpark: "2" or "3" (default: "3")
            spark_conf: Spark configuration properties (spark.*), optional
            hadoop_conf: Hadoop configuration properties, optional
            env_vars: Environment variables for driver and executors, optional
            deps: Dependencies dict with keys: "jars", "pyFiles", "files", optional
            **kwargs: Additional backend-specific parameters (e.g., volumes, GPUs)

        Returns:
            SparkApplicationResponse with submission_id and initial status

        Raises:
            RuntimeError: If submission fails
            TimeoutError: If submission times out
            ValueError: If invalid parameters provided

        Example:
            ```python
            response = client.submit_application(
                app_name="data-processing",
                main_application_file="s3a://my-bucket/jobs/process.py",
                driver_cores=2,
                driver_memory="4g",
                executor_cores=4,
                executor_memory="8g",
                num_executors=10,
                spark_conf={
                    "spark.sql.shuffle.partitions": "200",
                    "spark.hadoop.fs.s3a.endpoint": "s3.amazonaws.com",
                },
                arguments=["--input", "s3a://data/input", "--output", "s3a://data/output"],
            )
            print(f"Submitted: {response.submission_id}")
            ```
        """
        return self._backend.submit_application(
            app_name=app_name,
            main_application_file=main_application_file,
            spark_version=spark_version,
            app_type=app_type,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            num_executors=num_executors,
            queue=queue,
            arguments=arguments,
            python_version=python_version,
            spark_conf=spark_conf,
            hadoop_conf=hadoop_conf,
            env_vars=env_vars,
            deps=deps,
            **kwargs,
        )

    def get_status(self, submission_id: str) -> ApplicationStatus:
        """Get current status of a Spark application.

        Args:
            submission_id: Submission ID returned from submit_application()

        Returns:
            ApplicationStatus with current state, timestamps, and metadata

        Raises:
            RuntimeError: If request fails
            ValueError: If submission_id not found

        Example:
            ```python
            status = client.get_status("spark-pi-12345")
            print(f"State: {status.state}")
            print(f"App ID: {status.app_id}")
            ```
        """
        return self._backend.get_status(submission_id)

    def delete_application(self, submission_id: str) -> dict[str, Any]:
        """Delete a Spark application.

        This terminates a running application or removes a completed one.

        Args:
            submission_id: Submission ID to delete

        Returns:
            Dictionary with deletion response

        Raises:
            RuntimeError: If deletion fails
            ValueError: If submission_id not found

        Example:
            ```python
            response = client.delete_application("spark-pi-12345")
            print(f"Deleted: {response}")
            ```
        """
        return self._backend.delete_application(submission_id)

    def get_logs(
        self,
        submission_id: str,
        executor_id: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get application logs.

        Args:
            submission_id: Submission ID
            executor_id: Optional executor ID (if not provided, returns driver logs)
            follow: If True, stream logs in real-time (tail -f behavior)

        Yields:
            Log lines as strings

        Raises:
            RuntimeError: If request fails
            ValueError: If submission_id or executor_id not found

        Example:
            ```python
            # Get driver logs
            for line in client.get_logs("spark-pi-12345"):
                print(line)

            # Get specific executor logs
            for line in client.get_logs("spark-pi-12345", executor_id="1"):
                print(line)

            # Stream logs in real-time
            for line in client.get_logs("spark-pi-12345", follow=True):
                print(line)
            ```
        """
        return self._backend.get_logs(submission_id, executor_id, follow)

    def list_applications(
        self,
        namespace: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> list[ApplicationStatus]:
        """List Spark applications with optional filtering.

        Args:
            namespace: Optional namespace/queue filter
            labels: Optional label filters (key-value pairs)

        Returns:
            List of ApplicationStatus objects

        Raises:
            RuntimeError: If request fails

        Example:
            ```python
            # List all applications
            apps = client.list_applications()

            # List in specific namespace
            apps = client.list_applications(namespace="production")

            # Filter by labels
            apps = client.list_applications(labels={"team": "data-eng"})
            ```
        """
        return self._backend.list_applications(namespace, labels)

    def wait_for_completion(
        self,
        submission_id: str,
        timeout: int = 3600,
        polling_interval: int = 10,
    ) -> ApplicationStatus:
        """Wait for Spark application to complete.

        This method blocks until the application reaches a terminal state
        (COMPLETED, FAILED, SUBMISSION_FAILED, KILLED) or timeout is reached.

        Args:
            submission_id: Submission ID to monitor
            timeout: Maximum time to wait in seconds (default: 3600 = 1 hour)
            polling_interval: Polling interval in seconds (default: 10)

        Returns:
            Final ApplicationStatus

        Raises:
            TimeoutError: If application doesn't complete within timeout
            RuntimeError: If monitoring fails
            ValueError: If submission_id not found

        Example:
            ```python
            # Wait with defaults (1 hour timeout)
            status = client.wait_for_completion("spark-pi-12345")

            # Custom timeout and polling
            status = client.wait_for_completion(
                "spark-pi-12345",
                timeout=1800,  # 30 minutes
                polling_interval=5,  # Poll every 5 seconds
            )

            if status.state == ApplicationState.COMPLETED:
                print("Success!")
            else:
                print(f"Failed with state: {status.state}")
            ```
        """
        return self._backend.wait_for_completion(submission_id, timeout, polling_interval)

    def wait_for_pod_ready(
        self,
        submission_id: str,
        executor_id: Optional[str] = None,
        timeout: int = 300,
    ) -> bool:
        """Wait for driver or executor pod to be ready.

        Note: This method is only available when using OperatorBackend.

        Args:
            submission_id: Submission ID
            executor_id: Optional executor ID (if not provided, waits for driver)
            timeout: Maximum time to wait in seconds (default: 300 = 5 minutes)

        Returns:
            True if pod becomes ready, False if timeout

        Raises:
            NotImplementedError: If backend doesn't support this operation
            RuntimeError: If request fails

        Example:
            ```python
            # Wait for driver pod
            if client.wait_for_pod_ready("spark-pi-12345"):
                print("Driver is ready")

            # Wait for specific executor
            if client.wait_for_pod_ready("spark-pi-12345", executor_id="1"):
                print("Executor 1 is ready")
            ```
        """
        if isinstance(self._backend, OperatorBackend):
            return self._backend.wait_for_pod_ready(submission_id, executor_id, timeout)
        else:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support wait_for_pod_ready(). "
                "This method is only available with OperatorBackend."
            )
