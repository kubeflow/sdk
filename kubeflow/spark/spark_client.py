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

"""Unified SparkClient with auto-provisioning support.

This module provides a simplified, unified interface for Spark on Kubernetes
that combines both interactive sessions (via Spark Connect) and batch jobs
(via SparkApplication CRD) with optional auto-provisioning of Spark Connect servers.

Example - Auto-provisioning (creates Spark Connect server automatically):
    ```python
    from kubeflow.spark import SparkClient

    # Creates Spark Connect server on Kubernetes automatically
    with SparkClient.builder().num_executors(5).build() as client:
        spark = client.session()
        df = spark.read.parquet("s3a://data/sales/")
        df.show()
    # Server is automatically deleted on exit
    ```

Example - Connect to existing server:
    ```python
    from kubeflow.spark import SparkClient

    client = SparkClient.connect("sc://spark-cluster:15002")
    spark = client.session()
    df = spark.sql("SELECT * FROM table")
    ```

Example - Submit batch job:
    ```python
    from kubeflow.spark import SparkClient

    client = SparkClient.builder().namespace("etl-jobs").build()
    job = client.submit_batch(
        main_file="s3a://bucket/etl.py",
        arguments=["--date", "2024-01-15"],
    )
    client.wait_for_job_status(job.submission_id)
    ```
"""

import logging
from collections.abc import Iterator
from typing import Any, Optional

from kubeflow.spark.backends.connect import ConnectBackend, ConnectBackendConfig
from kubeflow.spark.backends.operator import OperatorBackend, OperatorBackendConfig
from kubeflow.spark.models import (
    ApplicationStatus,
    DynamicAllocation,
    SparkApplicationResponse,
    SparkConnectServerConfig,
    SparkConnectServerStatus,
    SparkConnectState,
)
from kubeflow.spark.session import ManagedSparkSession

logger = logging.getLogger(__name__)


class SparkClientBuilder:
    """Builder for SparkClient configuration.

    Provides a fluent API for configuring SparkClient with resource settings,
    Spark configuration, and auto-provisioning options.

    Example:
        ```python
        client = (
            SparkClient.builder()
            .namespace("spark-jobs")
            .server(cores=2, memory="4g")
            .executor(cores=4, memory="8g")
            .num_executors(10)
            .spark_conf("spark.sql.shuffle.partitions", "200")
            .build()
        )
        ```
    """

    def __init__(self) -> None:
        """Initialize SparkClientBuilder with default values."""
        # Kubernetes settings
        self._namespace: str = "default"
        self._service_account: str = "spark-operator-spark"

        # Server (driver) resources
        self._server_cores: int = 1
        self._server_memory: str = "1g"

        # Executor resources
        self._executor_cores: int = 1
        self._executor_memory: str = "1g"
        self._num_executors: int = 2

        # Dynamic allocation
        self._dynamic_allocation: Optional[DynamicAllocation] = None

        # Image settings
        self._image: Optional[str] = None
        self._spark_version: str = "3.5.0"

        # Spark configuration
        self._spark_conf: dict[str, str] = {}
        self._hadoop_conf: dict[str, str] = {}

        # Connection settings
        self._connect_url: Optional[str] = None

        # Lifecycle
        self._cleanup_on_exit: bool = True
        self._timeout: int = 300

        # Kubernetes config
        self._config_file: Optional[str] = None
        self._context: Optional[str] = None

    def namespace(self, ns: str) -> "SparkClientBuilder":
        """Set Kubernetes namespace.

        Args:
            ns: Kubernetes namespace name.

        Returns:
            Self for method chaining.
        """
        self._namespace = ns
        return self

    def service_account(self, sa: str) -> "SparkClientBuilder":
        """Set service account for Spark pods.

        Args:
            sa: Service account name.

        Returns:
            Self for method chaining.
        """
        self._service_account = sa
        return self

    def server(self, cores: int = 1, memory: str = "1g") -> "SparkClientBuilder":
        """Configure server (driver) resources.

        Args:
            cores: Number of CPU cores for the server.
            memory: Memory allocation (e.g., '1g', '2g').

        Returns:
            Self for method chaining.
        """
        self._server_cores = cores
        self._server_memory = memory
        return self

    def executor(self, cores: int = 1, memory: str = "1g") -> "SparkClientBuilder":
        """Configure executor resources.

        Args:
            cores: Number of CPU cores per executor.
            memory: Memory allocation per executor.

        Returns:
            Self for method chaining.
        """
        self._executor_cores = cores
        self._executor_memory = memory
        return self

    def num_executors(self, n: int) -> "SparkClientBuilder":
        """Set number of executors.

        Args:
            n: Number of executor instances.

        Returns:
            Self for method chaining.
        """
        self._num_executors = n
        return self

    def dynamic_allocation(
        self,
        enabled: bool = True,
        min_executors: Optional[int] = None,
        max_executors: Optional[int] = None,
        initial_executors: Optional[int] = None,
    ) -> "SparkClientBuilder":
        """Configure dynamic executor allocation.

        Args:
            enabled: Whether to enable dynamic allocation.
            min_executors: Minimum number of executors.
            max_executors: Maximum number of executors.
            initial_executors: Initial number of executors.

        Returns:
            Self for method chaining.
        """
        self._dynamic_allocation = DynamicAllocation(
            enabled=enabled,
            min_executors=min_executors,
            max_executors=max_executors,
            initial_executors=initial_executors,
        )
        return self

    def image(self, image: str) -> "SparkClientBuilder":
        """Set custom Spark image.

        Args:
            image: Container image name with tag.

        Returns:
            Self for method chaining.
        """
        self._image = image
        return self

    def spark_version(self, version: str) -> "SparkClientBuilder":
        """Set Spark version.

        Args:
            version: Spark version (e.g., '3.5.0').

        Returns:
            Self for method chaining.
        """
        self._spark_version = version
        return self

    def spark_conf(self, key: str, value: str) -> "SparkClientBuilder":
        """Add a Spark configuration property.

        Args:
            key: Configuration key (e.g., 'spark.sql.shuffle.partitions').
            value: Configuration value.

        Returns:
            Self for method chaining.
        """
        self._spark_conf[key] = value
        return self

    def spark_confs(self, conf: dict[str, str]) -> "SparkClientBuilder":
        """Add multiple Spark configuration properties.

        Args:
            conf: Dictionary of configuration key-value pairs.

        Returns:
            Self for method chaining.
        """
        self._spark_conf.update(conf)
        return self

    def hadoop_conf(self, key: str, value: str) -> "SparkClientBuilder":
        """Add a Hadoop configuration property.

        Args:
            key: Configuration key.
            value: Configuration value.

        Returns:
            Self for method chaining.
        """
        self._hadoop_conf[key] = value
        return self

    def connect_url(self, url: str) -> "SparkClientBuilder":
        """Set existing Spark Connect server URL.

        If set, the client will connect to the existing server instead of
        auto-provisioning a new one.

        Args:
            url: Spark Connect URL (e.g., 'sc://host:15002').

        Returns:
            Self for method chaining.
        """
        self._connect_url = url
        return self

    def cleanup_on_exit(self, cleanup: bool) -> "SparkClientBuilder":
        """Set whether to delete resources on exit.

        Args:
            cleanup: Whether to cleanup resources on exit.

        Returns:
            Self for method chaining.
        """
        self._cleanup_on_exit = cleanup
        return self

    def timeout(self, seconds: int) -> "SparkClientBuilder":
        """Set timeout for server provisioning.

        Args:
            seconds: Timeout in seconds.

        Returns:
            Self for method chaining.
        """
        self._timeout = seconds
        return self

    def kube_config(
        self, config_file: Optional[str] = None, context: Optional[str] = None
    ) -> "SparkClientBuilder":
        """Set Kubernetes configuration.

        Args:
            config_file: Path to kubeconfig file.
            context: Kubernetes context name.

        Returns:
            Self for method chaining.
        """
        self._config_file = config_file
        self._context = context
        return self

    def build(self) -> "SparkClient":
        """Create SparkClient with configured settings.

        Returns:
            Configured SparkClient instance.
        """
        return SparkClient(self)

    def get_or_create(self) -> ManagedSparkSession:
        """Build client and immediately return SparkSession.

        Creates a SparkClient and returns the SparkSession, useful for
        quick one-liner session creation.

        Returns:
            ManagedSparkSession connected to the Spark Connect server.
        """
        client = self.build()
        return client.session()


class SparkClient:
    """Unified Spark client for Kubeflow with auto-provisioning.

    Supports both interactive sessions (via Spark Connect) and batch jobs
    (via SparkApplication CRD). Can auto-provision Spark Connect servers
    on Kubernetes or connect to existing servers.

    Example (Auto-provisioning):
        ```python
        with SparkClient.builder().num_executors(5).build() as client:
            spark = client.session()
            df = spark.read.parquet("s3a://data/")
            df.show()
        ```

    Example (Existing server):
        ```python
        client = SparkClient.connect("sc://spark-cluster:15002")
        spark = client.session()
        ```

    Example (Batch job):
        ```python
        client = SparkClient.builder().namespace("etl").build()
        job = client.submit_batch(main_file="s3a://bucket/etl.py")
        client.wait_for_job_status(job.submission_id)
        ```
    """

    def __init__(self, builder: SparkClientBuilder) -> None:
        """Initialize SparkClient from builder.

        Args:
            builder: Configured SparkClientBuilder.
        """
        self._builder = builder
        self._operator_backend: Optional[OperatorBackend] = None
        self._connect_backend: Optional[ConnectBackend] = None
        self._server_status: Optional[SparkConnectServerStatus] = None
        self._managed_session: Optional[ManagedSparkSession] = None
        self._provisioned_server: bool = False

        # Initialize operator backend for auto-provisioning and batch jobs
        operator_config = OperatorBackendConfig(
            namespace=builder._namespace,
            config_file=builder._config_file,
            context=builder._context,
            service_account=builder._service_account,
        )
        self._operator_backend = OperatorBackend(operator_config)

    @classmethod
    def builder(cls) -> SparkClientBuilder:
        """Create a builder for configuring SparkClient.

        Returns:
            New SparkClientBuilder instance.
        """
        return SparkClientBuilder()

    @classmethod
    def connect(
        cls,
        url: str,
        token: Optional[str] = None,
        use_ssl: bool = False,
    ) -> "SparkClient":
        """Connect to an existing Spark Connect server.

        Args:
            url: Spark Connect URL (e.g., 'sc://host:port').
            token: Authentication token.
            use_ssl: Whether to use SSL.

        Returns:
            SparkClient connected to the existing server.
        """
        return cls.builder().connect_url(url).build()

    def session(self, app_name: Optional[str] = None) -> ManagedSparkSession:
        """Get or create SparkSession.

        If no connect_url was provided, this will auto-provision a
        Spark Connect server on Kubernetes first.

        Args:
            app_name: Optional application name.

        Returns:
            ManagedSparkSession for interactive operations.

        Raises:
            RuntimeError: Failed to create session.
        """
        if self._managed_session is not None:
            return self._managed_session

        # Determine connect URL
        connect_url = self._builder._connect_url

        if connect_url is None:
            # Auto-provision Spark Connect server
            self._provision_connect_server()
            connect_url = self._server_status.connect_url
            self._provisioned_server = True

        # Create ConnectBackend
        connect_config = ConnectBackendConfig(connect_url=connect_url)
        self._connect_backend = ConnectBackend(connect_config)

        # Create session
        session_name = app_name or "spark-client-session"
        self._managed_session = self._connect_backend.create_session(
            app_name=session_name, **self._builder._spark_conf
        )

        return self._managed_session

    @property
    def spark(self) -> ManagedSparkSession:
        """Shortcut to session().

        Returns:
            ManagedSparkSession.
        """
        return self.session()

    def _provision_connect_server(self) -> None:
        """Auto-provision Spark Connect server."""
        logger.info("Auto-provisioning Spark Connect server...")

        # Build server config
        server_config = SparkConnectServerConfig(
            namespace=self._builder._namespace,
            spark_version=self._builder._spark_version,
            image=self._builder._image,
            server_cores=self._builder._server_cores,
            server_memory=self._builder._server_memory,
            executor_cores=self._builder._executor_cores,
            executor_memory=self._builder._executor_memory,
            num_executors=self._builder._num_executors,
            dynamic_allocation=self._builder._dynamic_allocation,
            spark_conf=self._builder._spark_conf,
            hadoop_conf=self._builder._hadoop_conf,
            service_account=self._builder._service_account,
        )

        # Create server
        self._server_status = self._operator_backend.create_connect_server(server_config)

        # Wait for server to be ready
        self._server_status = self._operator_backend.wait_for_connect_server_ready(
            self._server_status.name,
            namespace=self._builder._namespace,
            timeout=self._builder._timeout,
        )

        logger.info(f"Spark Connect server ready at: {self._server_status.connect_url}")

    # =========================================================================
    # Batch Job Methods
    # =========================================================================

    def submit_batch(
        self,
        main_file: str,
        main_class: Optional[str] = None,
        arguments: Optional[list[str]] = None,
        app_name: Optional[str] = None,
        **kwargs: Any,
    ) -> SparkApplicationResponse:
        """Submit a batch Spark job.

        Args:
            main_file: Path to the main application file.
            main_class: Main class for Java/Scala applications.
            arguments: Arguments to pass to the application.
            app_name: Name for the job (auto-generated if not provided).
            **kwargs: Additional parameters for SparkApplication.

        Returns:
            SparkApplicationResponse with submission details.

        Raises:
            RuntimeError: Failed to submit job.
        """
        return self._operator_backend.submit_application(
            app_name=app_name,
            main_application_file=main_file,
            main_class=main_class,
            spark_version=self._builder._spark_version,
            driver_cores=self._builder._server_cores,
            driver_memory=self._builder._server_memory,
            executor_cores=self._builder._executor_cores,
            executor_memory=self._builder._executor_memory,
            num_executors=self._builder._num_executors,
            arguments=arguments,
            spark_conf=self._builder._spark_conf or None,
            hadoop_conf=self._builder._hadoop_conf or None,
            **kwargs,
        )

    def get_job(self, submission_id: str) -> ApplicationStatus:
        """Get status of a batch job.

        Args:
            submission_id: Submission ID from submit_batch().

        Returns:
            ApplicationStatus with current state.
        """
        return self._operator_backend.get_job(submission_id)

    def get_job_logs(
        self,
        submission_id: str,
        executor_id: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a Spark job.

        Args:
            submission_id: Submission ID from submit_batch().
            executor_id: Optional executor ID (e.g., "1", "2").
                If None, returns driver logs.
            follow: Whether to stream logs in realtime.

        Returns:
            Iterator of log lines.
        """
        return self._operator_backend.get_job_logs(submission_id, executor_id, follow)

    def wait_for_job_status(
        self,
        submission_id: str,
        timeout: int = 3600,
        polling_interval: int = 10,
    ) -> ApplicationStatus:
        """Wait for a batch job to complete.

        Args:
            submission_id: Submission ID from submit_batch().
            timeout: Maximum time to wait in seconds.
            polling_interval: Interval between status checks.

        Returns:
            Final ApplicationStatus.
        """
        return self._operator_backend.wait_for_job_status(
            submission_id, timeout, polling_interval
        )

    def delete_job(self, submission_id: str) -> dict[str, Any]:
        """Delete a batch job.

        Args:
            submission_id: Submission ID from submit_batch().

        Returns:
            Deletion response.
        """
        return self._operator_backend.delete_job(submission_id)

    def list_jobs(
        self,
        namespace: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> list[ApplicationStatus]:
        """List batch jobs.

        Args:
            namespace: Optional namespace filter.
            labels: Optional label filters.

        Returns:
            List of ApplicationStatus objects.
        """
        return self._operator_backend.list_jobs(namespace, labels)

    # =========================================================================
    # Server Status
    # =========================================================================

    def server_status(self) -> Optional[SparkConnectServerStatus]:
        """Get Spark Connect server status.

        Returns:
            SparkConnectServerStatus if server was auto-provisioned, None otherwise.
        """
        if self._server_status is None:
            return None
        return self._operator_backend.get_connect_server_status(
            self._server_status.name,
            namespace=self._builder._namespace,
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def stop(self) -> None:
        """Stop session and cleanup resources."""
        # Close session
        if self._managed_session is not None:
            try:
                self._managed_session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            self._managed_session = None

        # Close connect backend
        if self._connect_backend is not None:
            try:
                self._connect_backend.close()
            except Exception as e:
                logger.warning(f"Error closing connect backend: {e}")
            self._connect_backend = None

        # Delete auto-provisioned server
        if self._provisioned_server and self._builder._cleanup_on_exit:
            if self._server_status is not None:
                try:
                    self._operator_backend.delete_connect_server(
                        self._server_status.name,
                        namespace=self._builder._namespace,
                    )
                    logger.info(
                        f"Deleted auto-provisioned server: {self._server_status.name}"
                    )
                except Exception as e:
                    logger.warning(f"Error deleting server: {e}")
            self._server_status = None
            self._provisioned_server = False

        # Close operator backend
        if self._operator_backend is not None:
            try:
                self._operator_backend.close()
            except Exception as e:
                logger.warning(f"Error closing operator backend: {e}")
            self._operator_backend = None

    def __enter__(self) -> "SparkClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()

