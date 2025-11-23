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

"""Base backend interfaces for Spark applications.

This module defines the backend interface hierarchy for the Kubeflow Spark SDK:

- SparkBackend: Minimal base class with common functionality
- BatchSparkBackend: Interface for batch job submission (OperatorBackend, GatewayBackend)
- SessionSparkBackend: Interface for interactive sessions (ConnectBackend)

This design follows the Interface Segregation Principle (ISP), ensuring that
backends only implement methods relevant to their use case.
"""

import abc
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

from kubeflow.spark.models import ApplicationStatus, SessionInfo, SparkApplicationResponse

if TYPE_CHECKING:
    from kubeflow.spark.session import ManagedSparkSession


class SparkBackend(abc.ABC):
    """Minimal base class for all Spark backends.

    This class provides only the essential functionality common to all backends.
    Specific backend types (batch or session) inherit from BatchSparkBackend or
    SessionSparkBackend respectively.

    All backends should implement the close() method to clean up resources.
    """

    def close(self):
        """Close any open connections or resources.

        Subclasses should override this to clean up resources like:
        - Kubernetes API clients
        - HTTP connections
        - gRPC channels
        - File handles

        This method is called when the client is closed or when used as a context manager.
        """
        pass


class BatchSparkBackend(SparkBackend):
    """Abstract base class for batch-oriented Spark backends.

    This interface defines the contract for backends that support traditional
    batch Spark application submission, monitoring, and management.

    Backends implementing this interface:
    - OperatorBackend: Submits SparkApplication CRDs to Kubernetes
    - GatewayBackend: Submits jobs via REST API to Spark gateways

    Typical workflow:
        1. submit_application() -> Returns submission_id
        2. wait_for_completion() or poll get_status()
        3. get_logs() to retrieve output
        4. delete_application() for cleanup
    """

    @abc.abstractmethod
    def submit_application(
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
        queue: Optional[str],
        arguments: Optional[list[str]],
        python_version: str,
        spark_conf: Optional[dict[str, str]],
        hadoop_conf: Optional[dict[str, str]],
        env_vars: Optional[dict[str, str]],
        deps: Optional[dict[str, list[str]]],
        **kwargs: Any,
    ) -> SparkApplicationResponse:
        """Submit a Spark application for batch execution.

        Args:
            app_name: Name of the application
            main_application_file: Path to main application file (local://, s3a://, etc.)
            spark_version: Spark version to use (e.g., "4.0.0")
            app_type: Application type ("Python", "Scala", "Java", "R")
            driver_cores: Number of cores for driver
            driver_memory: Memory for driver (e.g., "4g", "512m")
            executor_cores: Number of cores per executor
            executor_memory: Memory per executor (e.g., "8g", "2g")
            num_executors: Number of executors to provision
            queue: Queue/namespace to submit to (backend-specific)
            arguments: Application arguments passed to main file
            python_version: Python version for PySpark apps (e.g., "3")
            spark_conf: Spark configuration properties (spark.*)
            hadoop_conf: Hadoop configuration properties
            env_vars: Environment variables for driver and executors
            deps: Dependencies dict with keys: "jars", "pyFiles", "files"
            **kwargs: Additional backend-specific parameters

        Returns:
            SparkApplicationResponse with submission_id and initial status

        Raises:
            RuntimeError: If submission fails
            TimeoutError: If submission times out
            ValueError: If invalid parameters provided
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_status(self, submission_id: str) -> ApplicationStatus:
        """Get current status of a Spark application.

        Args:
            submission_id: Submission ID returned from submit_application()

        Returns:
            ApplicationStatus with current state and metadata

        Raises:
            RuntimeError: If request fails
            TimeoutError: If request times out
            ValueError: If submission_id not found
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_application(self, submission_id: str) -> dict[str, Any]:
        """Delete a Spark application.

        This terminates a running application or removes a completed application.

        Args:
            submission_id: Submission ID to delete

        Returns:
            Dictionary with deletion response and status

        Raises:
            RuntimeError: If deletion fails
            TimeoutError: If deletion times out
            ValueError: If submission_id not found
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
            follow: Whether to stream logs in real-time (tail -f behavior)

        Yields:
            Log lines as strings

        Raises:
            RuntimeError: If request fails
            ValueError: If submission_id or executor_id not found
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
            TimeoutError: If request times out
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
            timeout: Maximum time to wait in seconds (default: 1 hour)
            polling_interval: Polling interval in seconds (default: 10)

        Returns:
            Final ApplicationStatus

        Raises:
            TimeoutError: If application doesn't complete within timeout
            RuntimeError: If monitoring fails
            ValueError: If submission_id not found
        """
        raise NotImplementedError()


class SessionSparkBackend(SparkBackend):
    """Abstract base class for session-oriented Spark backends.

    This interface defines the contract for backends that support interactive,
    long-lived Spark sessions for exploratory data analysis and notebook workflows.

    Backends implementing this interface:
    - ConnectBackend: Connects to Spark clusters via Spark Connect protocol (gRPC)

    Typical workflow:
        1. create_session() -> Returns ManagedSparkSession
        2. Use session.sql(), session.read(), etc. for interactive queries
        3. close_session() to release resources

    Unlike batch backends, sessions maintain state and support iterative development.
    """

    @abc.abstractmethod
    def create_session(
        self,
        app_name: str,
        **kwargs: Any,
    ) -> "ManagedSparkSession":
        """Create a new Spark Connect session.

        This establishes a connection to a Spark Connect server and returns
        a managed session wrapper that provides the full PySpark DataFrame API.

        Args:
            app_name: Name for the session/application
            **kwargs: Backend-specific configuration (e.g., Spark configs)

        Returns:
            ManagedSparkSession instance for interactive operations

        Raises:
            RuntimeError: If session creation fails
            ConnectionError: If cannot connect to Spark Connect server
            TimeoutError: If connection times out
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_session_status(self, session_id: str) -> SessionInfo:
        """Get status and metadata of a Spark Connect session.

        Args:
            session_id: Session UUID returned by create_session()

        Returns:
            SessionInfo with session metadata, state, and metrics

        Raises:
            RuntimeError: If request fails
            ValueError: If session_id not found
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list_sessions(self) -> list[SessionInfo]:
        """List all active Spark Connect sessions.

        Returns:
            List of SessionInfo objects for active sessions

        Raises:
            RuntimeError: If request fails
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def close_session(self, session_id: str, release: bool = True) -> dict[str, Any]:
        """Close a Spark Connect session.

        Args:
            session_id: Session UUID to close
            release: If True, release session resources on server

        Returns:
            Dictionary with closure response

        Raises:
            RuntimeError: If closure fails
            ValueError: If session_id not found
        """
        raise NotImplementedError()
