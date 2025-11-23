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

"""Gateway backend for Spark client (REST API based)."""

from collections.abc import Iterator
import logging
import os
from typing import Any, Optional
from urllib.parse import urljoin

from kubeflow.spark.backends.base import BatchSparkBackend
from kubeflow.spark.config import AuthMethod
from kubeflow.spark.models import ApplicationStatus, SparkApplicationResponse

logger = logging.getLogger(__name__)


class GatewayBackend(BatchSparkBackend):
    """Gateway backend for Spark applications.

    This backend communicates with a Batch Processing Gateway via REST API.
    It's useful for managed Spark environments where you don't have direct
    K8s access but can use a gateway service.

    Example:
        from kubeflow.spark.backends.gateway import GatewayBackend, GatewayBackendConfig

        config = GatewayBackendConfig(
            gateway_url="http://gateway:8080",
            user="myuser",
            auth_method=AuthMethod.HEADER
        )
        backend = GatewayBackend(config)
    """

    def __init__(self, config: "GatewayBackendConfig"):
        """Initialize the Gateway backend.

        Args:
            config: GatewayBackendConfig instance
        """
        self.config = config
        self._session = None
        self._initialize_session()

    def _initialize_session(self):
        """Initialize HTTP session with authentication."""
        try:
            import requests
            from requests.auth import HTTPBasicAuth
        except ImportError:
            raise ImportError(
                "requests library is required for GatewayBackend. "
                "Install with: pip install requests"
            )

        self._session = requests.Session()
        self._session.verify = self.config.verify_ssl

        # Configure authentication
        if self.config.auth_method == AuthMethod.BASIC:
            if self.config.user and self.config.password:
                self._session.auth = HTTPBasicAuth(self.config.user, self.config.password)
        elif self.config.auth_method == AuthMethod.HEADER and self.config.user:
            self._session.headers[self.config.auth_header_key] = self.config.user

        # Add extra headers
        self._session.headers.update(self.config.extra_headers)

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
        """Submit a Spark application through the gateway.

        Args:
            See SparkBackend.submit_application for parameter details

        Returns:
            SparkApplicationResponse with submission details

        Raises:
            RuntimeError: If submission fails
            TimeoutError: If submission times out
        """
        from kubeflow.spark.models import SparkApplicationRequest

        # Build request object
        request = SparkApplicationRequest(
            app_name=app_name,
            main_application_file=main_application_file,
            spark_version=spark_version,
            app_type=app_type,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            num_executors=num_executors,
            queue=queue or self.config.default_queue,
            arguments=arguments or [],
            python_version=python_version,
            spark_conf=spark_conf or {},
            hadoop_conf=hadoop_conf or {},
            env_vars=env_vars or {},
            deps=deps,
        )

        # Submit to gateway
        url = urljoin(self.config.gateway_url, "/spark")
        try:
            response = self._session.post(url, json=request.to_dict(), timeout=self.config.timeout)
            response.raise_for_status()

            return SparkApplicationResponse.from_dict(response.json())

        except Exception as e:
            raise RuntimeError(f"Failed to submit application to gateway: {e}") from e

    def get_status(self, submission_id: str) -> ApplicationStatus:
        """Get status of a Spark application from gateway.

        Args:
            submission_id: Submission ID returned from submit_application

        Returns:
            ApplicationStatus with current status

        Raises:
            RuntimeError: If request fails
        """
        url = urljoin(self.config.gateway_url, f"/spark/{submission_id}/status")
        try:
            response = self._session.get(url, timeout=self.config.timeout)
            response.raise_for_status()

            return ApplicationStatus.from_dict(response.json())

        except Exception as e:
            raise RuntimeError(f"Failed to get status from gateway: {e}") from e

    def delete_application(self, submission_id: str) -> dict[str, Any]:
        """Delete a Spark application through gateway.

        Args:
            submission_id: Submission ID to delete

        Returns:
            Dictionary with deletion response

        Raises:
            RuntimeError: If deletion fails
        """
        url = urljoin(self.config.gateway_url, f"/spark/{submission_id}")
        try:
            response = self._session.delete(url, timeout=self.config.timeout)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            raise RuntimeError(f"Failed to delete application from gateway: {e}") from e

    def get_logs(
        self,
        submission_id: str,
        executor_id: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get application logs from gateway.

        Args:
            submission_id: Submission ID
            executor_id: Optional executor ID
            follow: Whether to stream logs (not supported by gateway)

        Yields:
            Log lines as strings

        Raises:
            RuntimeError: If request fails
        """
        if follow:
            logger.warning("Log following is not supported by GatewayBackend")

        params = {"subId": submission_id}
        if executor_id:
            params["execId"] = executor_id

        url = urljoin(self.config.gateway_url, "/log")
        try:
            response = self._session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()

            yield from response.text.splitlines()

        except Exception as e:
            raise RuntimeError(f"Failed to get logs from gateway: {e}") from e

    def list_applications(
        self,
        namespace: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> list[ApplicationStatus]:
        """List Spark applications from gateway.

        Note: Gateway backend may not support listing applications.

        Args:
            namespace: Optional namespace filter (may not be supported)
            labels: Optional label filters (may not be supported)

        Returns:
            List of ApplicationStatus objects

        Raises:
            NotImplementedError: If gateway doesn't support listing
        """
        raise NotImplementedError(
            "GatewayBackend does not support listing applications. "
            "This feature is only available with OperatorBackend."
        )

    def wait_for_completion(
        self,
        submission_id: str,
        timeout: int = 3600,
        polling_interval: int = 10,
    ) -> ApplicationStatus:
        """Wait for Spark application to complete.

        Args:
            submission_id: Submission ID to monitor
            timeout: Maximum time to wait in seconds
            polling_interval: Polling interval in seconds

        Returns:
            Final ApplicationStatus

        Raises:
            TimeoutError: If application doesn't complete within timeout
        """
        import time

        from kubeflow.spark.models import ApplicationState

        start_time = time.time()

        while True:
            status = self.get_status(submission_id)

            # Check if application reached terminal state
            if status.state in [ApplicationState.COMPLETED, ApplicationState.FAILED]:
                return status

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Application {submission_id} did not complete within {timeout}s"
                )

            logger.debug(
                f"Application {submission_id} status: {status.state.value}. "
                f"Waiting {polling_interval}s..."
            )
            time.sleep(polling_interval)

    def close(self):
        """Close HTTP session."""
        if self._session:
            self._session.close()


class GatewayBackendConfig:
    """Configuration for Gateway backend.

    Attributes:
        gateway_url: URL of the Batch Processing Gateway
        user: Username for authentication
        password: Password for basic authentication
        auth_method: Authentication method to use
        auth_header_key: Header key for user authentication
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
        default_queue: Default queue for job submission
        default_spark_version: Default Spark version
        extra_headers: Additional headers to include in requests
    """

    def __init__(
        self,
        gateway_url: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        auth_method: AuthMethod = AuthMethod.NONE,
        auth_header_key: str = "X-User",
        timeout: int = 30,
        verify_ssl: bool = True,
        default_queue: str = "poc",
        default_spark_version: str = "3.5.0",
        extra_headers: Optional[dict[str, str]] = None,
    ):
        """Initialize Gateway backend configuration.

        Args:
            gateway_url: URL of the Batch Processing Gateway
            user: Username for authentication
            password: Password for basic authentication
            auth_method: Authentication method to use
            auth_header_key: Header key for user authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            default_queue: Default queue for job submission
            default_spark_version: Default Spark version
            extra_headers: Additional headers to include
        """
        self.gateway_url = gateway_url
        self.user = user
        self.password = password
        self.auth_method = auth_method
        self.auth_header_key = auth_header_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.default_queue = default_queue
        self.default_spark_version = default_spark_version
        self.extra_headers = extra_headers or {}

    @classmethod
    def from_env(cls, prefix: str = "KUBEFLOW_SPARK_") -> "GatewayBackendConfig":
        """Create config from environment variables.

        Args:
            prefix: Prefix for environment variables

        Returns:
            GatewayBackendConfig instance

        Environment variables:
            - {prefix}GATEWAY_URL (required)
            - {prefix}USER
            - {prefix}PASSWORD
            - {prefix}AUTH_METHOD (basic|header|none)
            - {prefix}DEFAULT_QUEUE
            - {prefix}DEFAULT_SPARK_VERSION
        """
        return cls(
            gateway_url=os.getenv(f"{prefix}GATEWAY_URL", ""),
            user=os.getenv(f"{prefix}USER"),
            password=os.getenv(f"{prefix}PASSWORD"),
            auth_method=AuthMethod(os.getenv(f"{prefix}AUTH_METHOD", "none").lower()),
            timeout=int(os.getenv(f"{prefix}TIMEOUT", "30")),
            verify_ssl=os.getenv(f"{prefix}VERIFY_SSL", "true").lower() == "true",
            default_queue=os.getenv(f"{prefix}DEFAULT_QUEUE", "poc"),
            default_spark_version=os.getenv(f"{prefix}DEFAULT_SPARK_VERSION", "3.5.0"),
        )
