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

"""Base class for Spark backends."""

import abc
from collections.abc import Iterator

from kubeflow.spark.types.types import Driver, Executor, SparkConnectInfo


class RuntimeBackend(abc.ABC):
    """Abstract base class for Spark backends.

    All Spark backends must implement these methods to manage SparkConnect sessions.
    """

    @abc.abstractmethod
    def connect(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
    ) -> SparkConnectInfo:
        """Create a new SparkConnect session (INTERNAL USE ONLY).

        This is an internal method used by SparkClient.connect().
        Use SparkClient.connect() instead of calling this directly.

        Args:
            num_executors: Number of executor instances.
            resources_per_executor: Resource requirements per executor.
            spark_conf: Spark configuration properties.
            driver: Driver configuration.
            executor: Executor configuration.
            options: List of configuration options (use Name option for custom name).

        Returns:
            SparkConnectInfo with session details (may be in PROVISIONING state).

        Raises:
            TimeoutError: If the creation request times out.
            RuntimeError: If session creation fails.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_session(self, name: str) -> SparkConnectInfo:
        """Get information about a SparkConnect session.

        Args:
            name: Session name.

        Returns:
            SparkConnectInfo with session details.

        Raises:
            TimeoutError: If the request times out.
            RuntimeError: If the session is not found or request fails.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list_sessions(self) -> list[SparkConnectInfo]:
        """List all SparkConnect sessions.

        Returns:
            List of SparkConnectInfo objects.

        Raises:
            TimeoutError: If the request times out.
            RuntimeError: If listing fails.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_session(self, name: str) -> None:
        """Delete a SparkConnect session.

        Args:
            name: Session name.

        Raises:
            TimeoutError: If the deletion request times out.
            RuntimeError: If the session is not found or deletion fails.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _wait_for_session_ready(
        self,
        name: str,
        timeout: int = 300,
        polling_interval: int = 2,
    ) -> SparkConnectInfo:
        """Wait for a SparkConnect session to become ready (INTERNAL USE ONLY).

        This is an internal method used by SparkClient.connect().
        Use SparkClient.connect() instead of calling this directly.

        Polls the session status until it reaches READY state or times out.

        Args:
            name: Session name.
            timeout: Maximum wait time in seconds. Default 300 (5 minutes).
            polling_interval: Seconds between status checks. Default 2.

        Returns:
            SparkConnectInfo when session is ready.

        Raises:
            TimeoutError: If session does not become ready within timeout.
            RuntimeError: If session fails.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_session_logs(
        self,
        name: str,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a SparkConnect session.

        Args:
            name: Session name.
            follow: If True, stream logs continuously.

        Returns:
            Iterator of log lines.

        Raises:
            TimeoutError: If reading logs times out.
            RuntimeError: If the session/pod is not found or reading fails.
        """
        raise NotImplementedError()
