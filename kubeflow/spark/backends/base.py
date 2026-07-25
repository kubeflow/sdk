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

from pyspark.sql import SparkSession

from kubeflow.spark.types.types import (
    Driver,
    Executor,
    FileJob,
    FuncJob,
    SparkConnectInfo,
    SparkJob,
    SparkJobStatus,
)


class RuntimeBackend(abc.ABC):
    """Abstract base class for Spark backends.

    All Spark backends must implement these methods to manage SparkConnect sessions and Spark batch jobs.
    """

    # ------------------------------------------------------------------
    # Spark Connect sessions
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def create_and_connect(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
        timeout: int = 300,
        connect_timeout: int = 120,
    ) -> SparkSession:
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
            timeout: Maximum time in seconds to wait for the session to become ready.
            connect_timeout: Maximum time in seconds to establish the Spark Connect session.

        Returns:
            Connected SparkSession.

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

    # ------------------------------------------------------------------
    # Spark batch jobs
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def submit_job(
        self,
        job: FileJob | FuncJob,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
    ) -> SparkJob:
        """Submit a Spark batch job.

        Args:
            job: Spark application to execute.
            num_executors: Number of executor instances.
            resources_per_executor: Resource requirements for each executor.

        Returns:
            Submitted Spark job information.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job(
        self,
        name: str,
    ) -> SparkJob:
        """Get a Spark job.

        Args:
            name: Name of the Spark job.

        Returns:
            Spark job information.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list_jobs(
        self,
        status: set[SparkJobStatus] | None = None,
    ) -> list[SparkJob]:
        """List Spark jobs.

        Args:
            status: Optional set of Spark job statuses used to filter results.

        Returns:
            List of Spark jobs.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_job(
        self,
        name: str,
    ) -> None:
        """Delete a Spark job.

        Args:
            name: Name of the Spark job to delete.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_job_status(
        self,
        name: str,
        status: set[SparkJobStatus] | None = None,
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> SparkJob:
        """Wait for a Spark job to reach one of the desired statuses.

        Args:
            name: Name of the Spark job.
            status: Optional set of target Spark job statuses.
                Defaults to ``{SparkJobStatus.COMPLETED}``.
            timeout: Maximum time in seconds to wait.
            polling_interval: Time in seconds between status checks.

        Returns:
            Spark job information after reaching one of the desired statuses.

        Raises:
            TimeoutError: If the timeout is exceeded before reaching the target status.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a Spark job.

        Args:
            name: Name of the Spark job.
            follow: Whether to stream logs in real time.

        Returns:
            Iterator over log lines.
        """
        raise NotImplementedError()
