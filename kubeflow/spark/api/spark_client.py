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

"""SparkClient for Kubeflow SDK."""

from collections.abc import Iterator
import warnings

from pyspark.sql import SparkSession

from kubeflow.common.types import KubernetesBackendConfig
import kubeflow.common.utils as common_utils
from kubeflow.spark.backends.kubernetes import KubernetesBackend
from kubeflow.spark.backends.kubernetes.utils import validate_spark_connect_url
from kubeflow.spark.types.types import (
    Driver,
    Executor,
    ExistingSession,
    FileJob,
    FuncJob,
    NewSession,
    SparkConnectInfo,
    SparkJob,
    SparkJobStatus,
)


class SparkClient:
    """Stateless Spark client for Kubeflow."""

    def __init__(self, backend_config: KubernetesBackendConfig | None = None):
        """Initialize the Spark client.

        Args:
            backend_config: Kubernetes backend configuration. If not provided, the default
                configuration is used.

        Raises:
            ValueError:
                If backend_config is not a KubernetesBackendConfig instance.
        """
        if backend_config is None:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config: {type(backend_config)}")

    # ------------------------------------------------------------------
    # Spark Connect sessions
    # ------------------------------------------------------------------

    def connect(
        self,
        base_url: str | None = None,
        token: str | None = None,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
        timeout: int = 300,
        connect_timeout: int = 120,
        *,
        session: ExistingSession | NewSession | None = None,
    ) -> SparkSession:
        """Connect to or create a SparkConnect session.

        Preferred usage uses typed session modes, symmetric with batch job modes:

        - ``session=ExistingSession(...)``: connect to an existing Spark Connect server
        - ``session=NewSession(...)``: create a new Spark Connect session

        ``Driver`` and ``Executor`` should be passed through ``options``.

        Legacy keyword arguments remain supported for compatibility and emit
        deprecation warnings when used without ``session=``.

        Args:
            base_url: Optional URL to existing Spark Connect server
                (e.g., "sc://server:15002"). Deprecated in favor of
                ``session=ExistingSession(base_url=...)``.
            token: Optional authentication token for existing server. Deprecated in
                favor of ``session=ExistingSession(..., token=...)``.
            num_executors: Number of executor instances (create mode only).
                Deprecated in favor of ``session=NewSession(num_executors=...)``.
            resources_per_executor: Resource requirements per executor as dict.
                Format: `{"cpu": "5", "memory": "10Gi"}` (create mode only).
                Deprecated in favor of ``session=NewSession(...)``.
            spark_conf: Spark configuration dictionary (create mode only).
                Deprecated in favor of ``session=NewSession(spark_conf=...)``.
            driver: Driver configuration object (create mode only). Deprecated in
                favor of ``options=[Driver(...)]``.
            executor: Executor configuration object (create mode only). Deprecated in
                favor of ``options=[Executor(...)]``.
            options: List of configuration options (create mode only).
                Use Name, Driver, Executor, and other options here.
            timeout: Timeout in seconds to wait for session ready.
            connect_timeout: Timeout in seconds for SparkSession.getOrCreate()
                (create mode only).
            session: Typed connection mode. Use ``ExistingSession`` or ``NewSession``.

        Returns:
            SparkSession connected to Spark (self-managing).

        Raises:
            ValueError: If base_url is invalid or the provided resource configuration is invalid.
            TimeoutError: If creating a Spark Connect session or connecting to it times out.
            RuntimeError: If the Spark Connect session cannot be created or connected to.

        Note:
            Server port defaults to 15002 (Spark Connect gRPC). PySpark and server Spark
            major.minor should match; see constants and pyproject.toml [spark].

        Examples:
            # Connect to existing server
            spark = client.connect(session=ExistingSession(base_url="sc://server:15002"))

            # Create with simple parameters
            spark = client.connect(
                session=NewSession(
                    num_executors=5,
                    resources_per_executor={"cpu": "5", "memory": "10Gi"},
                    spark_conf={"spark.sql.adaptive.enabled": "true"},
                )
            )

            # Create with Driver/Executor options
            spark = client.connect(
                session=NewSession(),
                options=[
                    Driver(resources={"cpu": "2", "memory": "4Gi"}),
                    Executor(
                        num_instances=5,
                        resources_per_executor={"cpu": "4", "memory": "8Gi"},
                    ),
                ],
            )

            # Minimal - use all defaults (auto-generated name)
            spark = client.connect()
        """
        (
            resolved_base_url,
            resolved_token,
            resolved_num_executors,
            resolved_resources_per_executor,
            resolved_spark_conf,
            resolved_options,
        ) = self._resolve_connect_args(
            session=session,
            options=options,
            base_url=base_url,
            token=token,
            num_executors=num_executors,
            resources_per_executor=resources_per_executor,
            spark_conf=spark_conf,
            driver=driver,
            executor=executor,
        )

        if resolved_base_url:
            validate_spark_connect_url(resolved_base_url)
            builder = SparkSession.builder.remote(resolved_base_url)
            if resolved_token:
                builder = builder.config("spark.connect.authenticate.token", resolved_token)
            return builder.getOrCreate()

        return self.backend.create_and_connect(
            num_executors=resolved_num_executors,
            resources_per_executor=resolved_resources_per_executor,
            spark_conf=resolved_spark_conf,
            driver=None,
            executor=None,
            options=resolved_options,
            timeout=timeout,
            connect_timeout=connect_timeout,
        )

    def _resolve_connect_args(
        self,
        *,
        session: ExistingSession | NewSession | None,
        options: list | None,
        base_url: str | None,
        token: str | None,
        num_executors: int | None,
        resources_per_executor: dict[str, str] | None,
        spark_conf: dict[str, str] | None,
        driver: Driver | None,
        executor: Executor | None,
    ) -> tuple[
        str | None,
        str | None,
        int | None,
        dict[str, str] | None,
        dict[str, str] | None,
        list | None,
    ]:
        """Normalize typed session modes and legacy connect kwargs."""
        legacy_connect = base_url is not None or token is not None
        legacy_create = any(
            value is not None
            for value in (
                num_executors,
                resources_per_executor,
                spark_conf,
                driver,
                executor,
            )
        )

        if session is not None and (legacy_connect or legacy_create):
            raise ValueError(
                "Pass either `session=` or legacy connect keyword arguments, not both."
            )

        if legacy_connect and not base_url:
            raise ValueError(
                "`base_url` must be a non-empty string to connect to an existing Spark "
                "Connect server. `token` alone is not sufficient."
            )

        resolved_options: list = list(options) if options is not None else []

        if isinstance(session, ExistingSession):
            if not session.base_url:
                raise ValueError("`ExistingSession.base_url` must be a non-empty string.")
            return session.base_url, session.token, None, None, None, None

        if isinstance(session, NewSession):
            return (
                None,
                None,
                session.num_executors,
                session.resources_per_executor,
                session.spark_conf,
                resolved_options or None,
            )

        if session is not None:
            raise TypeError(
                "`session` must be an instance of ExistingSession or NewSession, "
                f"got {type(session).__name__}."
            )

        if legacy_connect or legacy_create:
            warnings.warn(
                "Passing connect configuration as keyword arguments is deprecated. "
                "Use session=ExistingSession(...) or session=NewSession(...), and pass "
                "Driver/Executor through options.",
                DeprecationWarning,
                stacklevel=3,
            )

        if driver is not None:
            resolved_options.insert(0, driver)
        if executor is not None:
            resolved_options.insert(0, executor)

        return (
            base_url,
            token,
            num_executors,
            resources_per_executor,
            spark_conf,
            resolved_options or None,
        )

    def list_sessions(self) -> list[SparkConnectInfo]:
        """List SparkConnect sessions.

        Returns:
            List of SparkConnectInfo objects.
        """
        return self.backend.list_sessions()

    def get_session(self, name: str) -> SparkConnectInfo:
        """Get information about a SparkConnect session.

        Args:
            name:
                Name of the SparkConnect session.

        Returns:
            SparkConnectInfo containing information about the SparkConnect session.
        """
        return self.backend.get_session(name)

    def delete_session(self, name: str) -> None:
        """Delete a SparkConnect session.

        Args:
            name:
                Name of the SparkConnect session to delete.
        """
        self.backend.delete_session(name)

    def get_session_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        """Get logs from a SparkConnect session.

        Args:
            name:
                Name of the SparkConnect session.

            follow:
                Whether to stream logs continuously.

        Returns:
            Iterator of log lines from the SparkConnect driver pod.
        """
        return self.backend.get_session_logs(name, follow=follow)

    # ------------------------------------------------------------------
    # Spark batch jobs
    # ------------------------------------------------------------------

    def submit_job(
        self,
        job: FileJob | FuncJob,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        options: list | None = None,
    ) -> str:
        """Submit a batch Spark job.

        This method supports two job types:

        - **FileJob**: Submit a Spark application referenced by a local
        or remote file source.
        - **FuncJob**: Submit a Python function as a Spark batch job.

        Args:
            job:
                Job definition describing the workload to execute.
                Supports either ``FileJob`` or ``FuncJob``.

            num_executors:
                Number of executor instances.

            resources_per_executor:
                Resource requirements per executor.
                Format: ``{"cpu": "5", "memory": "10Gi"}``.

            spark_conf:
                Spark configuration dictionary.

            options:
                List of additional Spark configuration options.

        Raises:
            ValueError:
                If unsupported Phase 1 features are requested or the job definition is invalid.

            TypeError:
                If the job type is invalid.

            NotImplementedError:
                If unsupported features are requested.
        """
        if spark_conf is not None:
            raise NotImplementedError("spark_conf support is not yet implemented.")

        if options is not None:
            raise NotImplementedError("options are not supported in Phase 1.")

        return self.backend.submit_job(
            job=job,
            num_executors=num_executors,
            resources_per_executor=resources_per_executor,
        ).name

    def get_job(self, name: str) -> SparkJob:
        """Get information about a Spark job.

        Args:
            name:
                Name of the Spark job.

        Returns:
            SparkJob containing information about the Spark job.
        """

        return self.backend.get_job(name)

    def list_jobs(
        self,
        status: set[SparkJobStatus] | None = None,
    ) -> list[SparkJob]:
        """List Spark jobs.

        Args:
            status:
                Optional set of job statuses to filter the returned jobs.

        Returns:
            List of SparkJob objects.
        """
        return self.backend.list_jobs(status=status)

    def delete_job(self, name: str) -> None:
        """Delete a Spark job.

        Args:
            name:
                Name of the Spark job to delete.
        """
        self.backend.delete_job(name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[SparkJobStatus] = {SparkJobStatus.COMPLETED},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> SparkJob:
        """Wait for a Spark job to reach one of the target states.
        Args:
            name: Job name.
            status: Target job state(s).
            timeout: Maximum wait time in seconds. Default 600.
            polling_interval: Time in seconds between status checks.

        Returns:
            Spark job information after reaching one of the target statuses.

        Raises:
            ValueError:
                If the polling interval or timeout values are invalid.
        """
        common_utils.validate_wait_for_job_status(
            polling_interval,
            timeout,
        )

        return self.backend.wait_for_job_status(
            name=name,
            status=status,
            timeout=timeout,
            polling_interval=polling_interval,
        )

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a Spark job.

        Args:
            name: Spark job name.
            follow: Whether to stream logs in realtime.

        Returns:
            Iterator of log lines.
        """
        return self.backend.get_job_logs(
            name=name,
            follow=follow,
        )
