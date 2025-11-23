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

"""Session Spark client for interactive Spark sessions."""

from typing import Any

from kubeflow.spark.base_client import BaseSparkClient
from kubeflow.spark.backends.connect import (
    ConnectBackend,
    ConnectBackendConfig,
)
from kubeflow.spark.models import SessionInfo
from kubeflow.spark.session import ManagedSparkSession


class SparkSessionClient(BaseSparkClient):
    """Client for managing interactive Spark sessions.

    This client provides a high-level API for creating and managing long-lived
    Spark Connect sessions for interactive data analysis, exploratory workflows,
    and notebook-style development.

    Supported backends:
    - **ConnectBackend**: Connects to Spark clusters via Spark Connect protocol (gRPC)

    Features:
    - Interactive SQL queries
    - DataFrame API access
    - Artifact upload (JARs, Python files)
    - Session metrics and monitoring
    - Full PySpark API compatibility

    Example:
        ```python
        from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

        # Initialize client
        config = ConnectBackendConfig(
            connect_url="sc://spark-cluster.default.svc:15002",
            use_ssl=True,
        )
        client = SparkSessionClient(backend_config=config)

        # Create session
        session = client.create_session(app_name="data-exploration")

        # Use PySpark DataFrame API
        df = session.sql("SELECT * FROM sales WHERE date >= '2024-01-01'")
        result = df.groupBy("product").sum("amount").collect()

        # Upload artifacts
        session.upload_artifacts("/path/to/lib.jar")

        # Get metrics
        metrics = session.get_metrics()
        print(f"Queries executed: {metrics.queries_executed}")

        # Cleanup
        session.close()
        ```

    Context Manager:
        ```python
        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session("my-analysis")
            # Use session...
            # Cleanup happens automatically
        ```

    Notebook Workflow:
        ```python
        # Cell 1: Setup
        client = SparkSessionClient(backend_config=config)
        session = client.create_session("notebook-session")

        # Cell 2: Load data
        df = session.read.parquet("s3a://bucket/data/")

        # Cell 3: Analysis
        summary = df.describe()
        summary.show()

        # Cell 4: Cleanup
        session.close()
        ```
    """

    def __init__(self, backend_config: ConnectBackendConfig):
        """Initialize Spark Session client.

        Args:
            backend_config: ConnectBackendConfig with connection details

        Raises:
            ValueError: If invalid backend configuration provided
            ImportError: If pyspark[connect] is not installed
        """
        if not isinstance(backend_config, ConnectBackendConfig):
            raise ValueError(
                f"Invalid backend config type for SparkSessionClient: {type(backend_config)}. "
                "Expected ConnectBackendConfig."
            )

        # Initialize ConnectBackend
        backend = ConnectBackend(backend_config)

        # Initialize base class
        super().__init__(backend)

    def create_session(
        self,
        app_name: str,
        **kwargs: Any,
    ) -> ManagedSparkSession:
        """Create a new Spark Connect session.

        This establishes a connection to a Spark Connect server and returns
        a managed session that provides the full PySpark DataFrame API.

        Args:
            app_name: Name for the session/application
            **kwargs: Additional Spark configuration options
                     (passed to SparkSession.builder.config)

        Returns:
            ManagedSparkSession instance for interactive operations

        Raises:
            RuntimeError: If session creation fails
            ConnectionError: If cannot connect to Spark Connect server
            TimeoutError: If connection times out

        Example:
            ```python
            # Basic session
            session = client.create_session(app_name="data-analysis")

            # Session with custom configuration
            session = client.create_session(
                app_name="data-analysis",
                **{
                    "spark.sql.shuffle.partitions": "200",
                    "spark.sql.adaptive.enabled": "true",
                }
            )

            # Use session
            df = session.sql("SELECT * FROM table")
            result = df.collect()

            # Cleanup
            session.close()
            ```
        """
        return self._backend.create_session(app_name=app_name, **kwargs)

    def get_session_status(self, session_id: str) -> SessionInfo:
        """Get status and metadata of a Spark Connect session.

        Args:
            session_id: Session UUID (from session.session_id)

        Returns:
            SessionInfo with session metadata, state, and metrics

        Raises:
            RuntimeError: If request fails
            ValueError: If session_id not found

        Example:
            ```python
            # Create session
            session = client.create_session("my-app")

            # Get status
            info = client.get_session_status(session.session_id)
            print(f"Session ID: {info.session_id}")
            print(f"App name: {info.app_name}")
            print(f"State: {info.state}")
            print(f"Queries executed: {info.metrics.queries_executed}")
            print(f"Artifacts uploaded: {info.metrics.artifacts_uploaded}")
            ```
        """
        return self._backend.get_session_status(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        """List all active Spark Connect sessions.

        Returns:
            List of SessionInfo objects for active sessions

        Raises:
            RuntimeError: If request fails

        Example:
            ```python
            # List all sessions
            sessions = client.list_sessions()

            for session_info in sessions:
                print(f"Session: {session_info.session_id}")
                print(f"  App: {session_info.app_name}")
                print(f"  State: {session_info.state}")
                print(f"  Queries: {session_info.metrics.queries_executed}")
            ```
        """
        return self._backend.list_sessions()

    def close_session(self, session_id: str, release: bool = True) -> dict[str, Any]:
        """Close a Spark Connect session.

        Args:
            session_id: Session UUID to close
            release: If True, release session resources on server (default: True)

        Returns:
            Dictionary with closure response

        Raises:
            RuntimeError: If closure fails
            ValueError: If session_id not found

        Example:
            ```python
            # Create session
            session = client.create_session("my-app")
            session_id = session.session_id

            # Do work...

            # Close session and release resources
            response = client.close_session(session_id, release=True)
            print(f"Closed: {response}")

            # Alternative: use session.close() directly
            session.close()
            ```
        """
        return self._backend.close_session(session_id, release)
