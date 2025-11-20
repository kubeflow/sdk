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

"""Managed Spark Connect session wrapper."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from kubeflow.spark.models import SessionInfo, SessionMetrics

if TYPE_CHECKING:
    from kubeflow.spark.backends.connect import ConnectBackend

    # Only import pyspark types for type checking
    try:
        from pyspark.sql import DataFrame, DataFrameReader, SparkSession
        from pyspark.sql.streaming import DataStreamReader
    except ImportError:
        DataFrame = Any  # type: ignore
        DataFrameReader = Any  # type: ignore
        SparkSession = Any  # type: ignore
        DataStreamReader = Any  # type: ignore

logger = logging.getLogger(__name__)


class ManagedSparkSession:
    """Kubeflow-managed Spark Connect session.

    This class wraps a native PySpark Connect session and provides additional
    Kubeflow-specific functionality like metrics collection, artifact management,
    and pipeline integration.

    The underlying PySpark DataFrame API is accessible directly, allowing users
    to write standard PySpark code while benefiting from Kubeflow enhancements.

    Example:
        ```python
        from kubeflow.spark import SparkClient, ConnectBackendConfig

        config = ConnectBackendConfig(connect_url="sc://spark-cluster:15002")
        client = SparkClient(backend_config=config)

        # Create session
        session = client.create_session(app_name="data-analysis")

        # Use standard PySpark API
        df = session.sql("SELECT * FROM table")
        result = df.filter(df.status == "active").collect()

        # Kubeflow extensions
        metrics = session.get_metrics()
        session.export_to_pipeline_artifact(df, "/outputs/data.parquet")

        # Cleanup
        session.close()
        ```
    """

    def __init__(
        self,
        session: "SparkSession",
        session_id: str,
        app_name: str,
        backend: "ConnectBackend",
    ):
        """Initialize managed session.

        Args:
            session: Native PySpark Connect session
            session_id: Session UUID
            app_name: Application name
            backend: ConnectBackend instance for lifecycle management
        """
        self._session = session
        self._session_id = session_id
        self._app_name = app_name
        self._backend = backend
        self._closed = False
        self._metrics = SessionMetrics(session_id=session_id)

        logger.info(f"Created ManagedSparkSession: {session_id} (app: {app_name})")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def session_id(self) -> str:
        """Get session UUID."""
        return self._session_id

    @property
    def app_name(self) -> str:
        """Get application name."""
        return self._app_name

    @property
    def is_closed(self) -> bool:
        """Check if session is closed."""
        return self._closed

    @property
    def spark(self) -> "SparkSession":
        """Get underlying PySpark session.

        Use this to access the full PySpark API directly.
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")
        return self._session

    # =========================================================================
    # Delegate PySpark DataFrame API
    # =========================================================================

    def sql(self, query: str) -> "DataFrame":
        """Execute SQL query and return DataFrame.

        Args:
            query: SQL query string

        Returns:
            DataFrame with query results
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")
        self._metrics.queries_executed += 1
        return self._session.sql(query)

    @property
    def read(self) -> "DataFrameReader":
        """Get DataFrameReader for reading data sources."""
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")
        return self._session.read

    @property
    def readStream(self) -> "DataStreamReader":
        """Get DataStreamReader for reading streaming sources."""
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")
        return self._session.readStream

    def createDataFrame(self, data: Any, schema: Any = None) -> "DataFrame":
        """Create DataFrame from data.

        Args:
            data: Input data (list, pandas DataFrame, RDD, etc.)
            schema: Optional schema

        Returns:
            DataFrame
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")
        return self._session.createDataFrame(data, schema)

    def table(self, tableName: str) -> "DataFrame":
        """Get DataFrame for a table.

        Args:
            tableName: Name of the table

        Returns:
            DataFrame
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")
        return self._session.table(tableName)

    def range(
        self,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        numPartitions: Optional[int] = None,
    ) -> "DataFrame":
        """Create DataFrame with range of numbers.

        Args:
            start: Start of range (or end if `end` not provided)
            end: End of range (optional)
            step: Step size
            numPartitions: Number of partitions

        Returns:
            DataFrame
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")
        if end is None:
            return self._session.range(start)
        return self._session.range(start, end, step, numPartitions)

    # =========================================================================
    # Kubeflow Extensions
    # =========================================================================

    def upload_artifacts(self, *paths: str, pyfile: bool = False) -> None:
        """Upload artifacts to Spark Connect session.

        Args:
            *paths: File paths to upload (JARs, Python files, data files)
            pyfile: If True, treat as Python files (added to sys.path)

        Example:
            ```python
            # Upload JARs
            session.upload_artifacts("/path/to/lib.jar")

            # Upload Python packages
            session.upload_artifacts("/path/to/package.zip", pyfile=True)
            ```
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")

        try:
            # Use PySpark's addArtifacts method
            if hasattr(self._session, "addArtifacts"):
                self._session.addArtifacts(*paths, pyfile=pyfile)
                self._metrics.artifacts_uploaded += len(paths)
                logger.info(f"Uploaded {len(paths)} artifacts to session {self._session_id}")
            else:
                logger.warning("Session does not support artifact upload (requires PySpark 3.4+)")
        except Exception as e:
            logger.error(f"Failed to upload artifacts: {e}")
            raise

    def get_metrics(self) -> SessionMetrics:
        """Get session metrics.

        Returns:
            SessionMetrics with current statistics
        """
        return self._metrics

    def get_info(self) -> SessionInfo:
        """Get session information.

        Returns:
            SessionInfo with session metadata
        """
        return SessionInfo(
            session_id=self._session_id,
            app_name=self._app_name,
            state="closed" if self._closed else "active",
            metrics=self._metrics,
        )

    def export_to_pipeline_artifact(
        self, df: "DataFrame", path: str, format: str = "parquet", **options: Any
    ) -> None:
        """Export DataFrame to Kubeflow Pipeline artifact.

        Args:
            df: DataFrame to export
            path: Output path for artifact
            format: Output format (parquet, csv, json, etc.)
            **options: Additional write options

        Example:
            ```python
            df = session.sql("SELECT * FROM sales")
            session.export_to_pipeline_artifact(df, "/outputs/sales.parquet")
            ```
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")

        try:
            writer = df.write.format(format)
            for key, value in options.items():
                writer = writer.option(key, value)
            writer.save(path)
            logger.info(f"Exported DataFrame to {path} (format: {format})")
        except Exception as e:
            logger.error(f"Failed to export DataFrame: {e}")
            raise

    def clone(self) -> "ManagedSparkSession":
        """Clone the session with all state.

        Creates a new session that shares the same state (temp views, UDFs, etc.)
        but has its own session ID.

        Returns:
            New ManagedSparkSession instance
        """
        if self._closed:
            raise RuntimeError(f"Session {self._session_id} is closed")

        logger.info(f"Cloning session {self._session_id}")
        return self._backend._clone_session(self)

    def close(self, release: bool = True) -> None:
        """Close the session.

        Args:
            release: If True, release session resources on server
        """
        if self._closed:
            logger.warning(f"Session {self._session_id} already closed")
            return

        try:
            if release:
                # Stop the session
                self._session.stop()
                logger.info(f"Released session {self._session_id} on server")
            self._closed = True
        except Exception as e:
            logger.error(f"Error closing session {self._session_id}: {e}")
            raise

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "ManagedSparkSession":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "active"
        return f"ManagedSparkSession(id={self._session_id}, app={self._app_name}, status={status})"
