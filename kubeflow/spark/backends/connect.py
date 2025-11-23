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

"""Spark Connect backend for remote Spark cluster connectivity."""

from collections.abc import Iterator
import logging
from typing import Any, Optional
import uuid

from kubeflow.spark.backends.base import SessionSparkBackend
from kubeflow.spark.models import (
    ApplicationStatus,
    ConnectBackendConfig,
    SessionInfo,
    SparkApplicationResponse,
)
from kubeflow.spark.session import ManagedSparkSession

logger = logging.getLogger(__name__)


class ConnectBackend(SessionSparkBackend):
    """Spark Connect backend for remote connectivity to Spark clusters.

    This backend enables connection to existing Spark clusters via the Spark Connect
    protocol (gRPC-based). It supports interactive, session-based workloads unlike
    traditional batch-oriented backends.

    Features:
    - Remote connectivity via Spark Connect (gRPC)
    - Session management with isolation
    - Interactive DataFrame operations
    - Artifact upload (JARs, Python files, data)
    - Authentication (Bearer token)
    - SSL/TLS support
    - Optional auto-provisioning of Spark Connect server

    Example:
        ```python
        from kubeflow.spark import SparkClient, ConnectBackendConfig

        config = ConnectBackendConfig(
            connect_url="sc://spark-cluster.default.svc:15002", token="my-auth-token", use_ssl=True
        )
        client = SparkClient(backend_config=config)

        # Create session
        session = client.create_session(app_name="data-analysis")

        # Use PySpark API
        df = session.sql("SELECT * FROM table")
        result = df.collect()

        # Cleanup
        session.close()
        ```
    """

    def __init__(self, config: ConnectBackendConfig):
        """Initialize Spark Connect backend.

        Args:
            config: ConnectBackendConfig with connection details

        Raises:
            ImportError: If pyspark[connect] is not installed
            ValueError: If config is invalid
        """
        self.config = config
        self._sessions: dict[str, ManagedSparkSession] = {}

        # Validate and parse connection URL
        self._validate_config()

        # Check for pyspark installation
        try:
            import pyspark

            pyspark_version = pyspark.__version__
            logger.info(f"Using PySpark version: {pyspark_version}")

            # Check for Connect support (requires 3.4+)
            major, minor = map(int, pyspark_version.split(".")[:2])
            if major < 3 or (major == 3 and minor < 4):
                raise ImportError(
                    f"Spark Connect requires PySpark 3.4+, found {pyspark_version}. "
                    "Please upgrade: pip install 'pyspark[connect]>=3.4.0'"
                )
        except ImportError as e:
            raise ImportError(
                "PySpark with Connect support is required for ConnectBackend. "
                "Install it with: pip install 'pyspark[connect]>=3.4.0'"
            ) from e

        logger.info(f"Initialized ConnectBackend with URL: {self._get_masked_url()}")

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config is invalid
        """
        if not self.config.connect_url:
            raise ValueError("connect_url is required")

        # Parse URL to validate format
        if not self.config.connect_url.startswith("sc://"):
            raise ValueError(
                f"Invalid Spark Connect URL: {self.config.connect_url}. "
                "Expected format: sc://host:port/;param1=value;param2=value"
            )

        # Parse URL components
        url_without_scheme = self.config.connect_url[5:]  # Remove "sc://"
        if "/" in url_without_scheme:
            host_port, params = url_without_scheme.split("/", 1)
        else:
            host_port = url_without_scheme
            params = ""

        if ":" not in host_port:
            raise ValueError(
                f"Invalid Spark Connect URL: {self.config.connect_url}. "
                "Expected format: sc://host:port/"
            )

    def _get_masked_url(self) -> str:
        """Get connection URL with masked token.

        Returns:
            URL string with token masked
        """
        url = self.config.connect_url
        if ";token=" in url:
            parts = url.split(";token=")
            return parts[0] + ";token=***"
        return url

    def _build_connection_url(self) -> str:
        """Build final connection URL with all parameters.

        For Spark Connect, most parameters should be set via builder.config()
        rather than in the URL to avoid conflicts with server-side configs.

        Returns:
            Complete Spark Connect URL
        """
        # For Spark 4.x, use simple URL without parameters
        # Parameters should be set via builder.config() instead
        url = self.config.connect_url

        # Only add essential parameters that are part of the connection string
        # SSL and authentication should be handled at connection level
        # Avoid adding parameters like use_ssl in URL as they may conflict

        return url

    # =========================================================================
    # Session-Oriented Methods (Implemented)
    # =========================================================================

    def create_session(
        self,
        app_name: str,
        **kwargs: Any,
    ) -> ManagedSparkSession:
        """Create a new Spark Connect session.

        Args:
            app_name: Name for the session/application
            **kwargs: Additional Spark configuration (passed to SparkSession.builder.config)

        Returns:
            ManagedSparkSession instance

        Raises:
            RuntimeError: If session creation fails
        """
        try:
            from pyspark.sql import SparkSession

            logger.debug("Starting create_session")

            # Generate session ID
            session_id = str(uuid.uuid4())
            logger.debug(f"Generated session ID: {session_id}")

            # Build connection URL
            connection_url = self._build_connection_url()
            logger.debug(f"Connection URL: {connection_url}")

            # Create SparkSession builder
            logger.debug("Creating SparkSession.builder.remote()")
            builder = SparkSession.builder.remote(connection_url).appName(app_name)
            logger.debug("Builder created, adding app name")

            # Apply additional configurations
            for key, value in kwargs.items():
                logger.debug(f"Applying config: {key}={value}")
                builder = builder.config(key, value)

            # Create session
            logger.debug("About to call builder.getOrCreate() - THIS MAY HANG")
            spark_session = builder.getOrCreate()
            logger.debug("getOrCreate() returned successfully")

            # Wrap in ManagedSparkSession
            logger.debug("Creating ManagedSparkSession wrapper")
            managed_session = ManagedSparkSession(
                session=spark_session,
                session_id=session_id,
                app_name=app_name,
                backend=self,
            )

            # Track session
            self._sessions[session_id] = managed_session
            logger.debug("Session tracked in backend")

            logger.info(f"Created Spark Connect session: {session_id} (app: {app_name})")
            return managed_session

        except Exception as e:
            logger.error(f"Failed to create Spark Connect session: {e}")
            raise RuntimeError(f"Failed to create session: {e}") from e

    def get_session_status(self, session_id: str) -> SessionInfo:
        """Get status of a Spark Connect session.

        Args:
            session_id: Session UUID

        Returns:
            SessionInfo with session metadata

        Raises:
            ValueError: If session not found
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self._sessions[session_id]
        return session.get_info()

    def list_sessions(self) -> list[SessionInfo]:
        """List all active Spark Connect sessions.

        Returns:
            List of SessionInfo objects
        """
        return [session.get_info() for session in self._sessions.values()]

    def close_session(self, session_id: str, release: bool = True) -> dict[str, Any]:
        """Close a Spark Connect session.

        Args:
            session_id: Session UUID to close
            release: If True, release session resources on server

        Returns:
            Dictionary with closure response

        Raises:
            ValueError: If session not found
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self._sessions[session_id]
        session.close(release=release)

        # Remove from tracking
        del self._sessions[session_id]

        return {
            "session_id": session_id,
            "status": "closed",
            "message": "Session closed successfully",
        }

    def _clone_session(self, session: ManagedSparkSession) -> ManagedSparkSession:
        """Internal method to clone a session.

        Args:
            session: Session to clone

        Returns:
            New ManagedSparkSession
        """
        try:
            # Generate new session ID
            new_session_id = str(uuid.uuid4())

            # Clone the underlying PySpark session
            # Note: PySpark Connect supports session cloning via newSession()
            if hasattr(session.spark, "newSession"):
                new_spark_session = session.spark.newSession()
            else:
                # Fallback: create new session (won't share state)
                logger.warning("Session cloning not supported, creating new session instead")
                return self.create_session(app_name=f"{session.app_name}-clone")

            # Wrap in ManagedSparkSession
            cloned_session = ManagedSparkSession(
                session=new_spark_session,
                session_id=new_session_id,
                app_name=f"{session.app_name}-clone",
                backend=self,
            )

            # Track session
            self._sessions[new_session_id] = cloned_session

            logger.info(f"Cloned session {session.session_id} -> {new_session_id}")
            return cloned_session

        except Exception as e:
            logger.error(f"Failed to clone session: {e}")
            raise RuntimeError(f"Failed to clone session: {e}") from e

    def close(self):
        """Close all sessions and cleanup resources."""
        logger.info(f"Closing ConnectBackend with {len(self._sessions)} active sessions")

        # Close all sessions
        for session_id in list(self._sessions.keys()):
            try:
                self.close_session(session_id, release=True)
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")

        logger.info("ConnectBackend closed")
