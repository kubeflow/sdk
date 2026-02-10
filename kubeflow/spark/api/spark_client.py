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
import logging
from typing import Optional

from pyspark.sql import SparkSession

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.kubernetes import KubernetesBackend
from kubeflow.spark.backends.kubernetes.utils import validate_spark_connect_url
from kubeflow.spark.types.types import Driver, Executor, SparkConnectInfo

logger = logging.getLogger(__name__)


class SparkClient:
    """Stateless Spark client for Kubeflow."""

    def __init__(self, backend_config: Optional[KubernetesBackendConfig] = None):
        """Initialize SparkClient."""
        if backend_config is None:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config: {type(backend_config)}")

    def connect(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        num_executors: Optional[int] = None,
        resources_per_executor: Optional[dict[str, str]] = None,
        spark_conf: Optional[dict[str, str]] = None,
        driver: Optional[Driver] = None,
        executor: Optional[Executor] = None,
        options: Optional[list] = None,
        timeout: int = 300,
        connect_timeout: int = 120,
    ) -> SparkSession:
        """Connect to or create a SparkConnect session (KEP-107 lines 298-347).

        This method supports two modes based on parameters:
        - **Connect mode**: When `base_url` is provided, connects to an existing Spark Connect server
        - **Create mode**: When `base_url` is not provided, creates a new Spark Connect session

        Args:
            base_url: Optional URL to existing Spark Connect server (e.g., "sc://server:15002").
                 If provided, connects to existing server. If None, creates new session.
            token: Optional authentication token for existing server.
            num_executors: Number of executor instances (create mode only).
            resources_per_executor: Resource requirements per executor as dict.
                Format: `{"cpu": "5", "memory": "10Gi"}` (create mode only).
            spark_conf: Spark configuration dictionary (create mode only).
            driver: Driver configuration object (create mode only).
            executor: Executor configuration object (create mode only).
            options: List of configuration options (create mode only).
                Use Name option for custom session name.
            timeout: Timeout in seconds to wait for session ready.
            connect_timeout: Timeout in seconds for SparkSession.getOrCreate() (create mode only).

        Returns:
            SparkSession connected to Spark (self-managing).

        Examples:
            # Connect to existing server
            spark = client.connect(base_url="sc://server:15002")

            # Create with simple parameters
            spark = client.connect(
                num_executors=5,
                resources_per_executor={"cpu": "5", "memory": "10Gi"},
                spark_conf={"spark.sql.adaptive.enabled": "true"}
            )

            # Create with custom name
            from kubeflow.spark.types.options import Name
            spark = client.connect(options=[Name("my-session")])

            # Create with advanced configuration
            spark = client.connect(
                driver=Driver(resources={"cpu": "2", "memory": "4Gi"}),
                executor=Executor(
                    num_instances=5,
                    resources_per_executor={"cpu": "4", "memory": "8Gi"}
                )
            )

            # Minimal - use all defaults (auto-generated name)
            spark = client.connect()

        Note:
            Server port defaults to 15002 (Spark Connect gRPC). PySpark and server Spark
            major.minor should match; see constants and pyproject.toml [spark].
        """
        if base_url:
            validate_spark_connect_url(base_url)
            builder = SparkSession.builder.remote(base_url)
            if token:
                builder = builder.config("spark.connect.authenticate.token", token)
            return builder.getOrCreate()

        return self.backend.create_and_connect(
            num_executors=num_executors,
            resources_per_executor=resources_per_executor,
            spark_conf=spark_conf,
            driver=driver,
            executor=executor,
            options=options,
            timeout=timeout,
            connect_timeout=connect_timeout,
        )

    def list_sessions(self) -> list[SparkConnectInfo]:
        """List all SparkConnect sessions."""
        return self.backend.list_sessions()

    def get_session(self, name: str) -> SparkConnectInfo:
        """Get session info by name."""
        return self.backend.get_session(name)

    def delete_session(self, name: str) -> None:
        """Delete a SparkConnect session."""
        self.backend.delete_session(name)

    def get_session_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        """Get logs from a session."""
        return self.backend.get_session_logs(name, follow=follow)
