"""Kubeflow Spark Client for managing Spark applications on Kubernetes.

This module provides specialized Python clients for managing Apache Spark applications
on Kubernetes using different backends:

**Batch Jobs:**
- **BatchSparkClient**: For batch Spark application submission and management
  - OperatorBackend: Cloud-native using Kubeflow Spark Operator (recommended)
  - GatewayBackend: REST API for managed Spark gateways

**Interactive Sessions:**
- **SparkSessionClient**: For interactive Spark Connect sessions
  - ConnectBackend: gRPC-based remote connectivity for notebooks and exploration

Quick Start (Batch Jobs):
    ```python
    from kubeflow.spark import BatchSparkClient, OperatorBackendConfig

    # Create batch client (uses Operator backend by default)
    client = BatchSparkClient()

    # Submit a Spark application
    response = client.submit_application(
        app_name="spark-pi",
        main_application_file="local:///opt/spark/examples/src/main/python/pi.py",
        driver_cores=1,
        driver_memory="512m",
        executor_cores=1,
        executor_memory="512m",
        num_executors=2,
    )

    # Wait for completion
    status = client.wait_for_completion(response.submission_id)
    print(f"Application state: {status.state}")
    ```

Quick Start (Interactive Sessions):
    ```python
    from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

    # Connect to existing Spark cluster
    config = ConnectBackendConfig(connect_url="sc://spark-cluster:15002")
    client = SparkSessionClient(backend_config=config)

    # Create interactive session
    session = client.create_session(app_name="data-analysis")

    # Use standard PySpark API
    df = session.sql("SELECT * FROM table")
    result = df.filter(df.status == "active").collect()

    # Cleanup
    session.close()
    ```

For more examples, see the examples/ directory.
"""

# Import client classes
from kubeflow.spark.base_client import BaseSparkClient
from kubeflow.spark.batch_client import BatchSparkClient
from kubeflow.spark.session_client import SparkSessionClient

# Import backends and configs
from kubeflow.spark.backends import (
    BatchSparkBackend,
    ConnectBackend,
    ConnectBackendConfig,
    GatewayBackend,
    GatewayBackendConfig,
    OperatorBackend,
    OperatorBackendConfig,
    SessionSparkBackend,
    SparkBackend,
)

# Import models
from kubeflow.spark.models import (
    # States & Enums
    ApplicationState,
    # Status Models
    ApplicationStatus,
    BatchSchedulerConfig,
    DeployMode,
    DynamicAllocation,
    GPUSpec,
    MonitoringSpec,
    PrometheusSpec,
    # Configuration Models
    RestartPolicy,
    RestartPolicyType,
    # Session Models (for Spark Connect)
    SessionInfo,
    SessionMetrics,
    # Request & Response
    SparkApplicationRequest,
    SparkApplicationResponse,
    SparkUIConfiguration,
)

# Import session management
from kubeflow.spark.session import ManagedSparkSession

# Import validation
from kubeflow.spark.validation import (
    SparkApplicationValidator,
    ValidationError,
    ValidationErrorType,
    ValidationResult,
    validate_spark_application,
)

__all__ = [
    # Client classes
    "BaseSparkClient",
    "BatchSparkClient",
    "SparkSessionClient",
    # Backends (base classes)
    "SparkBackend",
    "BatchSparkBackend",
    "SessionSparkBackend",
    # Backend implementations
    "OperatorBackend",
    "OperatorBackendConfig",
    "GatewayBackend",
    "GatewayBackendConfig",
    "ConnectBackend",
    "ConnectBackendConfig",
    # Session Management (Spark Connect)
    "ManagedSparkSession",
    "SessionInfo",
    "SessionMetrics",
    # Request & Response Models
    "SparkApplicationRequest",
    "SparkApplicationResponse",
    "ApplicationStatus",
    # States & Enums
    "ApplicationState",
    "RestartPolicyType",
    "DeployMode",
    # Configuration Models
    "RestartPolicy",
    "GPUSpec",
    "DynamicAllocation",
    "BatchSchedulerConfig",
    "PrometheusSpec",
    "MonitoringSpec",
    "SparkUIConfiguration",
    # Validation
    "validate_spark_application",
    "SparkApplicationValidator",
    "ValidationResult",
    "ValidationError",
    "ValidationErrorType",
]

__version__ = "0.2.0"
