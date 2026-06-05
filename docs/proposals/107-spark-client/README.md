# KEP-107: Spark Client SDK for Kubeflow

## Authors

- Shekhar Rajak - [@shekharrajak](https://github.com/shekharrajak)

Ref: https://github.com/kubeflow/sdk/issues/107

## Summary

A simple Python SDK to run Spark on Kubernetes. The SDK provides `SparkClient`:

- **`connect()` API** - Creates new Spark Connect sessions or connects to existing servers
- **Auto-provisions** Spark Connect servers when configuration is provided
- **Connects** to existing Spark Connect servers when URL is provided
- **Auto-cleans up** resources on exit
- **Batch job support** - submit and manage SparkApplication jobs via `submit_job()`

## Motivation

Running Spark on Kubernetes requires managing complex infrastructure. Users want to focus on their Spark code, not:

- Creating SparkApplication CRDs
- Managing Spark Connect servers
- Writing YAML configurations
- Handling cleanup

## Goals

1. Simple Python API for Spark on Kubernetes
2. Auto-provision Spark Connect servers
3. Support connecting to existing servers
4. Full PySpark compatibility
5. Extensible architecture for future batch job support
6. Submit and manage batch Spark jobs via SparkApplication CRD
7. Kubeflow ecosystem integration (Pipelines, Trainer, Spark History MCP Server)

## Non-Goals

- Supporting Spark outside Kubernetes (local mode, standalone clusters)
- Managing Spark Operator installation
- Replacing the Spark Operator

---

## API

### Basic Usage

```python
from kubeflow.spark import SparkClient

client = SparkClient()
spark = client.connect(
    num_executors=5,
    resources_per_executor={
        "cpu": "5",
        "memory": "10Gi"
    },
    spark_conf={
        "spark.sql.adaptive.enabled": "true"
    }
)
df = spark.sql("SELECT * FROM my_table")
df.show()
spark.stop()
```

### With Namespace Configuration

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="spark-jobs"))
spark = client.connect(
    num_executors=10,
    resources_per_executor={
        "cpu": "4",
        "memory": "16Gi"
    },
    spark_conf={"spark.sql.adaptive.enabled": "true"}
)
spark.stop()
```

### Connect to Existing Server

```python
from kubeflow.spark import SparkClient

client = SparkClient()
spark = client.connect(url="sc://spark-server:15002")
df = spark.read.parquet("s3a://bucket/data/")
df.show()
spark.stop()
```

### Minimal Usage - Default Configuration

```python
from kubeflow.spark import SparkClient

client = SparkClient()
spark = client.connect()
df = spark.sql("SELECT * FROM my_table")
df.show()
spark.stop()
```

### Advanced Configuration

- **Resource configuration**: Dictionary-based resources (e.g., `{"cpu": "5", "memory": "10Gi"}`)
- **S3/Storage integration**: SeaweedFS, MinIO, and AWS S3 support via `s3_config()` and `seaweedfs_config()`
- **Spark configuration**: Any Spark config via `spark_conf()` and `spark_confs()`
- **Kubernetes features**: Advanced configs via `options` parameter (Labels, Annotations, PodTemplateOverrides)

---

## Unified `connect()` API

The `connect()` method provides a unified interface for both creating new Spark Connect sessions and connecting to existing servers. The method automatically determines the mode based on the parameters provided:

- **Create Mode**: When `url` is not provided, creates a new Spark Connect session with the specified configuration
- **Connect Mode**: When `url` is provided, connects to an existing Spark Connect server

This simplification reduces API surface area and makes the SDK easier to use:

```python
spark = client.connect(
    num_executors=5,
    resources_per_executor={
        "cpu": "5",
        "memory": "10Gi"
    },
    spark_conf={"spark.sql.adaptive.enabled": "true"}
)

spark = client.connect(url="sc://server:15002")

spark = client.connect()
```

---

## SparkClient API

### Resource Configuration

Resources are specified as dictionaries:

```python
resources_per_executor = {
    "cpu": "5",
    "memory": "10Gi",
}
```

### Structured Types

```python
from kubeflow.spark import Driver, Executor

@dataclass
class Driver:
    """Driver configuration for Spark Connect session."""
    image: Optional[str] = None
    resources: Optional[Dict[str, str]] = None
    java_options: Optional[str] = None
    service_account: Optional[str] = None

@dataclass
class Executor:
    """Executor configuration for Spark Connect session."""
    num_instances: Optional[int] = None
    resources_per_executor: Optional[Dict[str, str]] = None
    java_options: Optional[str] = None
```

### Options Pattern

Advanced Kubernetes configurations are provided via `options` parameter:

```python
from kubeflow.spark.options import Labels, Annotations, PodTemplateOverrides

options = [
    Labels({"app": "spark"}),
    Annotations({"description": "ETL job"}),
    PodTemplateOverrides(...)
]
```

```python
class SparkClientBuilder:
    """Builder for advanced SparkClient configuration."""

    def backend(self, config: Union[KubernetesBackendConfig, GatewayBackendConfig]) -> "SparkClientBuilder":
        """Set backend configuration (namespace, context, etc.)."""

    def service_account(self, sa: str) -> "SparkClientBuilder":
        """Set service account for Spark pods."""

    def memory_overhead_factor(self, factor: float) -> "SparkClientBuilder":
        """Set global memory overhead factor (default: 0.1 for JVM, 0.4 for non-JVM)."""

    def image(self, image: str) -> "SparkClientBuilder":
        """Set custom Spark image."""

    def spark_version(self, version: str) -> "SparkClientBuilder":
        """Set Spark version (default: 3.5.0)."""

    def spark_confs(self, conf: Dict[str, str]) -> "SparkClientBuilder":
        """Add multiple Spark configuration properties.

        Args:
            conf: Dictionary of Spark configuration key-value pairs
        """

    def spark_config_profile(self, profile: Union[str, Dict[str, str]]) -> "SparkClientBuilder":
        """Apply a Spark configuration profile.

        Profiles are predefined sets of Spark configurations that can be:
        - Built-in profiles: "seaweedfs", "aws-s3", "minio", "optimized"
        - Custom profiles: Pass a dict of key-value pairs
        - File-based: Pass a path to a YAML/JSON file (future)

        Profiles can be merged/overridden by subsequent spark_conf() calls.

        Args:
            profile: Profile name (str) or custom config dict (Dict[str, str])
        """

    def s3_config(self, conf: Dict[str, str]) -> "SparkClientBuilder":
        """Configure S3-compatible storage using key-value pairs.

        Maps directly to Spark S3A configuration (spark.hadoop.fs.s3a.*).
        Keys can be provided with or without the "spark.hadoop.fs.s3a." prefix.
        Future S3A configs work automatically without SDK code changes.

        Args:
            conf: Dictionary of S3A configuration (endpoint, access.key, secret.key,
                region, path.style.access, etc.)
        """

    def seaweedfs_config(
        self,
        conf: Optional[Dict[str, str]] = None,
        service_name: str = "seaweedfs",
        namespace: str = "kubeflow",
        port: int = 8333,
    ) -> "SparkClientBuilder":
        """Configure SeaweedFS S3 integration using key-value pairs.

        Auto-configures SeaweedFS with sensible defaults, then applies any additional
        S3A configs from the conf dict. Maps directly to Spark S3A configuration.

        Args:
            conf: Optional dictionary of additional S3A configuration overrides
            service_name: Kubernetes service name (default: "seaweedfs")
            namespace: Kubernetes namespace (default: "kubeflow")
            port: S3 port (default: 8333)
        """

    def volume(self, name: str, mount_path: str, **spec) -> "SparkClientBuilder":
        """Add a volume (for driver and executors)."""

    def node_selector(self, key: str, value: str) -> "SparkClientBuilder":
        """Add node selector."""

    def toleration(self, key: str, operator: str, value: str, effect: str) -> "SparkClientBuilder":
        """Add toleration."""

    def build(self) -> "SparkClient":
        """Create SparkClient with configured settings."""
```

### SparkClient

```python
class SparkClient:
    """Stateless Spark client for Kubeflow - manages Spark Connect servers and batch jobs."""

    def __init__(
        self,
        backend_config: Union[KubernetesBackendConfig, GatewayBackendConfig] = None,
    ):
        """Initialize SparkClient (TrainerClient-aligned constructor).

        Args:
            backend_config: Backend configuration. Defaults to KubernetesBackendConfig.
        """

    @classmethod
    def builder(cls) -> SparkClientBuilder:
        """Create a builder for advanced configuration."""

    def connect(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        name: Optional[str] = None,
        app_name: Optional[str] = None,
        num_executors: Optional[int] = None,
        resources_per_executor: Optional[Dict[str, str]] = None,
        spark_conf: Optional[Dict[str, str]] = None,
        driver: Optional[Driver] = None,
        executor: Optional[Executor] = None,
        options: Optional[List] = None,
    ) -> SparkSession:
        """Connect to Spark - unified API for both existing servers and new sessions.

        This method supports two modes based on parameters:
        - **Connect mode**: When `url` is provided, connects to an existing Spark Connect server
        - **Create mode**: When `url` is not provided, creates a new Spark Connect session

        Args:
            url: Optional URL to existing Spark Connect server (e.g., "sc://server:15002").
                 If provided, connects to existing server. If None, creates new session.
            token: Optional authentication token for existing server.
            name: Optional session name. Auto-generated if not provided (create mode only).
            app_name: Optional Spark application name (create mode only).
            num_executors: Number of executor instances (create mode only).
            resources_per_executor: Resource requirements per executor as dict.
                Format: `{"cpu": "5", "memory": "10Gi", "cpu_limit": "8", "memory_limit": "12Gi"}` (create mode only).
            spark_conf: Spark configuration dictionary (create mode only).
            driver: Driver configuration object (create mode only).
            executor: Executor configuration object (create mode only).
            options: List of configuration options (create mode only).

        Returns:
            SparkSession connected to Spark (self-managing).

        Examples:
            spark = client.connect(url="sc://server:15002")

            spark = client.connect(
                num_executors=5,
                resources_per_executor={
                    "cpu": "5",
                    "memory": "10Gi"
                },
                spark_conf={"spark.sql.adaptive.enabled": "true"}
            )

            spark = client.connect()
        """

    def list_sessions(self) -> List[SparkConnectInfo]:
        """List active Spark Connect sessions."""

    def get_session(self, name: str) -> SparkConnectInfo:
        """Get info about a specific Spark Connect session."""

    def get_session_logs(
        self,
        name: str,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a Spark Connect session.

        Args:
            name: Name of the Spark Connect session.
            follow: Whether to stream logs in realtime.
        """

    def delete_session(self, name: str) -> None:
        """Delete a Spark Connect session."""

    def submit_job(
        self,
        func: Optional[Callable[[SparkSession], Any]] = None,
        func_args: Optional[Dict[str, Any]] = None,
        main_file: Optional[str] = None,
        main_class: Optional[str] = None,
        arguments: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Submit a batch Spark job.

        Supports two modes based on parameters:
        - Function mode: Pass `func` to submit a Python function with Spark transformations
        - File mode: Pass `main_file` to submit an existing Python/Jar file

        Args:
            func: Python function that receives SparkSession (function mode).
            func_args: Arguments to pass to the function.
            main_file: Path to Python/Jar file (file mode).
            main_class: Main class for Jar files.
            arguments: Command-line arguments for the job.
            name: Optional job name.

        Returns:
            The job name (string) for tracking.

        Raises:
            ValueError: If neither `func` nor `main_file` is provided, or both are provided.
        """

    def list_jobs(
        self,
        status: Optional[SparkJobStatus] = None,
    ) -> List[SparkJob]:
        """List batch Spark jobs."""

    def get_job(self, name: str) -> SparkJob:
        """Get a specific Spark job by name."""

    def get_job_logs(
        self,
        name: str,
        container: str = "driver",
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a Spark job (driver or executor)."""

    def wait_for_job_status(
        self,
        name: str,
        status: Set[SparkJobStatus] = {SparkJobStatus.COMPLETED},
        timeout: int = 600,
    ) -> SparkJob:
        """Wait for a job to reach desired status."""

    def delete_job(self, name: str) -> None:
        """Delete a Spark job."""
```

---

## TrainerClient API Alignment

SparkClient follows similar API patterns used by TrainerClient to provide a consistent developer experience across Kubeflow SDKs.

Both clients expose lifecycle-oriented APIs for submitting workloads, monitoring execution, retrieving logs, and managing resources. Where applicable, SparkClient reuses naming conventions and behavioral patterns already established by TrainerClient.

### Lifecycle API Alignment

| TrainerClient | SparkClient |
|---------------|-------------|
| `train()` | `submit_job()` |
| `list_jobs()` | `list_jobs()` |
| `get_job()` | `get_job()` |
| `get_job_logs()` | `get_job_logs()` |
| `wait_for_job_status()` | `wait_for_job_status()` |
| `delete_job()` | `delete_job()` |

### Design Principles

- Consistent lifecycle management APIs across Kubeflow SDKs.
- Similar naming conventions for job operations where semantics match.
- Reuse of common Kubeflow SDK patterns for configuration, status tracking, and resource management.
- Seamless integration between Spark-based data processing workflows and Kubeflow Trainer workloads.

This alignment reduces the learning curve for users who work across multiple Kubeflow components and enables Spark-based preprocessing workflows to integrate naturally with model training workflows.

---

## User Personas

The SparkClient SDK is designed for different user personas with varying needs:

```
+------------------+     +------------------+     +------------------+
|  Data Engineer   |     |  Data Scientist  |     |   ML Engineer    |
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
| - Batch ETL jobs |     | - Interactive    |     | - Feature eng.   |
| - Job scheduling |     |   exploration    |     | - Training data  |
| - Log monitoring |     | - Notebooks      |     | - Hybrid workflow|
| - Queue routing  |     | - Ad-hoc queries |     | - KFP integration|
|                  |     |                  |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
    submit_job()            connect()            Both modes +
                                                  Kubeflow Trainer
```

---

## Use Cases

### 1. Data Scientist: Quick Data Exploration

```python
from kubeflow.spark import SparkClient

client = SparkClient()
spark = client.connect()

df = spark.read.parquet("s3a://data/sales/")
df.printSchema()
df.describe().show()

result = df.groupBy("product").sum("revenue").orderBy("sum(revenue)", ascending=False)
result.show(10)

spark.stop()
```

### 2. ML Engineer: Feature Engineering

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="ml-jobs"))
spark = client.connect(
    num_executors=20,
    resources_per_executor={
        "cpu": "4",
        "cpu_limit": "8",
        "memory": "32Gi",
        "memory_limit": "40Gi"
    },
    spark_conf={"spark.sql.adaptive.enabled": "true"},
    app_name="feature-engineering"
)

raw_data = spark.read.parquet("s3a://data/events/")
features = raw_data.select(
    "user_id",
    "event_type",
    F.hour("timestamp").alias("hour"),
    F.dayofweek("timestamp").alias("day_of_week"),
)
features.write.parquet("s3a://data/features/")

spark.stop()
```

### 3. Platform Engineer: Connect to Shared Cluster

```python
from kubeflow.spark import SparkClient

client = SparkClient()
spark = client.connect(
    url="sc://spark-cluster.spark-system.svc:15002",
    token="team-token",
)

spark.sql("SELECT * FROM shared_database.table").show()
spark.stop()
```

### 4. Notebook Workflow

```python
from kubeflow.spark import SparkClient

client = SparkClient()
spark = client.connect()

df = spark.read.json("s3a://logs/")
df.groupBy("status_code").count().show()

spark.stop()
```

### 5. Data Engineer: Batch ETL Job

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="etl-jobs"))

job_name = client.submit_job(
    main_file="s3a://bucket/etl/daily_pipeline.py",
    arguments=["--date", "2024-01-15", "--output", "s3a://bucket/output/"],
    name="daily-etl-2024-01-15",
)

job = client.get_job(job_name)
print(f"Job submitted: {job_name}")
print(f"Status: {job.status}")

completed_job = client.wait_for_job_status(job_name, timeout=3600)
print(f"Final status: {completed_job.status}")

for line in client.get_job_logs(job_name, container="driver"):
    print(line)
```

### 6. ML Engineer: Feature Pipeline with Kubeflow Trainer

```python
from kubeflow.spark import SparkClient
from kubeflow.trainer import TrainerClient, CustomTrainer
from kubeflow.trainer.types import S3DatasetInitializer
from kubeflow.common.types import KubernetesBackendConfig

# Step 1: Run Spark feature engineering
spark_client = SparkClient(backend_config=KubernetesBackendConfig(namespace="ml-jobs"))

job_name = spark_client.submit_job(
    main_file="s3a://ml/feature_pipeline.py",
    arguments=["--output", "s3a://ml/features/"],
)
spark_client.wait_for_job_status(job_name, timeout=7200)

# Step 2: Train model using extracted features
def train_model():
    import torch
    # Training logic using features from s3a://ml/features/
    ...

trainer = TrainerClient()
trainer.train(
    initializer=S3DatasetInitializer(storage_uri="s3a://ml/features/"),
    trainer=CustomTrainer(func=train_model),
)
```

---

## Batch Job Submission: `submit_job`

`submit_job` uses **function overloading** - the parameter you provide determines the mode:

| Parameter | Mode | Use Case |
|-----------|------|----------|
| `main_file=...` | File mode | Local Python files, remote application URIs, CI/CD pipelines |
| `func=...` | Function mode | Inline transformations, notebooks |

### File Mode (`main_file`)

File mode supports both local Python files and remote application URIs.

When a local Python file is provided, SparkClient automatically packages the file using a Kubernetes ConfigMap before creating the SparkApplication resource. This allows application code from the user's local environment to be executed inside Kubernetes-managed Spark workloads.

When a remote URI is provided, SparkClient directly references the remote resource without creating a ConfigMap.

Supported URI schemes include:

- `s3://`
- `s3a://`
- `gs://`
- `http://`
- `https://`
- `hdfs://`

The following diagram illustrates the high-level batch job submission flow:
```text

                submit_job()
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Local Python File         Remote URI
          |                       |
          v                       |
      ConfigMap                   |
          |                       |
          +-----------+-----------+
                      |
                      v
             SparkApplication
                      |
                      v
              Spark Operator
                      |
                      v
          Driver + Executor Pods
                      |
                      v
             Spark Job Execution
```

### Example: File Mode (`main_file`)

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="etl-jobs"))

job_name = client.submit_job(
    main_file="s3a://bucket/etl/daily_pipeline.py",
    arguments=["--date", "2024-01-15"],
)

client.wait_for_job_status(job_name)
```

#### Local File Packaging Workflow

When a local Python file is provided, SparkClient automatically packages the file using a Kubernetes ConfigMap before creating the SparkApplication resource.

```python
client.submit_job(
    main_file="./etl.py",
)
```
The resulting SparkApplication references the mounted file using a Spark-compatible local path:

```yaml
mainApplicationFile: local:///opt/spark/app/etl.py
```

This approach allows Spark workloads running inside Kubernetes to execute application code that originates from a user's local environment.

### Example: Function Mode (`func`)

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def etl_pipeline(spark: SparkSession, date: str, output_path: str):
    """ETL logic with Spark transformations."""
    df = spark.read.parquet(f"s3a://data/raw/{date}/")

    result = (
        df.filter(df.status == "valid")
        .groupBy("category")
        .agg(F.sum("amount").alias("total"))
    )

    result.write.parquet(output_path)

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="etl-jobs"))

job_name = client.submit_job(
    func=etl_pipeline,
    func_args={"date": "2024-01-15", "output_path": "s3a://data/processed/"},
)

client.wait_for_job_status(job_name)
```

> **Note**: Function mode (`func=...`) will be available in Phase 2.

---

## Job Status Tracking

SparkClient provides lifecycle management APIs for monitoring and tracking batch Spark jobs submitted through the Spark Operator.

The SDK abstracts SparkApplication status details and exposes a simplified interface for users to monitor job progress, retrieve execution information, and wait for completion.

### Lifecycle Management APIs

```python
job = client.get_job(job_name)

completed_job = client.wait_for_job_status(
    job_name,
    timeout=3600,
)

client.delete_job(job_name)
```

These APIs provide a consistent experience for interacting with Spark jobs throughout their lifecycle.

### Job Status Model

Spark jobs transition through a set of SDK-level states that are derived from the underlying SparkApplication status.

| SparkApplication State | SDK State |
|-----------------------|-----------|
| Submitted | Submitted |
| Running | Running |
| Completed | Completed |
| Failed | Failed |

This abstraction allows users to work with a stable API without requiring knowledge of Spark Operator implementation details.

### Waiting for Job Completion

The `wait_for_job_status()` API enables applications and pipelines to block until a job reaches a desired state.

```python
completed_job = client.wait_for_job_status(
    job_name,
    timeout=3600,
)
```

This API is particularly useful for workflow orchestration and integration with Kubeflow Pipelines and Trainer workflows.

### Job Information

The `get_job()` API returns metadata and execution information associated with a Spark job.

Potential job metadata includes:

- Job name
- Current status
- Spark application identifier
- Creation timestamp
- Start time
- Completion time
- Spark UI URL

This information enables users to track execution progress and navigate to Spark-native tooling for deeper inspection.

### Future Enhancements

Future work may include exposing additional execution metadata and job events through dedicated APIs to simplify troubleshooting and operational monitoring.

---

## Features

| Feature | Description |
|---------|-------------|
| **Unified `connect()` API** | Single method for both creating sessions and connecting to existing servers |
| **Auto-provisioning** | Creates Spark Connect server when configuration is provided to `connect()` |
| **Connect mode** | Connect to existing servers via `connect(url=...)` |
| **Self-managing sessions** | `connect()` returns SparkSession that manages itself |
| **Full PySpark API** | Returns standard `SparkSession` |
| **Simplified configuration** | Direct parameters like `num_executors`, `resources_per_executor`, `spark_conf` |
| **Resource config** | Memory, cores, executors, GPU with requests/limits |
| **S3/Storage integration** | SeaweedFS, MinIO, AWS S3 support |
| **K8s integration** | Volumes, node selectors, tolerations |
| **Spark config** | Any `spark.conf` settings |
| **Custom images** | Use your own Spark images |

---

## Architecture

```
SparkClient (Stateless)
    │
    ├── Interactive Sessions (returns self-managing SparkSession)
    │   │
    │   ├── connect() ──► Unified API
    │   │   │
    │   │   ├── connect(url=...) ──► sc://server:15002 ──► SparkSession
    │   │   │
    │   │   └── connect(num_executors=...) ──► SparkConnect CRD ──► Server Pod + Executors
    │   │                                                   │
    │   │                                                   ▼
    │   │                                             SparkSession
    │   │                                             (self-managing)
    │   │
    │   ├── list_sessions() ──► List[SparkConnectInfo]
    │   ├── get_session_logs(name) ──► Iterator[str]
    │   └── delete_session(name)
    │
    └── Batch Jobs (managed by name)
        │
        ├── submit_job(...) ──► SparkApplication CRD ──► job_name (str)
        ├── list_jobs() ──► List[SparkJob]
        ├── get_job(name) ──► SparkJob
        ├── get_job_logs(name) ──► Iterator[str]
        ├── wait_for_job_status(name) ──► SparkJob
        └── delete_job(name)
```

**Backend**: KubernetesBackend using Spark Operator CRDs

- Extensible for future backends (Gateway/Livy)

---

## Backend Architecture

The SparkClient uses a pluggable backend architecture that supports both direct Kubernetes access and REST API-based services:

```
                        SparkBackend (ABC)
                              |
              +---------------+---------------+
              |                               |
      KubernetesBackend              RESTSparkBackend (ABC)
      - SparkConnect CRD                      |
      - SparkApplication CRD        +---------+---------+
                                    |                   |
                            GatewayBackend        LivyBackend
                            - BPG REST API        - Livy REST API
                            - Queue routing       - Session mgmt
                            - Multi-cluster       - Batch/Interactive
```

### Backend Implementations

| Backend | Description | Use Case |
|---------|-------------|----------|
| **KubernetesBackend** | Direct K8s API with Spark Operator CRDs | Default, single cluster |
| **GatewayBackend** | Batch Processing Gateway REST API | Multi-cluster, queue routing |
| **LivyBackend** | Apache Livy REST API | Legacy systems, YARN integration |

### Selecting a Backend

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends import GatewayBackendConfig

# Default: Kubernetes backend (uses current namespace)
client = SparkClient.builder().build()

# Kubernetes backend with specific namespace
client = SparkClient.builder().backend(
    KubernetesBackendConfig(namespace="spark-jobs")
).build()

# Gateway backend for multi-cluster
client = SparkClient.builder().backend(
    GatewayBackendConfig(
        base_url="https://gateway.example.com",
        queue="production",
    )
).build()
```

---

## Kubeflow Ecosystem Integration

SparkClient integrates with the broader Kubeflow ecosystem:

```
SparkClient
    |
    +---> connect() (unified API)
    |         - Create new sessions: connect(num_executors=..., resources_per_executor=...)
    |         - Connect to existing: connect(url="sc://server:15002")
    |         - Interactive data exploration
    |         - Notebook workflows
    |
    +---> submit_job()
    |         - Batch ETL pipelines
    |         - Scheduled jobs
    |
    +---> Kubeflow Pipelines (SparkJobOp)
    |         - Pipeline step for Spark ETL
    |         - DAG orchestration with Spark jobs
    |
    +---> Kubeflow Trainer
    |         - Feature preparation -> Training workflow
    |         - S3DatasetInitializer with Spark output
    |
    +---> Spark History MCP Server
              - AI-powered job analysis
              - Performance bottleneck detection
              - Query job metrics via natural language
```

### Integration with Kubeflow Pipelines

```python
from kfp import dsl
from kubeflow.spark.pipelines import SparkJobOp

@dsl.pipeline(name="ml-pipeline")
def ml_pipeline():
    # Spark ETL step
    etl = SparkJobOp(
        name="feature-etl",
        main_file="s3a://ml/etl.py",
        executor_instances=20,
        executor_memory="8g",
    )

    # Training step depends on ETL completion
    train = TrainOp(
        dataset_path=etl.outputs["output_path"],
    )
    train.after(etl)
```

### Integration with Spark History MCP Server

After job completion, job metrics are available in Spark History Server. The MCP Server enables AI-powered analysis:

```python
job_name = client.submit_job(main_file="s3a://etl/job.py")
job = client.wait_for_job_status(job_name)

print(f"Spark UI: {job.spark_ui_url}")
print(f"App ID for history: {job.application_id}")
```

---

## Observability & Monitoring

SparkClient leverages native Spark and Kubernetes observability capabilities to help users understand job execution, troubleshoot failures, and monitor workload performance.

The SDK aims to expose relevant operational information while maintaining a simple and consistent user experience.

### Spark UI Integration

Spark jobs expose execution details through the Spark UI, including:

- Job execution progress
- Stage information
- Task metrics
- Executor information
- Resource utilization

Users can access Spark-native tooling for detailed workload inspection and debugging.

### Spark History Server Integration

Completed jobs can be inspected through Spark History Server.

The SDK may expose metadata such as:

- Spark application identifier
- Spark UI URL
- History Server URL

This enables users to navigate directly from SDK-managed jobs to Spark-native observability tools.

### Metrics

SparkClient may expose metrics collected from Spark and Kubernetes resources, including:

- Job duration
- Executor metrics
- Stage progress
- Resource utilization

These metrics can be used to understand workload behavior and identify performance bottlenecks.

### Monitoring and Troubleshooting

SparkClient provides visibility into job execution through:

- Job status tracking
- Log retrieval APIs
- Spark UI integration
- History Server integration

Together, these capabilities simplify debugging and operational monitoring of Spark workloads running on Kubernetes.

### Future Enhancements

Future work may include deeper integration with observability systems such as:

- Prometheus
- Grafana
- Spark metrics exporters
- Kubeflow observability tooling

The design remains compatible with existing Spark and Kubernetes monitoring ecosystems while allowing future extensions as requirements evolve.

---

## Testing Strategy

### Unit Testing

Unit tests validate individual SDK components, including:

- Input validation
- SparkApplication generation
- Configuration handling
- Resource specification validation
- Error handling

These tests provide fast feedback during development and help maintain API stability.

### Integration Testing

Integration tests validate interactions between SparkClient and the underlying Spark Operator components.

Key scenarios include:

- Spark Connect session creation
- File-based batch job submission (`submit_job(main_file=...)`)
- Function-based batch job submission (`submit_job(func=...)`)
- Job status tracking
- Log retrieval
- Resource cleanup

### End-to-End Testing

End-to-end tests validate complete user workflows running against real Kubernetes and Spark environments.

Representative workflows include:

- Interactive Spark Connect sessions
- File-based batch job submission (`submit_job(main_file=...)`)
- Function-based batch job submission (`submit_job(func=...)`)
- Spark and Kubeflow integration scenarios

Testing coverage will expand as new SparkClient capabilities are introduced across implementation phases.

### Version Compatibility

Testing should cover supported Spark versions, including:

- Spark 3.5.x
- Spark 4.x

This ensures that SparkClient remains compatible across multiple Spark releases and deployment environments.

### Future Enhancements

Future work may explore leveraging upstream Apache Spark integration tests where applicable and expanding automated coverage for additional deployment scenarios.

---

## Implementation Phases

| Phase | Feature | Description |
|-------|---------|-------------|
| **Phase 1** | `connect()` (unified) | Unified API: connect to existing servers or create new sessions |
| **Phase 1** | `connect(url=...)` | Connect to existing Spark Connect servers |
| **Phase 1** | `connect(num_executors=...)` | Auto-provision Spark Connect servers with configuration |
| **Phase 1** | `submit_job(main_file=...)` | File-based batch job submission |
| **Phase 2** | `submit_job(func=...)` | Function-based batch job submission |

---

## Examples and Notebooks

SparkClient will provide example workloads and notebooks that demonstrate common usage patterns across interactive and batch Spark workloads.

These examples serve as reference implementations and help users quickly adopt SparkClient within the Kubeflow ecosystem.

### Example Workloads

Example applications may include:

- Batch ETL pipelines
- Data transformation workloads
- Feature engineering workflows
- Spark and Kubeflow Trainer integration
- Kubeflow Pipelines integration

### Notebook Examples

Notebook-based examples may be provided for:

- Jupyter Notebook
- Kubeflow Notebook
- Google Colab

These examples demonstrate both interactive Spark Connect workflows and batch job submission workflows using SparkClient APIs.

### Kubeflow Ecosystem Examples

Examples may showcase integrations with other Kubeflow components, including:

- Kubeflow Trainer
- Kubeflow Pipelines
- Spark History Server
- Object storage systems such as S3, MinIO, and SeaweedFS

### Future Enhancements

Additional examples and tutorials may be added as SparkClient capabilities expand, including support for function-based job submission, advanced observability workflows, and production deployment patterns.
---


## Future Vision

The SparkClient SDK is designed to evolve with these future enhancements:

1. **Function-based Jobs** (Phase 2): Pass Spark transformations directly via `submit_job(func=...)`
2. **Scheduled Jobs**: Support for ScheduledSparkApplication CRD
3. **Cost Estimation**: Resource cost predictions before job submission
4. **Auto-scaling Recommendations**: Based on historical job metrics
5. **Multi-cluster Routing**: Automatic cluster selection via Gateway backend
6. **Interactive Debugging**: Integration with Spark Connect for live debugging

---

## Dependencies

- `pyspark>=3.4.0` (Spark Connect support)
- `kubernetes` (K8s client)
- Spark Operator installed in cluster (prerequisite)
