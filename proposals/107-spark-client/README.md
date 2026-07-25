# KEP-107: Spark Client SDK for Kubeflow

## Authors

- Shekhar Rajak - [@shekharrajak](https://github.com/shekharrajak)
- Sameer Yadav - [@Goku2099](https://github.com/Goku2099)

Ref: https://github.com/kubeflow/sdk/issues/107

## Summary

A simple Python SDK to run Spark on Kubernetes. The SDK provides `SparkClient`:

- **`connect()` API** - Creates new Spark Connect sessions or connects to existing servers
- **Auto-provisions** Spark Connect servers when configuration is provided
- **Connects** to existing Spark Connect servers when base URL is provided
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
5. Submit and manage batch Spark jobs via SparkApplication CRD
6. Provide lifecycle management APIs for Spark jobs

## Non-Goals

- Supporting Spark outside Kubernetes (local mode, standalone clusters)
- Managing Spark Operator installation
- Replacing the Spark Operator

---


## User Personas

The SparkClient SDK is designed for different user personas with varying needs:

```
+------------------+     +------------------+     +-------------------+
|  Data Engineer   |     |  Data Scientist  |     |   ML Engineer     |
+------------------+     +------------------+     +-------------------+
|                  |     |                  |     |                   |
| - Batch ETL jobs |     | - Interactive    |     | - Feature eng.    |
| - Job scheduling |     |   exploration    |     | - Training data   |
| - Log monitoring |     | - Notebooks      |     | - Batch processing|
| - Queue routing  |     | - Ad-hoc queries |     | - ETL workflows   |
|                  |     |                  |     |                   |
+--------+---------+     +--------+---------+     +--------+----------+
         |                        |                        |
         v                        v                        v
    submit_job()            connect()                  Both modes
```

---

## User Stories

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
    base_url="sc://spark-cluster.spark-system.svc:15002",
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
from kubeflow.spark.types.types import FileJob
from kubeflow.common.types import KubernetesBackendConfig

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="etl-jobs"))

job_name = client.submit_job(
    job=FileJob(
        file_source="s3a://bucket/etl/daily_pipeline.py",
        args=[
            "--date", "2024-01-15",
            "--output", "s3a://bucket/output/",
        ],
    ),
    spark_conf={
        "spark.sql.adaptive.enabled": "true",
    },
    options=[
        Name("daily-etl-2024-01-15"),
    ],
)

job = client.get_job(job_name)
print(f"Job submitted: {job_name}")
print(f"Status: {job.status}")

completed_job = client.wait_for_job_status(job_name, timeout=3600)
print(f"Final status: {completed_job.status}")

for line in client.get_job_logs(job_name):
    print(line)
```

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
spark = client.connect(base_url="sc://spark-server:15002")
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

- **Create Mode**: When `base_url` is not provided, creates a new Spark Connect session with the specified configuration
- **Connect Mode**: When `base_url` is provided, connects to an existing Spark Connect server

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

spark = client.connect(base_url="sc://server:15002")

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

@dataclass
class FileJob:
    """Spark application referenced by a local or remote file source."""

    file_source: str
    args: Optional[List[str]] = None
    main_class: Optional[str] = None


@dataclass
class FuncJob:
    """Function-based Spark application."""

    func: Callable
    func_args: dict | None = None
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
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        num_executors: Optional[int] = None,
        resources_per_executor: Optional[Dict[str, str]] = None,
        spark_conf: Optional[Dict[str, str]] = None,
        driver: Optional[Driver] = None,
        executor: Optional[Executor] = None,
        options: Optional[List] = None,
        timeout: int = 300,
        connect_timeout: int = 120,
    ) -> SparkSession:
        """Connect to Spark - unified API for both existing servers and new sessions.

        This method supports two modes based on parameters:

        - **Connect mode**: When `base_url` is provided, connects to an existing Spark Connect server.
        - **Create mode**: When `base_url` is not provided, creates a new Spark Connect session.

        Args:
            base_url: Optional URL to an existing Spark Connect server
                (e.g. ``sc://server:15002``). If provided, connects to the
                existing server. Otherwise, creates a new session.
            token: Optional authentication token for an existing server.
            num_executors: Number of executor instances (create mode only).
            resources_per_executor: Resource requirements per executor.
                Format: ``{"cpu": "5", "memory": "10Gi"}``.
            spark_conf: Spark configuration dictionary.
            driver: Driver configuration object.
            executor: Executor configuration object.
            options: List of advanced Spark configuration options.
            timeout: Timeout in seconds to wait for the Spark Connect session
                to become ready.
            connect_timeout: Timeout in seconds for establishing the
                SparkSession connection.

        Returns:
            SparkSession connected to Spark.

        Examples:

            # Connect to an existing Spark Connect server
            spark = client.connect(
                base_url="sc://server:15002",
                token="team-token",
            )

            # Create a new Spark Connect session
            spark = client.connect(
                num_executors=5,
                resources_per_executor={
                    "cpu": "5",
                    "memory": "10Gi",
                },
                spark_conf={
                    "spark.sql.adaptive.enabled": "true",
                },
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
        job: FileJob | FuncJob,
        num_executors: Optional[int] = None,
        resources_per_executor: Optional[Dict[str, str]] = None,
        spark_conf: Optional[Dict[str, str]] = None,
        options: Optional[List] = None,
    ) -> str:
        """Submit a batch Spark job.

        This method supports two job types:

        - **FileJob**:  Submit an existing Spark application referenced by a local or remote file source.
        - **FuncJob**: Submit a Python function as a Spark batch job.

        Args:
            job: Job definition describing the workload to execute.
                Supports either FileJob or FuncJob.
            num_executors: Number of executor instances.
            resources_per_executor: Resource requirements per executor as a dictionary.
                Format: `{"cpu": "5", "memory": "10Gi"}`.
            spark_conf: Spark configuration dictionary.
            options: List of additional Spark configuration options.

        Returns:
            The submitted Spark job name for lifecycle management and tracking.

        Examples:

            client.submit_job(
                job=FileJob(
                    file_source="local:///opt/spark/app/etl.py",
                    args=["--date", "2026-06-18"],
                )
            )

            client.submit_job(
                job=FileJob(
                    file_source="s3a://jobs/etl.py",
                    args=["--date", "2026-06-18"],
                )
            )

            client.submit_job(
                job=FuncJob(
                    func=etl_pipeline,
                    func_args={"date": "2026-06-18"},
                )
            )
        """

    def list_jobs(
        self,
        status: Optional[Set[SparkJobStatus]] = None,
    ) -> List[SparkJob]:
        """List batch Spark jobs.

        Args:
            status: Optional set of job statuses used to filter returned jobs.
        """

    def get_job(self, name: str) -> SparkJob:
        """Get a specific Spark job by name."""

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a Spark job.

        Logs are retrieved from the Kubernetes pods associated with the
        SparkApplication. Log retrieval is only available while the target
        pod exists. If the pod has been deleted (for example due to TTL-based
        cleanup), logs may no longer be available.
        """

    def wait_for_job_status(
        self,
        name: str,
        status: Set[SparkJobStatus] = {SparkJobStatus.COMPLETED},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> SparkJob:
        """Wait for a job to reach desired status."""

    def delete_job(self, name: str) -> None:
        """Delete a Spark job."""
```
---

### Batch Job Lifecycle APIs

The SparkClient batch job workflow follows a lifecycle-oriented API similar to other long-running workload APIs in the Kubeflow SDK.

Spark batch workloads share many characteristics with other long-running Kubernetes workloads managed through the Kubeflow SDK. For this reason, the lifecycle APIs intentionally follow patterns established by TrainerClient while remaining specific to SparkApplication-based execution.


| TrainerClient | SparkClient |
|---------------|-------------|
| `train()` | `submit_job()` |
| `list_jobs()` | `list_jobs()` |
| `get_job()` | `get_job()` |
| `get_job_logs()` | `get_job_logs()` |
| `wait_for_job_status()` | `wait_for_job_status()` |
| `delete_job()` | `delete_job()` |


---

## Batch Job Submission: `submit_job`

`submit_job` uses **function overloading** - the parameter you provide determines the mode:

| Job Type | Mode | Use Case |
|----------|------|----------|
| job=FileJob(...) | File mode | Existing scripts, CI/CD pipelines |
| job=FuncJob(...) | Function mode | Inline transformations, notebooks |

### Example: File Mode (`job=FileJob(...)`)

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="etl-jobs"))

job_name = client.submit_job(
    job=FileJob(
        file_source="s3a://bucket/etl/daily_pipeline.py",
        args=["--date", "2024-01-15"],
    )
)

client.wait_for_job_status(job_name)
```

### File Mode Implementation

File mode supports both local and remote application sources.

SparkClient implements long-running batch job execution by translating user-provided job specifications into SparkApplication resources managed by the Spark Operator.

When `file_source` uses a remote URI (for example `s3a://`, `gs://`, `hdfs://`, or `https://`), SparkApplication references the remote application directly.

When `file_source` uses a local URI (for example `local:///opt/spark/app/etl.py`), the application file is expected to already be available to the SparkApplication, such as through a mounted PersistentVolumeClaim (PVC) or a pre-built container image. SparkClient does not package or upload local files automatically.

```python
client.submit_job(
    job=FileJob(
        file_source="local:///opt/spark/app/etl.py",
    )
)
```

```yaml
mainApplicationFile: local:///opt/spark/app/etl.py
```

Remote application sources can be referenced directly:

```python
client.submit_job(
    job=FileJob(
        file_source="s3a://bucket/etl.py",
    )
)
```

SparkClient relies on the Spark Operator's native application submission and dependency management mechanisms. For local application sources, users are responsible for ensuring the referenced file is available to the SparkApplication. This can be achieved through mechanisms such as mounted PVCs or pre-built container images.

Additional dependencies (Python packages, JARs, archives, and other resources) continue to be managed using the Spark Operator's native dependency mechanisms (`deps.files`, `deps.pyFiles`, `deps.jars`, etc.).


### Example: Function Mode (`job=FuncJob(...)`)

```python
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def etl_pipeline(date: str, output_path: str):
    """ETL logic with Spark transformations."""
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(f"s3a://data/raw/{date}/")

    result = (
        df.filter(df.status == "valid")
        .groupBy("category")
        .agg(F.sum("amount").alias("total"))
    )

    result.write.parquet(output_path)

client = SparkClient(backend_config=KubernetesBackendConfig(namespace="etl-jobs"))

job_name = client.submit_job(
    job=FuncJob(
        func=etl_pipeline,
        func_args={
            "date": "2024-01-15",
            "output_path": "s3a://data/processed/",
        },
    )
)

client.wait_for_job_status(job_name)
```
### Function Mode Implementation

Function mode is intended for Python-first workflows where users submit Python callables instead of application files.

SparkClient follows a similar approach to TrainerClient for function-based execution.

1. The SDK extracts the user function source code and invocation arguments.
2. The SDK embeds the extracted function source and invocation directly into the `initContainer` command.
3. When the `initContainer` starts, it reconstructs a standalone Python application (for example `spark_job.py`) in an `emptyDir` volume shared between the `initContainer` and the Spark driver.
4. The SparkApplication references the generated application through a local Spark URI:

```yaml
mainApplicationFile: local:///opt/spark/app/spark_job.py
```

5. The Spark driver executes the generated application using the standard SparkApplication lifecycle.

A simplified SparkApplication illustrates this execution flow:

```yaml
spec:
  volumes:
    - name: spark-app-source
      emptyDir: {}

  initContainers:
    - name: prepare-spark-app
      volumeMounts:
        - name: spark-app-source
          mountPath: /opt/spark/app
      command:
        - bash
        - -c
        - |
          ...
          printf "%s" "$SCRIPT" > /opt/spark/app/main.py

  driver:
    volumeMounts:
      - name: spark-app-source
        mountPath: /opt/spark/app

  mainApplicationFile: local:///opt/spark/app/spark_job.py
```

This approach allows users to submit Python functions without manually creating or packaging Spark application files, while remaining aligned with the SparkApplication execution model and the existing TrainerClient implementation pattern.

> **Note:** The exact command template and generated wrapper may evolve during implementation, but the overall execution flow remains the same.

Once the generated application has been prepared by the `initContainer`, the Spark driver executes it using the same lifecycle management APIs as file-based submissions.

> **Note:** Function mode (`job=FuncJob(...)`) will be available in Phase 2.

---

## SparkJob Model

```python
@dataclass
class SparkJob:
    """Information about a Spark batch job."""

    name: str
    namespace: str
    status: SparkJobStatus | None = None
    creation_timestamp: datetime | None = None
    num_executors: int | None = None
    driver_pod_name: str | None = None
```

---

## Status Model

SparkClient provides lifecycle management APIs for monitoring and tracking Spark jobs submitted through the Spark Operator.

Job state information is derived from the underlying SparkApplication resource while exposing a consistent Python interface.

### Application States

SparkClient derives job state information from SparkApplication status.

```python
class SparkJobStatus(str, Enum):
    """State of a Spark batch job."""

    CREATED = "Created"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
```

SparkApplication-specific states are mapped into these SDK-level states to provide a simpler and more consistent user experience across Kubeflow SDK clients.

| SDK Status | SparkApplication States |
|------------|-------------------------|
| CREATED | SUBMITTED |
| RUNNING | RUNNING, SUCCEEDING, SUSPENDING, SUSPENDED, RESUMING |
| COMPLETED | COMPLETED |
| FAILED | FAILED, SUBMISSION_FAILED, FAILING, PENDING_RERUN, INVALIDATING, UNKNOWN |

The SDK intentionally exposes a simplified status model in Phase 1. Additional status categories may be introduced in future releases based on user feedback and operational requirements.

---

## Features

| Feature | Description |
|---------|-------------|
| **Unified `connect()` API** | Single method for both creating sessions and connecting to existing servers |
| **Auto-provisioning** | Creates Spark Connect server when configuration is provided to `connect()` |
| **Connect mode** | Connect to existing servers via `connect(base_url=...)` |
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
    │   │   ├── connect(base_url=...) ──► sc://server:15002 ──► SparkSession
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
    └── Batch Jobs (managed through SparkApplication)
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

## Implementation Phases

| Phase | Feature | Description |
|-------|---------|-------------|
| **Phase 1** | `connect()` (unified) | Unified API: connect to existing servers or create new sessions |
| **Phase 1** | `connect(base_url=...)` | Connect to existing Spark Connect servers |
| **Phase 1** | `connect(num_executors=...)` | Auto-provision Spark Connect servers with configuration |
| **Phase 1** | `submit_job(job=FileJob(...))` |  SparkApplication-based batch job submission and lifecycle APIs |
| **Phase 2** | `submit_job(job=FuncJob(...))` | Function-based batch job submission |

---

## Future Vision

The SparkClient SDK is designed to evolve with these future enhancements:

1. **Function-based Jobs** (Phase 2): Pass Spark transformations directly via `submit_job(job=FuncJob(...))`
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
