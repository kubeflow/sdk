# KEP-107: Spark Client SDK for Kubeflow

## Authors

- Shekhar Rajak - [@shekharrajak](https://github.com/shekharrajak)

Ref: https://github.com/kubeflow/sdk/issues/107

## Summary

A simple Python SDK to run Spark on Kubernetes. The SDK provides `SparkClient` - a single client that:

- **Auto-creates** a Spark Connect server if needed
- **Connects** to existing Spark Connect servers
- Returns a standard **PySpark SparkSession**
- **Auto-cleans up** resources on exit

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

# Creates Spark Connect server automatically with defaults
with SparkClient.builder().build() as client:
    spark = client.session()
    df = spark.sql("SELECT * FROM my_table")
    df.show()
# Auto cleanup on exit
```

### Quick One-Liner

```python
from kubeflow.spark import SparkClient

# Get SparkSession directly
spark = SparkClient.builder().num_executors(5).get_or_create()
```

### Connect to Existing Server

```python
from kubeflow.spark import SparkClient

# Connect to existing Spark Connect server (no auto-provisioning)
client = SparkClient.connect("sc://spark-server:15002")
spark = client.session()
df = spark.read.parquet("s3a://bucket/data/")
client.stop()
```

### Full Configuration (Builder Pattern)

```python
from kubeflow.spark import SparkClient

client = (
    SparkClient.builder()
    .namespace("spark-jobs")
    .driver(cores=2, memory="4g")
    .executor(cores=4, memory="8g")
    .num_executors(10)
    .image("my-spark:3.5.0")
    .spark_conf("spark.sql.shuffle.partitions", "200")
    .spark_conf("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
    .executor_gpu("nvidia.com/gpu", 1)
    .volume("data", "/mnt/data", persistentVolumeClaim={"claimName": "my-pvc"})
    .build()
)

spark = client.session(app_name="my-analysis")
# ...
client.stop()
```

---

## SparkClient API

### SparkClientBuilder

The Builder Pattern provides a fluent API for configuration - readable, discoverable, and extensible.

```python
class SparkClientBuilder:
    """Builder for SparkClient configuration."""

    # Kubernetes settings
    def namespace(self, ns: str) -> "SparkClientBuilder":
        """Set Kubernetes namespace."""

    def service_account(self, sa: str) -> "SparkClientBuilder":
        """Set service account for Spark pods."""

    # Resource configuration
    def driver(self, cores: int = 1, memory: str = "1g") -> "SparkClientBuilder":
        """Configure driver resources."""

    def executor(self, cores: int = 1, memory: str = "1g") -> "SparkClientBuilder":
        """Configure executor resources."""

    def num_executors(self, n: int) -> "SparkClientBuilder":
        """Set number of executors."""

    def driver_gpu(self, resource: str, count: int) -> "SparkClientBuilder":
        """Add GPU to driver (e.g., 'nvidia.com/gpu', 1)."""

    def executor_gpu(self, resource: str, count: int) -> "SparkClientBuilder":
        """Add GPU to executors."""

    # Image settings
    def image(self, image: str) -> "SparkClientBuilder":
        """Set custom Spark image."""

    def spark_version(self, version: str) -> "SparkClientBuilder":
        """Set Spark version (default: 3.5.0)."""

    # Spark configuration
    def spark_conf(self, key: str, value: str) -> "SparkClientBuilder":
        """Add a Spark configuration property."""

    def spark_confs(self, conf: Dict[str, str]) -> "SparkClientBuilder":
        """Add multiple Spark configuration properties."""

    # Kubernetes advanced
    def volume(self, name: str, mount_path: str, **spec) -> "SparkClientBuilder":
        """Add a volume (for driver and executors)."""

    def node_selector(self, key: str, value: str) -> "SparkClientBuilder":
        """Add node selector."""

    def toleration(self, key: str, operator: str, value: str, effect: str) -> "SparkClientBuilder":
        """Add toleration."""

    # Lifecycle
    def cleanup_on_stop(self, cleanup: bool) -> "SparkClientBuilder":
        """Set whether to delete server on stop (default: True)."""

    # Build
    def build(self) -> "SparkClient":
        """Create SparkClient with configured settings."""

    def get_or_create(self) -> SparkSession:
        """Build client and immediately return SparkSession."""
```

### SparkClient

```python
class SparkClient:
    """Spark client for Kubeflow - auto-provisions Spark Connect servers."""

    @classmethod
    def builder(cls) -> SparkClientBuilder:
        """Create a builder for configuring SparkClient."""

    @classmethod
    def connect(cls, url: str, token: str = None, use_ssl: bool = False) -> "SparkClient":
        """Connect to an existing Spark Connect server (no auto-provisioning)."""

    def session(self, app_name: Optional[str] = None) -> SparkSession:
        """Get or create SparkSession. Creates Spark Connect server if needed."""

    @property
    def spark(self) -> SparkSession:
        """Shortcut to session()."""

    def stop(self) -> None:
        """Stop session and cleanup server."""

    def status(self) -> ServerStatus:
        """Get Spark Connect server status."""

    def logs(self, follow: bool = False) -> Iterator[str]:
        """Get server logs."""

    # Context manager
    def __enter__(self) -> "SparkClient": ...
    def __exit__(self, ...): ...

    # === Batch Job Support ===
    def submit_batch(
        self,
        main_file: str,
        main_class: Optional[str] = None,
        arguments: List[str] = None,
        name: Optional[str] = None,
    ) -> SparkJob:
        """Submit a batch Spark job using SparkApplication CRD."""

    def list_jobs(
        self,
        mode: Optional[SparkMode] = None,
        status: Optional[SparkJobStatus] = None,
    ) -> List[SparkJob]:
        """List Spark jobs (interactive sessions or batch jobs)."""

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
    submit_batch()           session()             Both modes +
                                                  Kubeflow Trainer
```

---

## Use Cases

### 1. Data Scientist: Quick Data Exploration

```python
from kubeflow.spark import SparkClient

client = (
    SparkClient.builder()
    .executor(memory="16g")
    .num_executors(5)
    .build()
)

with client:
    spark = client.session()

    # Load and explore data
    df = spark.read.parquet("s3a://data/sales/")
    df.printSchema()
    df.describe().show()

    # Interactive analysis
    result = df.groupBy("product").sum("revenue").orderBy("sum(revenue)", ascending=False)
    result.show(10)
```

### 2. ML Engineer: Feature Engineering

```python
from kubeflow.spark import SparkClient

client = (
    SparkClient.builder()
    .namespace("ml-jobs")
    .executor(cores=4, memory="32g")
    .num_executors(20)
    .spark_conf("spark.sql.adaptive.enabled", "true")
    .spark_conf("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
    .build()
)

spark = client.session(app_name="feature-engineering")

# Large-scale feature extraction
raw_data = spark.read.parquet("s3a://data/events/")
features = raw_data.select(
    "user_id",
    "event_type",
    F.hour("timestamp").alias("hour"),
    F.dayofweek("timestamp").alias("day_of_week"),
)
features.write.parquet("s3a://data/features/")

client.stop()
```

### 3. Platform Engineer: Connect to Shared Cluster

```python
from kubeflow.spark import SparkClient

# Connect to team's shared Spark Connect server
client = SparkClient.connect(
    url="sc://spark-cluster.spark-system.svc:15002",
    token="team-token",
)

spark = client.session()
# Run queries on shared cluster
spark.sql("SELECT * FROM shared_database.table").show()

client.stop()  # Disconnects only, doesn't delete shared server
```

### 4. Notebook Workflow

```python
# Cell 1: Setup
from kubeflow.spark import SparkClient

client = SparkClient.builder().num_executors(10).build()
spark = client.session()

# Cell 2: Load data
df = spark.read.json("s3a://logs/")

# Cell 3: Analysis
df.groupBy("status_code").count().show()

# Cell 4: More analysis...

# Cell N: Cleanup
client.stop()
```

### 5. Data Engineer: Batch ETL Job

```python
from kubeflow.spark import SparkClient

# Submit a batch ETL job
client = SparkClient.builder().namespace("etl-jobs").build()

job = client.submit_batch(
    main_file="s3a://bucket/etl/daily_pipeline.py",
    arguments=["--date", "2024-01-15", "--output", "s3a://bucket/output/"],
    name="daily-etl-2024-01-15",
)

# Monitor job progress
print(f"Job submitted: {job.name}")
print(f"Status: {job.status}")

# Wait for completion
completed_job = client.wait_for_job_status(job.name, timeout=3600)
print(f"Final status: {completed_job.status}")

# Get logs for debugging
for line in client.get_job_logs(job.name, container="driver"):
    print(line)
```

### 6. ML Engineer: Feature Pipeline with Kubeflow Trainer

```python
from kubeflow.spark import SparkClient
from kubeflow.trainer import TrainerClient, CustomTrainer
from kubeflow.trainer.types import S3DatasetInitializer

# Step 1: Run Spark feature engineering
spark_client = SparkClient.builder().namespace("ml-jobs").build()

feature_job = spark_client.submit_batch(
    main_file="s3a://ml/feature_pipeline.py",
    arguments=["--output", "s3a://ml/features/"],
)
spark_client.wait_for_job_status(feature_job.name, timeout=7200)

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

## Features

| Feature | Description |
|---------|-------------|
| **Auto-provisioning** | Creates Spark Connect server automatically |
| **Connect mode** | Connect to existing servers via `SparkClient.connect()` |
| **Auto-cleanup** | Deletes server on `stop()` (configurable) |
| **Context manager** | `with client:` for automatic cleanup |
| **Full PySpark API** | Returns standard `SparkSession` |
| **Resource config** | Memory, cores, executors, GPU |
| **K8s integration** | Volumes, node selectors, tolerations |
| **Spark config** | Any `spark.conf` settings |
| **Custom images** | Use your own Spark images |
| **Builder pattern** | Fluent, readable configuration |

---

## Architecture

```
SparkClient
    │
    ├── session() ──► creates Spark Connect server (if needed)
    │                     │
    │                     ▼
    │               SparkApplication CRD
    │                     │
    │                     ▼
    │               Spark Driver Pod
    │               (with Spark Connect enabled)
    │                     │
    └── connect() ──────► sc://server:15002 ──► SparkSession
```

**Backend**: KubernetesBackend using Spark Operator CRDs

- Extensible for future backends (Gateway/Livy)
- Batch job support can be added later

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
from kubeflow.spark.backends import KubernetesBackendConfig, GatewayBackendConfig

# Default: Kubernetes backend
client = SparkClient.builder().build()

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
# Job info available for AI analysis
job = client.submit_batch(main_file="s3a://etl/job.py")
client.wait_for_job_status(job.name)

print(f"Spark UI: {job.spark_ui_url}")
print(f"App ID for history: {job.application_id}")

# AI agents can now query:
# - "Why was stage 5 slow?"
# - "Compare this job with yesterday's run"
# - "What caused the failure?"
```

---

## Future Vision

The SparkClient SDK is designed to evolve with these future enhancements:

1. **Scheduled Jobs**: Support for ScheduledSparkApplication CRD
2. **Cost Estimation**: Resource cost predictions before job submission
3. **Auto-scaling Recommendations**: Based on historical job metrics
4. **Multi-cluster Routing**: Automatic cluster selection via Gateway backend
5. **Interactive Debugging**: Integration with Spark Connect for live debugging

---

## Dependencies

- `pyspark>=3.4.0` (Spark Connect support)
- `kubernetes` (K8s client)
- Spark Operator installed in cluster (prerequisite)
