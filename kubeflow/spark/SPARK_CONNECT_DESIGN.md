# Spark Connect Integration Design

**Version:** 2.0
**Status:** Implementation Complete
**Last Updated:** 2025-11-23

## Overview

This document describes the architecture and design of Spark Connect support in Kubeflow Spark SDK, enabling interactive, session-based Spark workloads through a dedicated client class.

**Key Features:**
- Remote connectivity to Spark clusters via gRPC (Spark Connect protocol)
- Specialized clients for batch jobs and interactive sessions
- Native PySpark API compatibility with Kubeflow enhancements
- Kubernetes-native integration with automatic secret/config injection
- Type-safe API following Interface Segregation Principle

---

## Architecture

### System Components

```
┌──────────────────────────────────────────────────────────────┐
│                   User Code (Python)                         │
│                                                              │
│  Batch Jobs:                  Interactive Sessions:          │
│  from kubeflow.spark import   from kubeflow.spark import    │
│    BatchSparkClient             SparkSessionClient          │
│  client = BatchSparkClient()  client = SparkSessionClient() │
│  client.submit_application()  session = client.create_session()│
└──────────────┬────────────────────────────┬──────────────────┘
               │                            │
               ▼                            ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   BatchSparkClient       │  │  SparkSessionClient      │
│                          │  │                          │
│  • submit_application()  │  │  • create_session()      │
│  • get_status()          │  │  • list_sessions()       │
│  • wait_for_completion() │  │  • close_session()       │
│  • delete_application()  │  │  • get_session_status()  │
└──────────┬───────────────┘  └──────────┬───────────────┘
           │                             │
     ┌─────┴─────┐                       │
     ▼           ▼                       ▼
┌──────────┐ ┌──────────┐      ┌──────────────┐
│ Operator │ │ Gateway  │      │  Connect     │
│ Backend  │ │ Backend  │      │  Backend     │
│          │ │          │      │              │
│ (Batch)  │ │ (Batch)  │      │ (Session)    │
└────┬─────┘ └────┬─────┘      └──────┬───────┘
     │            │                   │
     ▼            ▼                   ▼
┌──────────┐ ┌──────────┐      ┌──────────────┐
│  Spark   │ │  Livy/   │      │    Spark     │
│ Operator │ │ Gateway  │      │   Connect    │
│(K8s CRDs)│ │ (HTTP)   │      │   (gRPC)     │
└──────────┘ └──────────┘      └──────────────┘
```

### ConnectBackend Architecture

```
┌─────────────────────────────────────────────────────────┐
│           SparkSessionClient                            │
│                                                         │
│  create_session(app_name)                               │
│    ↓                                                     │
│  Delegates to ConnectBackend                            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              ConnectBackend                             │
│              (SessionSparkBackend)                      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  create_session(app_name, **config)             │  │
│  │    ↓                                             │  │
│  │  1. Generate session_id (UUID)                   │  │
│  │  2. Build connection URL                         │  │
│  │  3. Create PySpark SparkSession.builder.remote() │  │
│  │  4. Wrap in ManagedSparkSession                  │  │
│  │  5. Track in _sessions dict                      │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  Session Management:                                    │
│  • list_sessions()                                      │
│  • get_session_status(session_id)                      │
│  • close_session(session_id)                           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│           ManagedSparkSession (Wrapper)                 │
│                                                         │
│  PySpark API (delegated):        Kubeflow Extensions:  │
│  • sql(query)                    • get_metrics()       │
│  • createDataFrame(data)         • get_info()          │
│  • read.parquet()                • upload_artifacts()  │
│  • All DataFrame operations      • context manager     │
│                                                         │
│  Wraps: pyspark.sql.SparkSession (Spark Connect)       │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Code → SparkSessionClient → ConnectBackend → gRPC → Spark Connect Server
                                                              ↓
                                                        Spark Cluster
                                                        (Driver + Executors)
                                                              ↓
Results ← ManagedSparkSession ← gRPC Stream ← Spark Connect Server
```

---

## Design Principles

### 1. Specialized Client Classes

Separate clients for different workloads:

```python
# Batch jobs
from kubeflow.spark import BatchSparkClient, OperatorBackendConfig

config = OperatorBackendConfig(namespace="spark-jobs")
client = BatchSparkClient(backend_config=config)
response = client.submit_application(app_name="batch-job", ...)

# Interactive sessions
from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

config = ConnectBackendConfig(connect_url="sc://spark-connect:15002")
client = SparkSessionClient(backend_config=config)
session = client.create_session(app_name="analysis")
```

Benefits:
- Type safety: Clients only expose relevant methods
- No runtime errors from unsupported operations
- Clear API boundaries
- Follows Interface Segregation Principle

### 2. Native PySpark Delegation

`ManagedSparkSession` delegates all DataFrame operations to native PySpark:

```python
# These call PySpark directly - no wrapping overhead
df = session.sql("SELECT * FROM table")
df = session.createDataFrame(data, schema)
df = session.read.parquet("s3://bucket/data")
result = df.filter(df.age > 30).collect()
```

Benefits:
- Full PySpark API compatibility
- Zero wrapping overhead for DataFrame operations
- Automatic updates when PySpark adds new features
- Standard PySpark documentation applies

### 3. Backend Abstraction

Backends implement specialized abstract base classes:

```python
# Base interface (minimal shared functionality)
class SparkBackend(abc.ABC):
    def close(self): pass

# Batch workloads
class BatchSparkBackend(SparkBackend):
    def submit_application(...) -> SparkApplicationResponse
    def get_status(app_id) -> ApplicationStatus
    def delete_application(app_id) -> Dict
    def get_logs(...) -> Iterator[str]
    def list_applications(...) -> List[ApplicationStatus]
    def wait_for_completion(...) -> ApplicationStatus

# Session workloads
class SessionSparkBackend(SparkBackend):
    def create_session(app_name, **kwargs) -> ManagedSparkSession
    def list_sessions() -> List[SessionInfo]
    def close_session(session_id) -> Dict[str, Any]
    def get_session_status(session_id) -> SessionInfo
```

Implementation hierarchy:
- OperatorBackend extends BatchSparkBackend
- GatewayBackend extends BatchSparkBackend
- ConnectBackend extends SessionSparkBackend

---

## Component Details

### ConnectBackendConfig

Configuration for Spark Connect connectivity:

```python
@dataclass
class ConnectBackendConfig:
    connect_url: str                    # "sc://host:port"
    token: Optional[str] = None         # Bearer token for auth
    use_ssl: bool = True                # Enable TLS
    user_id: Optional[str] = None       # User identity
    timeout: int = 300                  # Connection timeout (seconds)
    grpc_max_message_size: int = 128MB  # gRPC message limit
    namespace: str = "default"          # K8s namespace
```

### ManagedSparkSession

Kubeflow wrapper around native PySpark Connect session:

```python
class ManagedSparkSession:
    # Properties
    @property
    def session_id(self) -> str          # Unique session UUID
    @property
    def app_name(self) -> str            # Application name
    @property
    def is_closed(self) -> bool          # Session state

    # PySpark API (delegated to self._session)
    def sql(self, query: str) -> DataFrame
    def createDataFrame(self, data, schema) -> DataFrame
    def read(self) -> DataFrameReader
    def table(self, table_name: str) -> DataFrame
    def range(self, start, end, step) -> DataFrame

    # Kubeflow extensions
    def get_metrics(self) -> SessionMetrics
    def get_info(self) -> SessionInfo
    def upload_artifacts(self, *paths) -> None
    def close(self) -> None

    # Context manager support
    def __enter__(self) -> "ManagedSparkSession"
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
```

### SessionMetrics

Tracks session activity:

```python
@dataclass
class SessionMetrics:
    session_id: str
    queries_executed: int = 0      # SQL/DataFrame operations
    active_queries: int = 0        # Currently running queries
    artifacts_uploaded: int = 0    # Uploaded JARs/files
    data_read_bytes: int = 0       # Data read
    data_written_bytes: int = 0    # Data written
    execution_time_ms: int = 0     # Total execution time
```

---

## Usage Examples

### Basic Connection

```python
from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

# Configure connection
config = ConnectBackendConfig(
    connect_url="sc://localhost:30000",
    use_ssl=False,
)

# Create client and session
client = SparkSessionClient(backend_config=config)
session = client.create_session(app_name="demo")

# Use standard PySpark API
df = session.sql("SELECT 1 AS id, 'Hello' AS message")
df.show()

# Cleanup
session.close()
client.close()
```

### Context Manager Pattern

```python
from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

config = ConnectBackendConfig(connect_url="sc://spark-server:15002")

with SparkSessionClient(backend_config=config) as client:
    with client.create_session(app_name="analysis") as session:
        # Session auto-closes on exit
        df = session.sql("SELECT * FROM sales")
        result = df.filter(df.amount > 100).collect()
```

### DataFrame Operations

```python
# Create DataFrame from Python data
sales_data = [
    (1, "Electronics", "Laptop", 1200.00, 2),
    (2, "Electronics", "Mouse", 25.00, 5),
    (3, "Clothing", "Shirt", 35.00, 3),
]

df = session.createDataFrame(
    sales_data,
    ["id", "category", "product", "price", "quantity"]
)

# Show data
df.show()
# +---+-----------+--------+------+--------+
# | id|   category| product| price|quantity|
# +---+-----------+--------+------+--------+
# |  1|Electronics|  Laptop|1200.0|       2|
# |  2|Electronics|   Mouse|  25.0|       5|
# |  3|   Clothing|   Shirt|  35.0|       3|
# +---+-----------+--------+------+--------+
```

### Aggregations and GroupBy

```python
from pyspark.sql import functions as F

# Calculate revenue
revenue_df = df.withColumn("revenue", F.col("price") * F.col("quantity"))

# Group by category with multiple aggregations
category_stats = revenue_df.groupBy("category").agg(
    F.sum("revenue").alias("total_revenue"),
    F.avg("price").alias("avg_price"),
    F.count("*").alias("num_transactions")
)

category_stats.show()
# +-----------+-------------+---------+----------------+
# |   category|total_revenue|avg_price|num_transactions|
# +-----------+-------------+---------+----------------+
# |   Clothing|        105.0|     35.0|               1|
# |Electronics|       2525.0|    612.5|               2|
# +-----------+-------------+---------+----------------+

# Sort by revenue
category_stats.orderBy(F.desc("total_revenue")).show()
```

### Window Functions

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Running total by date
window_spec = Window.orderBy("date").rowsBetween(
    Window.unboundedPreceding,
    Window.currentRow
)

daily_revenue = revenue_df.groupBy("date").agg(
    F.sum("revenue").alias("daily_revenue")
).withColumn(
    "running_total",
    F.sum("daily_revenue").over(window_spec)
)

daily_revenue.orderBy("date").show()
```

### Session Management

```python
# List all active sessions
sessions = client.list_sessions()
for s in sessions:
    print(f"Session: {s.session_id}, App: {s.app_name}, State: {s.state}")

# Get session status
status = client.get_session_status(session.session_id)
print(f"Session state: {status.state}")

# Get metrics
metrics = session.get_metrics()
print(f"Queries executed: {metrics.queries_executed}")
print(f"Active queries: {metrics.active_queries}")

# Get session info
info = session.get_info()
print(f"App: {info.app_name}, State: {info.state}")
```

### Multiple Concurrent Sessions

```python
with SparkSessionClient(backend_config=config) as client:
    # Create multiple sessions
    session1 = client.create_session(app_name="analysis-1")
    session2 = client.create_session(app_name="analysis-2")

    try:
        # Each session is independent
        df1 = session1.sql("SELECT 'session1' AS source")
        df2 = session2.sql("SELECT 'session2' AS source")

        print(df1.collect())  # [Row(source='session1')]
        print(df2.collect())  # [Row(source='session2')]
    finally:
        session1.close()
        session2.close()
```

---

## Deployment Guide

### Kubernetes Setup

#### 1. Deploy Spark Connect Server

Use the provided Kubernetes manifest:

```bash
# Deploy Spark Connect server
kubectl apply -f examples/spark/spark-connect-server.yaml

# Verify deployment
kubectl get pods -l app=spark-connect
kubectl logs -l app=spark-connect -f
```

#### 2. Port Forwarding (Local Development)

```bash
# Forward Spark Connect port to localhost
kubectl port-forward -n default svc/spark-connect 30000:15002

# Verify connectivity
nc -zv localhost 30000
```

#### 3. Connect from Python

```python
config = ConnectBackendConfig(
    connect_url="sc://localhost:30000",  # Local port forward
    use_ssl=False,
)

client = SparkSessionClient(backend_config=config)
session = client.create_session(app_name="my-app")
```

### Production Setup

For production, use Kubernetes DNS:

```python
config = ConnectBackendConfig(
    connect_url="sc://spark-connect.default.svc.cluster.local:15002",
    use_ssl=True,
    token=os.getenv("SPARK_TOKEN"),  # From K8s secret
)
```

---

## Interactive Demo

### Quick Start

```bash
# 1. Setup Kubernetes cluster with Spark Connect
cd examples/spark
./setup_spark_connect.sh

# 2. Install dependencies
pip install 'pyspark[connect]>=4.0.0'

# 3. Launch IPython shell
python ipython_spark_connect_shell.py
```

### Step-by-Step Tutorial

The IPython shell provides a guided tutorial. Key steps:

```python
# 1. Create config and client
config = ConnectBackendConfig(
    connect_url="sc://localhost:30000",
    use_ssl=False,
)
client = SparkSessionClient(backend_config=config)

# 2. Create session
session = client.create_session(app_name="tutorial")

# 3. Simple query
df = session.sql("SELECT 1 AS id, 'Hello' AS msg")
df.show()

# 4. Create DataFrame
data = [
    (1, "Electronics", 1200.00),
    (2, "Clothing", 35.00),
]
df = session.createDataFrame(data, ["id", "category", "price"])
df.show()

# 5. Aggregations
from pyspark.sql import functions as F
df.groupBy("category").agg(F.avg("price")).show()

# 6. Cleanup
session.close()
client.close()
```

---

## Key Design Decisions

### 1. Separate Client Classes (Interface Segregation)

**Decision:** Use BatchSparkClient and SparkSessionClient instead of unified client

**Rationale:**
- Different use cases have distinct method requirements
- Batch: submit_application, wait_for_completion, delete_application
- Session: create_session, list_sessions, close_session
- Prevents runtime NotImplementedError exceptions
- Type-safe: clients only expose relevant methods

**Benefits:**
- Compile-time type checking
- Clear API boundaries
- No confusion about which methods work with which backend
- IDE autocomplete shows only valid methods

### 2. Delegation to Native PySpark

**Decision:** Delegate DataFrame operations to native PySpark

**Alternatives Considered:**
- Wrap all PySpark methods → Rejected (maintenance burden)
- Custom DataFrame implementation → Rejected (no value-add)

**Benefits:**
- Zero wrapping overhead
- Full PySpark compatibility
- Automatic feature updates

### 3. URL Parameter Handling (Spark 4.0)

**Decision:** Use simple URL format without parameters

**Issue:** Spark Connect 4.0 doesn't support URL parameters like `/;use_ssl=false`

**Solution:** Pass configuration via `builder.config()` instead of URL

```python
# Before (doesn't work in Spark 4.0)
url = "sc://host:port/;use_ssl=false"

# After (works)
url = "sc://host:port"
builder.config("spark.ssl.enabled", "false")
```

### 4. IPv4 vs IPv6 Binding

**Issue:** Spark Connect server was binding to IPv6 (:::15002) causing connection failures

**Solution:** Force IPv4 binding via Java options

```yaml
env:
- name: JAVA_TOOL_OPTIONS
  value: "-Djava.net.preferIPv4Stack=true"
```

---

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| PySpark Client | 4.0.x | Must match server version |
| Spark Connect Server | 4.0.0 | Running in Kubernetes |
| Kubeflow SDK | Latest | This implementation |
| Kubernetes | 1.24+ | For Spark Operator |
| Python | 3.8+ | Required for PySpark |

**Important:** Client and server versions must match. PySpark 4.0 cannot connect to Spark 3.5 Connect servers.

---

## Resources

### Files Created

- `kubeflow/spark/backends/connect.py` - ConnectBackend implementation
- `kubeflow/spark/session.py` - ManagedSparkSession wrapper
- `kubeflow/spark/models.py` - Data models (ConnectBackendConfig, SessionMetrics, SessionInfo)
- `examples/spark/ipython_spark_connect_shell.py` - Interactive demo shell
- `examples/spark/ipython_spark_connect_demo.py` - Automated demo
- `examples/spark/spark-connect-server.yaml` - Kubernetes deployment
- `examples/spark/setup_spark_connect.sh` - Setup automation
- `examples/spark/SPARK_CONNECT_DEMO.md` - Demo documentation

### Documentation

- [Spark Connect Overview](https://spark.apache.org/docs/latest/spark-connect-overview.html)
- [PySpark Connect API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/connect.html)
- [Demo Guide](examples/spark/SPARK_CONNECT_DEMO.md)

---

## Summary

Spark Connect integration provides:

- **Specialized Clients** - BatchSparkClient for batch jobs, SparkSessionClient for interactive sessions
- **Type Safety** - Interface Segregation Principle prevents runtime errors
- **Native PySpark** - Full DataFrame API with zero overhead
- **Kubernetes-Native** - Automatic config/secret injection
- **Production-Ready** - Session management, metrics, error handling
- **Developer-Friendly** - Context managers, IPython integration, examples

The implementation follows SOLID design principles while providing the full power of PySpark Connect for interactive data analysis and ML workloads.
