# Kubeflow Spark Client

Cloud-native Python client for managing Apache Spark applications on Kubernetes using the Kubeflow Spark Operator.

## Overview

The Kubeflow Spark Client provides a Pythonic interface for submitting, monitoring, and managing Spark applications on Kubernetes. The SDK offers two specialized clients for different workloads:

- **BatchSparkClient**: For batch Spark application submission and management
- **SparkSessionClient**: For interactive Spark Connect sessions

### Key Features

- **Specialized Clients**: Separate clients for batch jobs and interactive sessions
- **Cloud-Native Architecture**: Direct integration with Kubeflow Spark Operator CRDs
- **Multiple Backends**: Operator (K8s-native), Gateway (REST API), and Connect (gRPC) backends
- **Dynamic Resource Allocation**: Automatic executor scaling based on workload
- **Comprehensive Monitoring**: Prometheus metrics and Spark UI integration
- **Production-Ready**: Error handling, retries, and comprehensive logging
- **Type-Safe**: Clean APIs with proper type hints and IDE support

## Architecture

```
BaseSparkClient (shared functionality)
├── BatchSparkClient (batch workloads)
│   └── Backend: BatchSparkBackend
│       ├── OperatorBackend (Kubernetes CRDs)
│       └── GatewayBackend (REST API)
│
└── SparkSessionClient (interactive workloads)
    └── Backend: SessionSparkBackend
        └── ConnectBackend (Spark Connect/gRPC)
```

### Design Principles

The Spark client follows best practices and SOLID principles:

1. **Interface Segregation**: Separate clients expose only relevant methods
2. **Backend Abstraction**: Pluggable backends for different platforms
3. **Type Safety**: Strong typing prevents runtime errors
4. **Kubernetes-Native**: Direct CRD manipulation for cloud-native deployments

## Installation

```bash
# Install from PyPI (when released)
pip install kubeflow

# Or install from source
cd sdk
pip install -e .

# For Spark Connect support
pip install 'pyspark[connect]>=3.4.0'
```

### Prerequisites

**For BatchSparkClient with OperatorBackend** (recommended for batch jobs):
- Kubernetes cluster (1.16+)
- Kubeflow Spark Operator installed
- kubectl configured with proper context
- Service account with SparkApplication permissions

**For BatchSparkClient with GatewayBackend**:
- Access to a Spark Gateway (e.g., Apache Livy)
- API credentials (if required)

**For SparkSessionClient with ConnectBackend**:
- Spark cluster with Spark Connect server (Spark 3.4+)
- Network connectivity to Spark Connect endpoint
- PySpark with Connect support installed

## Quick Start

### Batch Jobs

#### Basic Batch Application

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
    num_executors=2
)

print(f"Submitted: {response.submission_id}")

# Wait for completion
status = client.wait_for_completion(response.submission_id)
print(f"Final state: {status.state}")

# Get logs
for line in client.get_logs(response.submission_id):
    print(line)
```

#### DataFrame Processing with S3

```python
from kubeflow.spark import BatchSparkClient, OperatorBackendConfig

# Configure client
config = OperatorBackendConfig(
    namespace="spark-jobs",
    enable_monitoring=True,
    enable_ui=True,
)
client = BatchSparkClient(backend_config=config)

# Submit DataFrame processing job
response = client.submit_application(
    app_name="dataframe-analysis",
    main_application_file="s3a://my-bucket/jobs/analysis.py",
    spark_version="4.0.0",
    driver_cores=2,
    driver_memory="4g",
    executor_cores=2,
    executor_memory="8g",
    num_executors=5,
    spark_conf={
        "spark.sql.shuffle.partitions": "200",
        "spark.hadoop.fs.s3a.endpoint": "s3.amazonaws.com",
    },
    env_vars={
        "AWS_ACCESS_KEY_ID": "your-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret",
    }
)
```

#### Advanced Features: Dynamic Allocation and Volumes

```python
from kubeflow.spark import BatchSparkClient, OperatorBackendConfig

config = OperatorBackendConfig(namespace="default")
client = BatchSparkClient(backend_config=config)

response = client.submit_application(
    app_name="advanced-job",
    main_application_file="local:///app/job.py",
    spark_version="4.0.0",
    driver_cores=2,
    driver_memory="4g",
    executor_cores=2,
    executor_memory="8g",
    num_executors=3,

    # Enable dynamic allocation
    enable_dynamic_allocation=True,
    initial_executors=2,
    min_executors=1,
    max_executors=10,

    # Configure volumes
    volumes=[{
        "name": "data-volume",
        "persistentVolumeClaim": {"claimName": "my-pvc"}
    }],
    driver_volume_mounts=[{
        "name": "data-volume",
        "mountPath": "/data"
    }],

    # Node selector and tolerations
    node_selector={"node-type": "compute"},
    tolerations=[{
        "key": "spark",
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule"
    }],
)
```

### Interactive Sessions

#### Basic Interactive Session

```python
from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

# Connect to Spark Connect server
config = ConnectBackendConfig(
    connect_url="sc://spark-cluster.default.svc:15002",
    use_ssl=True,
)
client = SparkSessionClient(backend_config=config)

# Create interactive session
session = client.create_session(app_name="data-exploration")

# Use standard PySpark DataFrame API
df = session.sql("SELECT * FROM sales WHERE date >= '2024-01-01'")
result = df.groupBy("product").sum("amount").collect()

for row in result:
    print(f"{row.product}: {row['sum(amount)']}")

# Cleanup
session.close()
```

#### Notebook Workflow

```python
from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

# Cell 1: Setup
config = ConnectBackendConfig(connect_url="sc://spark-cluster:15002")
client = SparkSessionClient(backend_config=config)
session = client.create_session("notebook-analysis")

# Cell 2: Load data
df = session.read.parquet("s3a://bucket/data/")
df.show()

# Cell 3: Feature engineering
features = df.withColumn("spend_per_year", df.spend_total / df.age)
features.describe().show()

# Cell 4: Export results
session.export_to_pipeline_artifact(features, "/outputs/features.parquet")

# Cell 5: Cleanup
session.close()
```

#### Session Management

```python
from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

config = ConnectBackendConfig(connect_url="sc://spark-cluster:15002")
client = SparkSessionClient(backend_config=config)

# List all active sessions
sessions = client.list_sessions()
for session_info in sessions:
    print(f"Session: {session_info.session_id}")
    print(f"  App: {session_info.app_name}")
    print(f"  Queries executed: {session_info.metrics.queries_executed}")

# Get specific session status
session = client.create_session("my-app")
info = client.get_session_status(session.session_id)
print(f"Session state: {info.state}")

# Close session
client.close_session(session.session_id, release=True)
```

## API Reference

### BatchSparkClient

Client for managing batch Spark applications.

#### Constructor

```python
BatchSparkClient(backend_config: Union[OperatorBackendConfig, GatewayBackendConfig, None] = None)
```

#### Methods

**submit_application(...) → SparkApplicationResponse**
- Submit a new Spark application
- Returns submission ID and initial status

**get_status(submission_id) → ApplicationStatus**
- Get current status of an application
- Returns state, app ID, executor info, timestamps

**wait_for_completion(submission_id, timeout=3600, polling_interval=10) → ApplicationStatus**
- Block until application completes
- Returns final status

**get_logs(submission_id, executor_id=None, follow=False) → Iterator[str]**
- Stream application logs
- Can retrieve driver or specific executor logs

**list_applications(namespace=None, labels=None) → List[ApplicationStatus]**
- List applications with optional filtering
- Supports namespace and label filters

**delete_application(submission_id) → Dict**
- Delete an application
- Stops running application and cleans up resources

**wait_for_pod_ready(submission_id, executor_id=None, timeout=300) → bool**
- Wait for driver or executor pod to be ready
- Only available with OperatorBackend

### SparkSessionClient

Client for managing interactive Spark sessions.

#### Constructor

```python
SparkSessionClient(backend_config: ConnectBackendConfig)
```

#### Methods

**create_session(app_name, **kwargs) → ManagedSparkSession**
- Create a new Spark Connect session
- Returns managed session with PySpark API access

**get_session_status(session_id) → SessionInfo**
- Get status and metadata of a session
- Returns state, metrics, and session details

**list_sessions() → List[SessionInfo]**
- List all active Spark Connect sessions

**close_session(session_id, release=True) → Dict**
- Close a session and release resources

### ManagedSparkSession

Wrapper around PySpark SparkSession with Kubeflow enhancements.

#### Properties

- **session_id**: Unique session identifier
- **app_name**: Application name
- **spark**: Access to underlying PySpark SparkSession

#### Methods

**sql(query) → DataFrame**
- Execute SQL query and return DataFrame

**read → DataFrameReader**
- Access DataFrameReader for reading data sources

**readStream → DataStreamReader**
- Access DataStreamReader for streaming sources

**upload_artifacts(*paths, pyfile=False)**
- Upload JARs or Python files to session

**get_metrics() → SessionMetrics**
- Get session metrics (queries executed, artifacts uploaded)

**close(release=True)**
- Close the session

### Backend Configurations

#### OperatorBackendConfig

Configuration for Kubernetes Spark Operator backend.

```python
from kubeflow.spark import OperatorBackendConfig

config = OperatorBackendConfig(
    namespace="default",
    context=None,
    service_account="spark-operator-spark",
    image_pull_policy="IfNotPresent",
    default_spark_image="docker.io/library/spark",
    enable_monitoring=True,
    enable_ui=True,
    timeout=60,
)
```

#### GatewayBackendConfig

Configuration for REST Gateway backend.

```python
from kubeflow.spark import GatewayBackendConfig

config = GatewayBackendConfig(
    gateway_url="http://gateway:8080",
    user="myuser",
    password="mypassword",
    timeout=30,
    verify_ssl=True,
)
```

#### ConnectBackendConfig

Configuration for Spark Connect backend.

```python
from kubeflow.spark import ConnectBackendConfig

config = ConnectBackendConfig(
    connect_url="sc://spark-cluster.default.svc:15002",
    token="bearer-token",  # Optional
    use_ssl=True,
)
```

## Choosing the Right Client

### Use BatchSparkClient when:
- Running scheduled ETL pipelines
- Submitting production batch jobs
- Integrating with CI/CD workflows
- Need dynamic allocation and auto-scaling
- Running jobs as Kubernetes CRDs

### Use SparkSessionClient when:
- Performing interactive data exploration
- Working in Jupyter or IPython notebooks
- Iterative development and testing
- Need immediate feedback from queries
- Connecting to remote Spark clusters

## Examples

The `examples/spark/` directory contains comprehensive examples:

**Batch Examples:**
- `01_hello_spark_pi.py`: Basic Spark Pi calculation
- `02_csv_data_analysis.py`: CSV data processing
- `04_etl_pipeline_simple.py`: ETL pipeline example
- `05_scheduled_batch_job.py`: Scheduled job pattern
- `06_autoscaling_dynamic_allocation.py`: Dynamic allocation

**Interactive Session Examples:**
- `07_spark_connect_interactive.py`: Interactive data analysis
- `ipython_spark_connect_demo.py`: IPython integration
- `ipython_spark_connect_shell.py`: Interactive shell

Run examples:

```bash
cd examples/spark

# Batch example
python 01_hello_spark_pi.py

# Interactive session example
python 07_spark_connect_interactive.py
```

## Testing

### Setup Test Environment

Use the provided script to set up a Kind cluster with Spark Operator:

```bash
cd examples/spark
./setup_test_environment.sh
```

This will:
1. Create a Kind cluster
2. Install Spark Operator
3. Configure RBAC and service accounts
4. Verify the installation

### Run Integration Tests

```bash
python test_spark_client_integration.py
```

### Cleanup

```bash
kind delete cluster --name spark-test
```

## Monitoring and Debugging

### Access Spark UI

Port forward to Spark UI:
```bash
kubectl port-forward -n default svc/spark-ui 4040:4040
```

Open in browser: http://localhost:4040

### View Application Logs

Using BatchSparkClient:
```python
# Stream driver logs
for line in client.get_logs(submission_id):
    print(line)

# Get executor logs
for line in client.get_logs(submission_id, executor_id="1"):
    print(line)
```

Using kubectl:
```bash
# Driver logs
kubectl logs <app-name>-driver -n default

# Executor logs
kubectl logs <app-name>-exec-1 -n default
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from kubeflow.spark import BatchSparkClient
client = BatchSparkClient()
```

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'pyspark'**

For SparkSessionClient, install PySpark with Connect support:
```bash
pip install 'pyspark[connect]>=3.4.0'
```

**2. SparkApplication not being created**

Check Spark Operator is running:
```bash
kubectl get pods -n spark-operator
```

**3. Cannot connect to Spark Connect server**

Verify the server is running and accessible:
```bash
kubectl get svc -n default | grep spark
kubectl port-forward svc/spark-connect 15002:15002
```

**4. Permission denied**

Verify service account permissions:
```bash
kubectl auth can-i create pods --as=system:serviceaccount:default:spark-operator-spark
```

## Comparison with Trainer Client

| Aspect | Trainer Client | Spark Client |
|--------|---------------|--------------|
| **CRD** | TrainJob | SparkApplication |
| **Operator** | Training Operator | Spark Operator |
| **Client Classes** | TrainingClient | BatchSparkClient, SparkSessionClient |
| **Backends** | Kubernetes, LocalProcess | Operator, Gateway, Connect |
| **Workload Types** | Batch training jobs | Batch jobs + interactive sessions |
| **API Style** | train(), list_jobs() | submit_application(), create_session() |

Both clients provide:
- Backend abstraction for flexibility
- Kubernetes-native CRD management
- Status monitoring with polling
- Log streaming capabilities
- Context manager support

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Apache License 2.0

## References

- [Kubeflow Spark Operator](https://github.com/kubeflow/spark-operator)
- [Apache Spark on Kubernetes](https://spark.apache.org/docs/latest/running-on-kubernetes.html)
- [Spark Connect](https://spark.apache.org/docs/latest/spark-connect-overview.html)
- [Kubeflow Training Client](https://github.com/kubeflow/training-operator)

## Support

For issues and questions:
- GitHub Issues: [kubeflow/sdk](https://github.com/kubeflow/sdk/issues)
- Slack: #kubeflow-spark
- Mailing List: kubeflow-discuss@googlegroups.com
