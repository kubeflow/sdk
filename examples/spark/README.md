# Kubeflow Spark SDK Examples

This directory contains examples demonstrating the Kubeflow Spark SDK capabilities
for running Apache Spark on Kubernetes. Examples are organized by complexity level
and persona (Data Scientist, Data Engineer, ML Engineer).

## Quick Start

```bash
# 1. Setup test environment (creates Kind cluster with Spark Operator)
./setup_test_environment.sh

# 2. Install SDK dependencies
pip install -r requirements-core.txt

# 3. Run your first example
python 01_hello_spark_pi.py
```

## Example Overview

### By Persona

| Persona | Examples | Key Features |
|---------|----------|--------------|
| **Data Scientist** | 00, 03, 08 | Interactive sessions, quick exploration, auto-provisioning |
| **Data Engineer** | 01, 02, 04, 05, 06 | Batch jobs, ETL pipelines, scheduling, scaling |
| **ML Engineer** | 09, 10 | Feature engineering, Trainer integration |

### By Complexity

| Level | Examples | Description |
|-------|----------|-------------|
| **Beginner** | 00, 01, 02 | First steps, simple batch jobs |
| **Intermediate** | 03, 04, 05, 06 | Interactive sessions, pipelines, scheduling |
| **Advanced** | 07, 08, 09, 10 | Auto-provisioning, GPU, ML integration |

## Examples List

### Level 0: Quickstart (No Kubernetes Required)

- **`00_local_quickstart.py`** - Run Spark locally without Kubernetes
  - Best for: Testing, development, learning PySpark
  - Time: 1 minute
  - No prerequisites

### Level 1: Beginner (Batch Jobs)

- **`01_hello_spark_pi.py`** - Your first Spark job (Calculate Pi)
  - Best for: Data Scientists new to Spark
  - Time: 2-3 minutes
  - Demonstrates: Submit job, monitor, get logs

- **`02_csv_data_analysis.py`** - CSV data analysis
  - Best for: Data Scientists analyzing tabular data
  - Time: 2-3 minutes
  - Demonstrates: DataFrame operations, aggregations

### Level 2: Intermediate

- **`03_interactive_dataframe_exploration.py`** - Interactive data exploration
  - Best for: Data Scientists doing ad-hoc analysis
  - Time: 5 minutes
  - Demonstrates: Spark Connect sessions, SQL queries

- **`04_etl_pipeline_simple.py`** - Simple ETL pipeline
  - Best for: Data Engineers building pipelines
  - Time: 3-4 minutes
  - Demonstrates: Extract, transform, load patterns

- **`05_scheduled_batch_job.py`** - Scheduled recurring jobs
  - Best for: Data Engineers with recurring workloads
  - Time: 4-5 minutes
  - Demonstrates: ScheduledSparkApplication CRD

- **`06_autoscaling_dynamic_allocation.py`** - Dynamic executor scaling
  - Best for: Data Engineers optimizing resources
  - Time: 4-5 minutes
  - Demonstrates: Auto-scaling, resource efficiency

### Level 3: Advanced

- **`07_spark_connect_interactive.py`** - Spark Connect deep-dive
  - Best for: Advanced interactive workflows
  - Time: 5 minutes
  - Demonstrates: Full Spark Connect features

- **`08_auto_provision_spark_connect.py`** - Auto-provision Spark Connect server
  - Best for: Data Scientists wanting simple setup
  - Time: 3-4 minutes
  - Demonstrates: SparkClient auto-provisioning

- **`09_spark_trainer_integration.py`** - Spark + Kubeflow Trainer
  - Best for: ML Engineers doing feature engineering
  - Time: 5-6 minutes
  - Demonstrates: Feature prep → Model training workflow

## Client Selection Guide

The SDK provides multiple clients for different use cases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Which Client Should I Use?                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Need interactive session?                                           │
│     │                                                                │
│     ├── YES → Have existing Spark Connect server?                    │
│     │           │                                                    │
│     │           ├── YES → SparkSessionClient (connect to existing)   │
│     │           │                                                    │
│     │           └── NO → SparkClient (auto-provisions server)        │
│     │                                                                │
│     └── NO → Need to submit batch job?                               │
│               │                                                      │
│               └── YES → BatchSparkClient                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### SparkClient (Unified, Auto-Provisioning)

```python
from kubeflow.spark import SparkClient

# Auto-creates Spark Connect server, deletes on exit
with SparkClient.builder().num_executors(5).build() as client:
    spark = client.session()
    df = spark.read.parquet("s3a://data/")
    df.show()
```

### BatchSparkClient (Batch Jobs)

```python
from kubeflow.spark import BatchSparkClient

client = BatchSparkClient()
response = client.submit_application(
    main_application_file="s3a://bucket/etl.py",
    num_executors=10,
)
client.wait_for_job_status(response.submission_id)
```

### SparkSessionClient (Connect to Existing Server)

```python
from kubeflow.spark import SparkSessionClient, ConnectBackendConfig

config = ConnectBackendConfig(connect_url="sc://spark-cluster:15002")
client = SparkSessionClient(backend_config=config)
session = client.create_session(app_name="analysis")
```

## Configuration Options

### Resource Configuration

```python
# Batch job resources
client.submit_application(
    driver_cores=2,
    driver_memory="4g",
    executor_cores=4,
    executor_memory="8g",
    num_executors=10,
)

# SparkClient builder (auto-provision)
SparkClient.builder() \
    .server(cores=2, memory="4g") \
    .executor(cores=4, memory="8g") \
    .num_executors(10) \
    .build()
```

### Dynamic Allocation

```python
from kubeflow.spark import DynamicAllocation

dyn_alloc = DynamicAllocation(
    enabled=True,
    initial_executors=2,
    min_executors=1,
    max_executors=20,
    shuffle_tracking_enabled=True,
)

client.submit_application(
    dynamic_allocation=dyn_alloc,
    ...
)
```

### Spark Configuration

```python
# Via submit_application
client.submit_application(
    spark_conf={
        "spark.sql.shuffle.partitions": "200",
        "spark.sql.adaptive.enabled": "true",
    },
    hadoop_conf={
        "fs.s3a.endpoint": "s3.amazonaws.com",
    },
)

# Via SparkClient builder
SparkClient.builder() \
    .spark_conf("spark.sql.shuffle.partitions", "200") \
    .hadoop_conf("fs.s3a.endpoint", "s3.amazonaws.com") \
    .build()
```

## S3/MinIO Integration

For examples using S3 storage:

```bash
# Setup MinIO (S3-compatible storage)
./setup_minio.sh

# Run S3 examples
python 02_csv_data_analysis_s3.py
```

## Troubleshooting

### Common Issues

1. **"spark-operator-spark service account not found"**
   ```bash
   kubectl create serviceaccount spark-operator-spark
   kubectl create rolebinding spark-role --clusterrole=edit --serviceaccount=default:spark-operator-spark
   ```

2. **"Connection refused to Spark Connect"**
   - Ensure Spark Connect server is running
   - Check service name and port
   - For auto-provisioning, wait for server to be Ready

3. **"Image pull failed"**
   - Check image name and tag
   - Ensure cluster has internet access
   - Consider using local images for Kind

4. **"PySpark not found"**
   ```bash
   pip install 'pyspark[connect]>=3.4.0'
   ```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Running Without Kubernetes

For local development without Kubernetes:

```python
# Local PySpark (no SDK needed)
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("LocalTest") \
    .getOrCreate()

df = spark.range(100)
df.show()
```

See `00_local_quickstart.py` for a complete local example.

## Next Steps

1. Start with `00_local_quickstart.py` or `01_hello_spark_pi.py`
2. Try interactive exploration with `03_interactive_dataframe_exploration.py`
3. Learn auto-provisioning with `08_auto_provision_spark_connect.py`
4. Build ETL pipelines with `04_etl_pipeline_simple.py`
5. Integrate with ML workflows using `09_spark_trainer_integration.py`

## References

- [Kubeflow Spark SDK Documentation](../../kubeflow/spark/README.md)
- [Spark Operator Documentation](https://github.com/kubeflow/spark-operator)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Spark Connect Guide](https://spark.apache.org/docs/latest/spark-connect-overview.html)

