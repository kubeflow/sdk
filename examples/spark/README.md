# Spark Examples

This directory contains examples for using the Kubeflow Spark SDK.

## Examples

- **spark_connect_simple.py** - Basic SparkClient usage with simple API
- **spark_advanced_options.py** - Advanced configuration with Driver/Executor objects
- **demo_existing_sparkconnect.py** - Connect to existing SparkConnect cluster
- **test_connect_url.py** - Test URL-based connection to Spark Connect
- **iceberg_minio.py** - SparkClient example with Apache Iceberg and MinIO

## Prerequisites

Install spark dependencies:
```bash
uv pip install kubeflow[spark]
```

## Running Examples

```bash
# Run from repository root
uv run python examples/spark/spark_connect_simple.py
```

## Iceberg + MinIO Example

Demonstrates reading and writing Iceberg tables using MinIO as S3-compatible storage through the Kubeflow Spark SDK.

### Prerequisites

- A Kind-based Spark e2e setup with the optional Iceberg/MinIO services enabled

### Setup

Enable the Iceberg/MinIO services in the Spark e2e cluster bootstrap:

```bash
SPARK_E2E_ENABLE_ICEBERG_MINIO=1 make test-e2e-setup-cluster
```

Wait for the setup to finish. The Spark dependency comes from `pyproject.toml` via `kubeflow[spark]`, so there is no separate PySpark install step.

### Run

```bash
python examples/spark/iceberg_minio.py
```

You should see:

```text
--- Reading data ---
+---+-----+
| id| name|
+---+-----+
|  1|Alice|
|  2|  Bob|
+---+-----+
[SUCCESS] Iceberg + MinIO example complete!
```

### Configuration

The example can be configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `ICEBERG_REST_URI` | `http://localhost:8181` | Iceberg REST catalog URI |
| `ICEBERG_WAREHOUSE` | `s3://warehouse/` | Warehouse location |

The e2e setup also accepts overrides for `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`,
and `AWS_REGION`. The defaults are intended for local development only.

### E2E

The Iceberg example is exercised by the Spark e2e test harness in `test/e2e/spark/test_spark_examples.py` when the Iceberg service endpoints are available in the environment.

### Teardown

```bash
bash hack/e2e-setup-cluster.sh --delete
```
