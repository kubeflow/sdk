# Spark Examples

This directory contains examples for using the Kubeflow Spark SDK.

## Examples

- **spark_connect_simple.py** - Basic SparkClient usage with simple API
- **spark_advanced_options.py** - Advanced configuration with Driver/Executor objects
- **demo_existing_sparkconnect.py** - Connect to existing SparkConnect cluster
- **test_connect_url.py** - Test URL-based connection to Spark Connect
- **iceberg_minio.py** - Local PySpark SparkSession with Apache Iceberg and MinIO

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

Demonstrates reading and writing Iceberg tables using MinIO as S3-compatible storage.

### Prerequisites

- Docker and Docker Compose

### Setup

Start MinIO and the Iceberg REST catalog:

```bash
docker compose -f examples/spark/docker-compose-iceberg-minio.yml up -d
```

Wait a few seconds for the services to be ready, then install dependencies:

```bash
pip install pyspark==3.5.0
```

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

The compose file also accepts overrides for `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`,
and `AWS_REGION`. The defaults are intended for local development only.

### Teardown

```bash
docker compose -f examples/spark/docker-compose-iceberg-minio.yml down
```
