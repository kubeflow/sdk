# Local Spark Connect Server Testing Guide

This guide shows how to run a local Spark Connect server and test `SparkClient.connect(url=...)`.

## Prerequisites

- Docker installed and running
- Python 3.9+ with `pyspark` installed
- The kubeflow SDK installed (`uv sync`)

## Option 1: Docker-based Spark Connect Server (Recommended)

### Start Spark Connect Server

```bash
# Pull and run the official Spark image with Connect server
docker run -d --name spark-connect \
  -p 15002:15002 \
  -e SPARK_NO_DAEMONIZE=true \
  apache/spark:3.5.0 \
  /opt/spark/sbin/start-connect-server.sh \
  --packages org.apache.spark:spark-connect_2.12:3.5.0
```

### Verify Server is Running

```bash
# Check container logs
docker logs spark-connect

# You should see: "SparkConnectServer started on port 15002"
```

### Test with SparkClient

```bash
cd sdk

# Run the test
SPARK_CONNECT_URL="sc://localhost:15002" uv run python -c "
from kubeflow.spark import SparkClient
from kubeflow.common.types import KubernetesBackendConfig

# Create client (backend not used when connecting to URL directly)
client = SparkClient(backend_config=KubernetesBackendConfig(namespace='default'))

# Connect to existing Spark Connect server
spark = client.connect(url='sc://localhost:15002')

# Test with a simple DataFrame operation
df = spark.createDataFrame([(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')], ['id', 'name'])
print('DataFrame created successfully!')
print(f'Row count: {df.count()}')
df.show()

# Run SQL
df.createOrReplaceTempView('people')
result = spark.sql('SELECT * FROM people WHERE id > 1')
result.show()

spark.stop()
print('Test completed successfully!')
"
```

### Cleanup

```bash
docker stop spark-connect && docker rm spark-connect
```

---

## Option 2: Local Spark Installation

### Install Spark Locally

```bash
# macOS with Homebrew
brew install apache-spark

# Or download from https://spark.apache.org/downloads.html
```

### Start Spark Connect Server

```bash
# Start the server (requires SPARK_HOME to be set)
$SPARK_HOME/sbin/start-connect-server.sh \
  --packages org.apache.spark:spark-connect_2.12:3.5.0

# The server runs on port 15002 by default
```

### Test with SparkClient

Same as Option 1 above.

### Stop Server

```bash
$SPARK_HOME/sbin/stop-connect-server.sh
```

---

## Option 3: Kubernetes with Port-Forward

If you have a Spark Connect session running in Kubernetes:

```bash
# Get the service name
kubectl get svc -n spark-test | grep spark-connect

# Port-forward to access locally
kubectl port-forward svc/my-spark-session-svc 15002:15002 -n spark-test
```

Then test with the same SparkClient code above.

---

## Integration Test Script

A ready-to-run test script:

```bash
# Save as: test_spark_connect_local.py
cd sdk

uv run python kubeflow/spark/examples/test_connect_url.py
```

expecting Output:

Connecting to Spark Connect server at: sc://localhost:15002


--- Test 1: Create DataFrame ---
Created DataFrame with 3 rows
+---+-------+---+
| id|   name|age|
+---+-------+---+
|  1|  Alice| 28|
|  2|    Bob| 35|
|  3|Charlie| 42|
+---+-------+---+

--- Test 2-4: Transformations, SQL, Aggregations ---
All tests passed!


---

## Troubleshooting

### Connection Refused
```
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
    status = StatusCode.UNAVAILABLE
```
**Solution:** Ensure the Spark Connect server is running and accessible on port 15002.

### Package Not Found
```
java.lang.ClassNotFoundException: org.apache.spark.sql.connect.service.SparkConnectServer
```
**Solution:** Add the `--packages org.apache.spark:spark-connect_2.12:3.5.0` flag when starting the server.

### Version Mismatch
Ensure your PySpark version matches the Spark Connect server version:
```bash
uv run python -c "import pyspark; print(pyspark.__version__)"
```

---

## Quick Start Script

Copy-paste this to test immediately:

```bash
# Start server
docker run -d --name spark-connect -p 15002:15002 \
  apache/spark:3.5.0 \
  /opt/spark/sbin/start-connect-server.sh \
  --packages org.apache.spark:spark-connect_2.12:3.5.0

# Wait for startup
sleep 10

# Test
cd sdk
uv run python kubeflow/spark/examples/test_connect_url.py

# Cleanup
docker stop spark-connect && docker rm spark-connect
```
