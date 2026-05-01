# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test SparkClient.connect(url=...) with a local Spark Connect server.

Prerequisites:
    Start a Spark Connect server using Docker:

    docker run -d --name spark-connect -p 15002:15002 \\
        apache/spark:3.5.0 \\
        /opt/spark/sbin/start-connect-server.sh \\
        --packages org.apache.spark:spark-connect_2.12:3.5.0

Run:
    uv run python examples/spark/test_connect_url.py

    # Or with custom URL:
    SPARK_CONNECT_URL=sc://localhost:15002 uv run python examples/spark/test_connect_url.py
"""

import os
import sys

DEFAULT_URL = "sc://localhost:15002"


def test_spark_connect_url() -> bool:
    """Test SparkClient.connect() with a Spark Connect URL."""
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.spark import SparkClient

    url = os.environ.get("SPARK_CONNECT_URL", DEFAULT_URL)
    print(f"Connecting to Spark Connect server at: {url}")

    client = SparkClient(backend_config=KubernetesBackendConfig(namespace="default"))

    try:
        spark = client.connect(base_url=url)
        print("Connected to Spark Connect server")

        print("\n--- Test 1: Create DataFrame ---")
        data = [(1, "Alice", 28), (2, "Bob", 35), (3, "Charlie", 42)]
        df = spark.createDataFrame(data, ["id", "name", "age"])
        print(f"Created DataFrame with {df.count()} rows")
        df.show()

        print("\n--- Test 2: DataFrame Transformations ---")
        filtered = df.filter(df.age > 30)
        print(f"Filtered to {filtered.count()} rows where age > 30")
        filtered.show()

        print("\n--- Test 3: SQL Operations ---")
        df.createOrReplaceTempView("people")
        result = spark.sql("SELECT name, age FROM people ORDER BY age DESC")
        result.show()

        print("\n--- Test 4: Aggregations ---")
        from pyspark.sql import functions as func

        agg_result = df.agg(func.avg("age").alias("avg_age"), func.max("age").alias("max_age"))
        agg_result.show()

        spark.stop()
        print("\nAll tests passed!")
        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Spark Connect server is running:")
        print("   docker run -d --name spark-connect -p 15002:15002 \\")
        print("       apache/spark:3.5.0 \\")
        print("       /opt/spark/sbin/start-connect-server.sh \\")
        print("       --packages org.apache.spark:spark-connect_2.12:3.5.0")
        print("\n2. Check if container is running: docker ps | grep spark-connect")
        print("3. Check container logs: docker logs spark-connect")
        return False


if __name__ == "__main__":
    success = test_spark_connect_url()
    sys.exit(0 if success else 1)
