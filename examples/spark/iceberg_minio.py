#!/usr/bin/env python3

# Copyright The Kubeflow Authors.
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

"""Example: SparkClient with Apache Iceberg and MinIO (S3-compatible storage).

This example demonstrates how to use the Kubeflow Spark SDK with:
- Apache Iceberg as the table format
- MinIO as the S3-compatible object storage backend
- Iceberg REST catalog for table metadata management

The example creates a Spark Connect session through ``kubeflow.spark.SparkClient``
so the Iceberg path is exercised end-to-end through the SDK.

This is a local development example. For production use, replace the
MinIO endpoint and credentials with your actual S3/GCS/ABS configuration.

Prerequisites:
    # Start MinIO
    docker run -d -p 9000:9000 -p 9001:9001 \\
        -e MINIO_ROOT_USER=minioadmin \\
        -e MINIO_ROOT_PASSWORD=minioadmin \\
        --name minio \\
        minio/minio server /data --console-address ":9001"

    # Create warehouse bucket (via UI at http://localhost:9001 or mc CLI)

    # Start Iceberg REST catalog
    docker run -d -p 8181:8181 \\
        --name iceberg-rest \\
        -e AWS_ACCESS_KEY_ID=minioadmin \\
        -e AWS_SECRET_ACCESS_KEY=minioadmin \\
        -e AWS_REGION=us-east-1 \\
        -e CATALOG_WAREHOUSE=s3://warehouse/ \\
        -e CATALOG_IO__IMPL=org.apache.iceberg.aws.s3.S3FileIO \\
        -e CATALOG_S3_ENDPOINT=http://minio:9000 \\
        -e CATALOG_S3_PATH__STYLE__ACCESS=true \\
        tabulario/iceberg-rest

Usage:
    uv run python examples/spark/iceberg_minio.py
"""

import os
import uuid

from pyspark.sql import SparkSession

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import Name, SparkClient

# MinIO / S3 credentials — replace with your own for production
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
ICEBERG_REST_URI = os.environ.get("ICEBERG_REST_URI", "http://localhost:8181")
WAREHOUSE = os.environ.get("ICEBERG_WAREHOUSE", "s3://warehouse/")


def _backend_config(namespace_default: str = "default") -> KubernetesBackendConfig:
    """Backend config; uses SPARK_TEST_NAMESPACE in CI."""
    return KubernetesBackendConfig(
        namespace=os.environ.get("SPARK_TEST_NAMESPACE", namespace_default)
    )


def _session_name(base: str) -> str:
    """Use a unique session name in E2E in-cluster runs to avoid conflicts."""
    if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") == "1":
        return f"{base}-{uuid.uuid4().hex[:8]}"
    return base


def create_spark_session() -> tuple[SparkClient, SparkSession, str]:
    """Create a Spark Connect session configured for Iceberg + MinIO.

    Key configurations:
    - spark.jars.packages: pulls Iceberg runtime and AWS SDK v2 from Maven
    - spark.sql.extensions: enables Iceberg SQL extensions (time travel, etc.)
    - spark.sql.catalog.*: configures the Iceberg REST catalog
    - spark.sql.catalog.lakehouse.io-impl and .s3.*: configure Iceberg S3FileIO for MinIO access

    Note: AWS SDK v2 (software.amazon.awssdk) is required for Iceberg 1.9.x.
    """
    # Set local defaults for the AWS env vars only when this process has not
    # already configured credentials or region.
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", MINIO_ACCESS_KEY)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", MINIO_SECRET_KEY)

    client = SparkClient(backend_config=_backend_config())
    session_name = _session_name("kubeflow-iceberg-minio")

    spark = client.connect(
        spark_conf={
            # Iceberg runtime + a minimal AWS SDK v2 set (required for S3FileIO in Iceberg 1.9.x)
            # Use direct JAR URLs so the server can fetch artifacts without
            # relying on Ivy/package resolution (helpful in air-gapped CI).
            "spark.jars": (
                "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.0_2.13/1.9.1/iceberg-spark-runtime-4.0_2.13-1.9.1.jar,"
                "https://repo1.maven.org/maven2/software/amazon/awssdk/s3/2.26.24/s3-2.26.24.jar,"
                "https://repo1.maven.org/maven2/software/amazon/awssdk/url-connection-client/2.26.24/url-connection-client-2.26.24.jar"
            ),
            # Pass AWS region to driver JVM.
            "spark.driver.extraJavaOptions": "-Daws.region=us-east-1",
            # Enables: CREATE TABLE ... USING iceberg, time travel, etc.
            "spark.sql.extensions": (
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
            ),
            # Iceberg REST catalog.
            "spark.sql.catalog.lakehouse": "org.apache.iceberg.spark.SparkCatalog",
            "spark.sql.catalog.lakehouse.type": "rest",
            "spark.sql.catalog.lakehouse.uri": ICEBERG_REST_URI,
            "spark.sql.catalog.lakehouse.warehouse": WAREHOUSE,
            # Use S3FileIO for reading/writing Iceberg data files to MinIO.
            "spark.sql.catalog.lakehouse.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
            "spark.sql.catalog.lakehouse.s3.endpoint": MINIO_ENDPOINT,
            "spark.sql.catalog.lakehouse.s3.path-style-access": "true",
            "spark.sql.catalog.lakehouse.s3.region": "us-east-1",
        },
        options=[Name(session_name)],
        timeout=180 if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") == "1" else 300,
        connect_timeout=60 if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") == "1" else 120,
    )
    return client, spark, session_name


def run_iceberg_example(spark: SparkSession) -> None:
    """Run a basic Iceberg read/write example.

    Demonstrates:
    1. Creating an Iceberg namespace and table
    2. Writing data to the table (stored as Parquet in MinIO)
    3. Reading data back via the Iceberg catalog
    """
    print("\n--- Creating namespace and table ---")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lakehouse.demo")
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS lakehouse.demo.users (
            id   BIGINT,
            name STRING
        ) USING iceberg
        """
    )

    print("\n--- Writing data ---")
    spark.sql("INSERT INTO lakehouse.demo.users VALUES (1, 'Alice'), (2, 'Bob')")

    print("\n--- Reading data ---")
    df = spark.sql("SELECT * FROM lakehouse.demo.users")
    df.show()

    print("\n--- Table metadata ---")
    spark.sql("DESCRIBE EXTENDED lakehouse.demo.users").show(truncate=False)


def main() -> None:
    print("=" * 60)
    print("KUBEFLOW SPARKCLIENT — Iceberg + MinIO Example")
    print("=" * 60)

    client, spark, session_name = create_spark_session()

    try:
        run_iceberg_example(spark)
        print("\n[SUCCESS] Iceberg + MinIO example complete!")
    except Exception as e:
        print(f"\n[FAILED] {e}")
        raise
    finally:
        spark.stop()
        client.delete_session(session_name)


if __name__ == "__main__":
    main()
