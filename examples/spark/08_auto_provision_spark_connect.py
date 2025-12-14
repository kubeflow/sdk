#!/usr/bin/env python3
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
Title: Auto-Provision Spark Connect Server
Level: 3 (Advanced)
Target Audience: Data Scientists wanting simple Spark setup
Time to Run: ~3-4 minutes

Description:
This example demonstrates the SparkClient's auto-provisioning feature, which
automatically creates a Spark Connect server on Kubernetes when you need an
interactive session. No manual server setup required!

This is the simplest way to get started with interactive Spark on Kubernetes.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Spark Operator with SparkConnect CRD support
- PySpark: pip install 'pyspark[connect]>=3.4.0'

What You'll Learn:
- How to use SparkClient.builder() for simple configuration
- Auto-provisioning of Spark Connect servers
- Interactive DataFrame operations
- Automatic cleanup on exit

Best For:
- Data Scientists who want simple Spark setup
- Quick interactive analysis without infrastructure concerns
- Notebook-style workflows
"""

import logging
import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_interactive_analysis(spark) -> None:
    """Run interactive data analysis with Spark.

    Args:
        spark: SparkSession from the auto-provisioned server
    """
    from pyspark.sql.functions import avg, col, count, sum as _sum

    logger.info("=" * 60)
    logger.info("Running Interactive Data Analysis")
    logger.info("=" * 60)

    # Create sample e-commerce data
    logger.info("\nCreating sample e-commerce dataset...")
    orders = [
        (1, "2024-01-15", "alice", "laptop", 1, 1299.99),
        (2, "2024-01-15", "bob", "phone", 2, 899.99),
        (3, "2024-01-16", "alice", "headphones", 1, 199.99),
        (4, "2024-01-16", "carol", "laptop", 1, 1299.99),
        (5, "2024-01-17", "bob", "tablet", 1, 599.99),
        (6, "2024-01-17", "dave", "phone", 1, 899.99),
        (7, "2024-01-18", "eve", "laptop", 2, 1299.99),
        (8, "2024-01-18", "alice", "tablet", 1, 599.99),
    ]
    columns = ["order_id", "date", "customer", "product", "quantity", "price"]
    df = spark.createDataFrame(orders, columns)

    logger.info(f"Created {df.count()} orders")

    # Show raw data
    logger.info("\nRaw Order Data:")
    df.show()

    # Analysis 1: Customer spending
    logger.info("\n" + "=" * 40)
    logger.info("Analysis 1: Customer Spending Summary")
    logger.info("=" * 40)

    customer_spending = (
        df.withColumn("total", col("quantity") * col("price"))
        .groupBy("customer")
        .agg(
            count("order_id").alias("num_orders"),
            _sum("total").alias("total_spent"),
            avg("total").alias("avg_order_value"),
        )
        .orderBy(col("total_spent").desc())
    )
    customer_spending.show()

    # Analysis 2: Product popularity
    logger.info("=" * 40)
    logger.info("Analysis 2: Product Popularity")
    logger.info("=" * 40)

    product_stats = (
        df.groupBy("product")
        .agg(
            _sum("quantity").alias("units_sold"),
            _sum(col("quantity") * col("price")).alias("revenue"),
        )
        .orderBy(col("revenue").desc())
    )
    product_stats.show()

    # Analysis 3: Daily trends (SQL)
    logger.info("=" * 40)
    logger.info("Analysis 3: Daily Sales Trend (SQL)")
    logger.info("=" * 40)

    df.createOrReplaceTempView("orders")
    daily_trend = spark.sql("""
        SELECT
            date,
            COUNT(DISTINCT customer) as unique_customers,
            SUM(quantity) as units_sold,
            SUM(quantity * price) as daily_revenue
        FROM orders
        GROUP BY date
        ORDER BY date
    """)
    daily_trend.show()

    logger.info("\n✅ Interactive analysis completed!")


def main():
    """Main example: Auto-provision Spark Connect and run analysis."""

    print("=" * 80)
    print("EXAMPLE 08: Auto-Provision Spark Connect Server")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. SparkClient.builder() for simple configuration")
    print("  2. Auto-provisioning of Spark Connect server on Kubernetes")
    print("  3. Interactive DataFrame operations")
    print("  4. Automatic cleanup on exit")
    print()
    print("Key Feature: No manual Spark Connect server setup needed!")
    print()

    try:
        from kubeflow.spark import SparkClient
    except ImportError as e:
        print(f"ERROR: Could not import Kubeflow Spark SDK: {e}")
        print("Make sure you're in the correct directory or SDK is installed")
        sys.exit(1)

    # Configuration
    namespace = os.getenv("SPARK_NAMESPACE", "default")
    kube_context = os.getenv("KUBE_CONTEXT", "kind-spark-test")

    print("Step 1: Configuring SparkClient...")
    print(f"  Namespace: {namespace}")
    print(f"  Kube context: {kube_context}")
    print()

    # Build SparkClient with auto-provisioning
    print("Step 2: Building SparkClient with auto-provisioning...")
    print("  Server: 1 core, 1g memory")
    print("  Executors: 2 x (1 core, 1g memory)")
    print("  Spark version: 3.5.0")
    print()

    try:
        # Use context manager for automatic cleanup
        with SparkClient.builder() \
                .namespace(namespace) \
                .kube_config(context=kube_context) \
                .server(cores=1, memory="1g") \
                .executor(cores=1, memory="1g") \
                .num_executors(2) \
                .spark_version("3.5.0") \
                .spark_conf("spark.sql.shuffle.partitions", "10") \
                .cleanup_on_exit(True) \
                .timeout(300) \
                .build() as client:

            print("Step 3: Auto-provisioning Spark Connect server...")
            print("  This creates a SparkConnect CRD on Kubernetes")
            print("  Waiting for server to be ready...")
            print()

            # Get SparkSession (triggers auto-provisioning)
            spark = client.session(app_name="auto-provision-demo")

            # Check server status
            status = client.server_status()
            if status:
                print(f"  ✅ Server ready!")
                print(f"     Name: {status.name}")
                print(f"     State: {status.state.value}")
                print(f"     URL: {status.connect_url}")
            print()

            print("Step 4: Running interactive analysis...")
            run_interactive_analysis(spark)

            print()
            print("Step 5: Cleanup (automatic on context exit)...")
            print("  Server will be deleted automatically")

        print("  ✅ Server deleted!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("What you learned:")
    print("  - SparkClient.builder() for fluent configuration")
    print("  - Auto-provisioning creates SparkConnect CRD automatically")
    print("  - Context manager ensures cleanup on exit")
    print("  - Interactive analysis with auto-provisioned server")
    print()
    print("Code Pattern:")
    print()
    print("  from kubeflow.spark import SparkClient")
    print()
    print("  with SparkClient.builder() \\")
    print("          .namespace('spark-jobs') \\")
    print("          .num_executors(5) \\")
    print("          .build() as client:")
    print("      spark = client.session()")
    print("      df = spark.read.parquet('s3a://data/')")
    print("      df.show()")
    print("  # Server auto-deleted here")
    print()
    print("When to use SparkClient (auto-provision):")
    print("  ✓ Quick interactive exploration")
    print("  ✓ Ad-hoc data analysis")
    print("  ✓ Notebook workflows")
    print("  ✓ Don't want to manage infrastructure")
    print()
    print("When to use SparkSessionClient (existing server):")
    print("  ✓ Long-lived shared Spark cluster")
    print("  ✓ Multiple users sharing same server")
    print("  ✓ Production environments with dedicated resources")
    print()
    print("When to use BatchSparkClient:")
    print("  ✓ ETL pipelines")
    print("  ✓ Scheduled jobs")
    print("  ✓ Non-interactive workloads")
    print()
    print("Next steps:")
    print("  - Try with larger executors: .executor(cores=4, memory='8g')")
    print("  - Enable dynamic allocation: .dynamic_allocation(True, min=1, max=10)")
    print("  - Read from S3: spark.read.parquet('s3a://bucket/data/')")
    print()


if __name__ == "__main__":
    main()

