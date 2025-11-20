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

"""Spark Connect Interactive Session Example.

This example demonstrates how to use Kubeflow SparkSessionClient with Spark Connect
to create interactive data analysis sessions. Unlike batch job submission,
Spark Connect enables long-lived sessions for exploratory data analysis,
iterative development, and notebook-style workflows.

Prerequisites:
1. A Spark cluster with Spark Connect server running (Spark 3.4+)
2. PySpark with Connect support: pip install 'pyspark[connect]>=3.4.0'
3. Network connectivity to Spark Connect server

Key Features Demonstrated:
- Remote connectivity to existing Spark clusters
- Interactive SQL queries and DataFrame operations
- Artifact upload (Python files, JARs)
- Session metrics and monitoring
- Session lifecycle management

Usage:
    python 07_spark_connect_interactive.py --connect-url sc://spark-cluster:15002
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Spark Connect Interactive Session Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--connect-url",
        type=str,
        required=True,
        help=(
            "Spark Connect URL (e.g., sc://spark-cluster:15002). "
            "For Kubernetes: sc://{service-name}.{namespace}.svc.cluster.local:15002"
        ),
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Bearer token for authentication (optional)",
    )
    parser.add_argument(
        "--use-ssl",
        action="store_true",
        default=True,
        help="Use SSL/TLS for secure communication (default: true)",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="kubeflow-spark-connect-demo",
        help="Application name for the session",
    )
    return parser.parse_args()


def run_sql_analysis(session) -> None:
    """Run interactive SQL analysis.

    Args:
        session: ManagedSparkSession instance
    """
    logger.info("=" * 80)
    logger.info("Example 1: Interactive SQL Queries")
    logger.info("=" * 80)

    # Create sample data
    logger.info("Creating sample sales data...")
    sales_data = [
        ("2024-01-01", "Product A", 100, 29.99),
        ("2024-01-01", "Product B", 150, 19.99),
        ("2024-01-02", "Product A", 120, 29.99),
        ("2024-01-02", "Product C", 80, 49.99),
        ("2024-01-03", "Product B", 200, 19.99),
        ("2024-01-03", "Product C", 90, 49.99),
    ]

    # Create DataFrame and register as temp view
    df = session.createDataFrame(sales_data, ["date", "product", "quantity", "price"])
    df.createOrReplaceTempView("sales")

    logger.info("Sample data created and registered as 'sales' view")

    # Run SQL query
    logger.info("\nExecuting SQL: SELECT product, SUM(quantity * price) AS revenue ...")
    result_df = session.sql("""
        SELECT
            product,
            SUM(quantity * price) AS total_revenue,
            SUM(quantity) AS total_quantity,
            AVG(price) AS avg_price
        FROM sales
        GROUP BY product
        ORDER BY total_revenue DESC
    """)

    # Show results
    logger.info("\nQuery Results:")
    results = result_df.collect()
    for row in results:
        logger.info(
            f"  {row.product}: Revenue=${row.total_revenue:.2f}, "
            f"Quantity={row.total_quantity}, AvgPrice=${row.avg_price:.2f}"
        )


def run_dataframe_operations(session) -> None:
    """Run DataFrame transformations.

    Args:
        session: ManagedSparkSession instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: DataFrame Operations")
    logger.info("=" * 80)

    # Create sample user data
    logger.info("Creating user activity data...")
    user_data = [
        (1, "alice@example.com", "premium", 150),
        (2, "bob@example.com", "free", 25),
        (3, "carol@example.com", "premium", 200),
        (4, "dave@example.com", "free", 10),
        (5, "eve@example.com", "premium", 180),
    ]

    df = session.createDataFrame(user_data, ["user_id", "email", "subscription", "activity_score"])

    # Apply transformations
    logger.info("Applying DataFrame transformations...")

    # Filter premium users
    premium_users = df.filter(df.subscription == "premium")

    # Add derived column
    premium_users = premium_users.withColumn(
        "engagement_level",
        session.spark.sql.functions.when(premium_users.activity_score >= 180, "high")
        .when(premium_users.activity_score >= 150, "medium")
        .otherwise("low"),
    )

    # Show results
    logger.info("\nPremium Users with Engagement Levels:")
    results = premium_users.collect()
    for row in results:
        logger.info(
            f"  User {row.user_id} ({row.email}): "
            f"Score={row.activity_score}, Level={row.engagement_level}"
        )


def run_aggregation_analysis(session) -> None:
    """Run aggregation and grouping operations.

    Args:
        session: ManagedSparkSession instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Aggregation and Grouping")
    logger.info("=" * 80)

    # Create sample event data
    logger.info("Creating event stream data...")
    events = [
        ("2024-01-01", "login", "mobile", 1250),
        ("2024-01-01", "login", "web", 3500),
        ("2024-01-01", "purchase", "mobile", 150),
        ("2024-01-02", "login", "mobile", 1300),
        ("2024-01-02", "login", "web", 3800),
        ("2024-01-02", "purchase", "web", 220),
        ("2024-01-03", "login", "mobile", 1400),
        ("2024-01-03", "purchase", "mobile", 180),
    ]

    df = session.createDataFrame(events, ["date", "event_type", "platform", "count"])

    # Group and aggregate
    logger.info("Computing aggregations by platform and event type...")
    agg_df = (
        df.groupBy("platform", "event_type")
        .agg({"count": "sum", "date": "count"})
        .withColumnRenamed("sum(count)", "total_events")
        .withColumnRenamed("count(date)", "num_days")
        .orderBy("platform", "event_type")
    )

    # Show results
    logger.info("\nAggregation Results:")
    results = agg_df.collect()
    for row in results:
        logger.info(
            f"  {row.platform}/{row.event_type}: Total={row.total_events}, Days={row.num_days}"
        )


def demonstrate_session_features(session) -> None:
    """Demonstrate session-specific features.

    Args:
        session: ManagedSparkSession instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Session Features & Metrics")
    logger.info("=" * 80)

    # Get session info
    info = session.get_info()
    logger.info(f"Session ID: {info.session_id}")
    logger.info(f"App Name: {info.app_name}")
    logger.info(f"State: {info.state}")

    # Get metrics
    metrics = session.get_metrics()
    logger.info("\nSession Metrics:")
    logger.info(f"  Queries Executed: {metrics.queries_executed}")
    logger.info(f"  Active Queries: {metrics.active_queries}")
    logger.info(f"  Artifacts Uploaded: {metrics.artifacts_uploaded}")


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Spark Connect Interactive Session Example")
    logger.info("=" * 80)
    logger.info(f"Connect URL: {args.connect_url}")
    logger.info(f"App Name: {args.app_name}")
    logger.info(f"SSL Enabled: {args.use_ssl}")

    try:
        # Import Kubeflow Spark client
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        # Configure ConnectBackend
        logger.info("\nInitializing Spark Connect backend...")
        config = ConnectBackendConfig(
            connect_url=args.connect_url,
            token=args.token,
            use_ssl=args.use_ssl,
            timeout=300,
        )

        # Create SparkSessionClient
        with SparkSessionClient(backend_config=config) as client:
            logger.info("SparkSessionClient initialized successfully")

            # Create interactive session
            logger.info(f"\nCreating Spark Connect session: {args.app_name}")
            session = client.create_session(app_name=args.app_name)
            logger.info(f"Session created: {session.session_id}")

            try:
                # Run examples
                run_sql_analysis(session)
                run_dataframe_operations(session)
                run_aggregation_analysis(session)
                demonstrate_session_features(session)

                logger.info("\n" + "=" * 80)
                logger.info("All examples completed successfully!")
                logger.info("=" * 80)

            finally:
                # Cleanup session
                logger.info("\nClosing session...")
                session.close()
                logger.info("Session closed successfully")

    except ImportError as e:
        logger.error(
            "Failed to import required packages. "
            "Please install: pip install 'pyspark[connect]>=3.4.0'"
        )
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
