#!/usr/bin/env python3
"""
IPython Demo Script for Spark Connect Integration

This script demonstrates connecting to a Spark Connect server running in Kubernetes
and performing interactive DataFrame operations like groupBy, aggregations, etc.

Prerequisites:
1. Kubernetes cluster with Spark Connect server deployed
2. PySpark with Connect support: pip install 'pyspark[connect]>=3.4.0'
3. Kubeflow SDK installed

Usage:
  python ipython_spark_connect_demo.py

Or in IPython:
  %run ipython_spark_connect_demo.py
"""

import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, sdk_path)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_basic_connection():
    """Demonstrate basic connection to Spark Connect server."""
    print_section("1. Connect to Spark Connect Server")

    from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

    # Configuration for Kubernetes Spark Connect server
    # The server is exposed via NodePort on port 30000
    config = ConnectBackendConfig(
        connect_url="sc://localhost:30000",
        use_ssl=False,
        timeout=60,
    )

    print(f"Connecting to: {config.connect_url}")

    # Create client
    client = SparkSessionClient(backend_config=config)
    print("✓ SparkSessionClient created")

    return client


def demo_create_session(client):
    """Demonstrate creating a Spark session."""
    print_section("2. Create Spark Session")

    session = client.create_session(app_name="ipython-demo")
    print(f"✓ Session created: {session.session_id}")
    print(f"  App name: {session.app_name}")
    print(f"  Closed: {session.is_closed}")

    return session


def demo_simple_sql(session):
    """Demonstrate simple SQL queries."""
    print_section("3. Simple SQL Query")

    df = session.sql("SELECT 1 AS id, 'Hello Spark Connect' AS message")
    print("Query: SELECT 1 AS id, 'Hello Spark Connect' AS message")
    print("\nResult:")
    df.show()

    result = df.collect()
    print(f"\nCollected: {result[0].message}")

    return df


def demo_create_dataframe(session):
    """Demonstrate creating DataFrames from Python data."""
    print_section("4. Create DataFrame from Python Data")

    # Sample sales data
    sales_data = [
        (1, "Electronics", "Laptop", 1200.00, 2, "2024-01-15"),
        (2, "Electronics", "Mouse", 25.00, 5, "2024-01-15"),
        (3, "Clothing", "Shirt", 35.00, 3, "2024-01-16"),
        (4, "Electronics", "Keyboard", 75.00, 4, "2024-01-16"),
        (5, "Clothing", "Pants", 55.00, 2, "2024-01-17"),
        (6, "Electronics", "Monitor", 300.00, 3, "2024-01-17"),
        (7, "Clothing", "Jacket", 120.00, 1, "2024-01-18"),
        (8, "Electronics", "Mouse", 25.00, 10, "2024-01-18"),
        (9, "Clothing", "Shirt", 35.00, 5, "2024-01-19"),
        (10, "Electronics", "Laptop", 1200.00, 1, "2024-01-19"),
    ]

    schema = ["id", "category", "product", "price", "quantity", "date"]

    df = session.createDataFrame(sales_data, schema)
    print(f"✓ DataFrame created with {df.count()} rows")
    print("\nSample data:")
    df.show(5)

    return df


def demo_dataframe_operations(session, df):
    """Demonstrate DataFrame transformations."""
    print_section("5. DataFrame Operations - Filter & Select")

    # Filter expensive items
    expensive = df.filter(df.price > 100)
    print("Filter: price > 100")
    expensive.show()

    # Select specific columns
    print("\nSelect: category, product, price")
    df.select("category", "product", "price").show(5)

    return expensive


def demo_groupby_aggregations(session, df):
    """Demonstrate groupBy and aggregations."""
    print_section("6. GroupBy and Aggregations")

    # Group by category and calculate statistics
    print("Aggregation: Total revenue by category")
    from pyspark.sql import functions as F  # noqa: N812

    revenue_df = df.withColumn("revenue", F.col("price") * F.col("quantity"))

    category_stats = revenue_df.groupBy("category").agg(
        F.sum("revenue").alias("total_revenue"),
        F.avg("price").alias("avg_price"),
        F.sum("quantity").alias("total_quantity"),
        F.count("*").alias("num_transactions"),
    )

    print("\nRevenue by Category:")
    category_stats.show()

    # Group by product and sort
    print("\nTop Products by Revenue:")
    product_revenue = revenue_df.groupBy("product").agg(
        F.sum("revenue").alias("total_revenue"),
        F.sum("quantity").alias("total_sold"),
    )

    product_revenue.orderBy(F.desc("total_revenue")).show(5)

    return category_stats


def demo_advanced_aggregations(session, df):
    """Demonstrate advanced aggregations and window functions."""
    print_section("7. Advanced Aggregations")

    from pyspark.sql import functions as F  # noqa: N812
    from pyspark.sql.window import Window

    # Add computed column
    df_with_revenue = df.withColumn("revenue", F.col("price") * F.col("quantity"))

    # Window function: Running total by date
    print("Running Total Revenue by Date:")
    window_spec = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

    daily_revenue = (
        df_with_revenue.groupBy("date")
        .agg(F.sum("revenue").alias("daily_revenue"))
        .withColumn("running_total", F.sum("daily_revenue").over(window_spec))
    )

    daily_revenue.orderBy("date").show()

    # Pivot: Revenue by category and date
    print("\nPivot: Revenue by Category and Date:")
    pivot_df = (
        df_with_revenue.groupBy("date").pivot("category").agg(F.sum("revenue").alias("revenue"))
    )

    pivot_df.orderBy("date").show()

    return daily_revenue


def demo_session_metrics(session):
    """Demonstrate session metrics tracking."""
    print_section("8. Session Metrics")

    metrics = session.get_metrics()
    print(f"Session ID: {metrics.session_id}")
    print(f"Queries Executed: {metrics.queries_executed}")
    print(f"Active Queries: {metrics.active_queries}")
    print(f"Artifacts Uploaded: {metrics.artifacts_uploaded}")

    info = session.get_info()
    print(f"\nSession State: {info.state}")
    print(f"App Name: {info.app_name}")


def demo_multiple_operations(session):
    """Demonstrate chaining multiple operations."""
    print_section("9. Chained Operations")

    # Create sample employee data
    employees = [
        (1, "Alice", "Engineering", 95000, 28),
        (2, "Bob", "Engineering", 120000, 35),
        (3, "Carol", "Sales", 80000, 42),
        (4, "David", "Engineering", 110000, 30),
        (5, "Eve", "Sales", 90000, 38),
        (6, "Frank", "Marketing", 85000, 45),
        (7, "Grace", "Engineering", 105000, 29),
        (8, "Henry", "Marketing", 88000, 33),
    ]

    df = session.createDataFrame(employees, ["id", "name", "dept", "salary", "age"])

    print("Original Data:")
    df.show()

    # Chain multiple operations
    from pyspark.sql import functions as F  # noqa: N812

    result = (
        df.filter(F.col("age") < 40)
        .groupBy("dept")
        .agg(F.avg("salary").alias("avg_salary"), F.count("*").alias("count"))
        .filter(F.col("count") >= 2)
        .orderBy(F.desc("avg_salary"))
    )

    print("\nFiltered Analysis (age < 40, departments with 2+ people):")
    result.show()

    return result


def run_complete_demo():
    """Run complete demonstration."""
    print("\n" + "=" * 80)
    print(" Kubeflow Spark Connect - Interactive Demo")
    print(" Connecting to Kubernetes Spark Connect Server")
    print("=" * 80)

    try:
        # Step 1: Connect
        client = demo_basic_connection()

        # Step 2: Create session
        session = demo_create_session(client)

        # Step 3: Simple SQL
        demo_simple_sql(session)

        # Step 4: Create DataFrame
        df = demo_create_dataframe(session)

        # Step 5: Basic operations
        demo_dataframe_operations(session, df)

        # Step 6: GroupBy aggregations
        demo_groupby_aggregations(session, df)

        # Step 7: Advanced aggregations
        demo_advanced_aggregations(session, df)

        # Step 8: Session metrics
        demo_session_metrics(session)

        # Step 9: Chained operations
        demo_multiple_operations(session)

        print_section("Demo Complete")
        print("✓ All operations completed successfully!")
        print("\nTo continue experimenting:")
        print("  - session object is available for more queries")
        print("  - Try: session.sql('SELECT * FROM ...')")
        print("  - Try: session.createDataFrame(...)")
        print("  - Remember to call: session.close() when done")

        return client, session

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Is Kubernetes cluster running? (kubectl get nodes)")
        print("  2. Is Spark Connect deployed? (kubectl get pods -l app=spark-connect)")
        print(
            "  3. Is port forwarding active? (kubectl port-forward svc/spark-connect 30000:15002)"
        )
        print("  4. Is PySpark installed? (pip install 'pyspark[connect]>=3.4.0')")
        return None, None


# Manual step-by-step execution helper
def print_manual_steps():
    """Print manual steps for running in IPython."""
    print("\n" + "=" * 80)
    print(" Manual Step-by-Step Execution in IPython")
    print("=" * 80)
    print("""
# Step 1: Import and configure
from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

config = ConnectBackendConfig(
    connect_url="sc://localhost:30000",
    use_ssl=False,
    timeout=60
)

# Step 2: Create client and session
client = SparkSessionClient(backend_config=config)
session = client.create_session(app_name="my-analysis")

# Step 3: Create sample data
sales_data = [
    (1, "Electronics", "Laptop", 1200.00, 2),
    (2, "Electronics", "Mouse", 25.00, 5),
    (3, "Clothing", "Shirt", 35.00, 3),
    (4, "Electronics", "Keyboard", 75.00, 4),
]
df = session.createDataFrame(sales_data, ["id", "category", "product", "price", "quantity"])

# Step 4: View data
df.show()

# Step 5: Run aggregations
from pyspark.sql import functions as F
revenue_df = df.withColumn("revenue", F.col("price") * F.col("quantity"))
revenue_df.groupBy("category").agg(F.sum("revenue").alias("total")).show()

# Step 6: Clean up
session.close()
client.close()
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spark Connect Interactive Demo")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual steps instead of running automated demo",
    )
    args = parser.parse_args()

    if args.manual:
        print_manual_steps()
    else:
        client, session = run_complete_demo()

        # Keep objects available for interactive use
        if client and session:
            print("\nObjects available for continued use:")
            print("  - client: SparkSessionClient instance")
            print("  - session: ManagedSparkSession instance")
            print("\nEntering interactive mode... (Ctrl+D to exit)")

            try:
                import IPython

                IPython.embed()
            except ImportError:
                print("\nIPython not installed. Install with: pip install ipython")
                print("Keeping session open for manual cleanup...")
                input("\nPress Enter to close session and exit...")

            if session and not session.is_closed:
                session.close()
                print("✓ Session closed")

            if client:
                client.close()
                print("✓ Client closed")
