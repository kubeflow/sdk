#!/usr/bin/env python3
"""
Title: Local Spark Quickstart (No Kubernetes Required)
Level: 0 (Quickstart)
Target Audience: Anyone learning PySpark or testing locally
Time to Run: ~1 minute

Description:
This example demonstrates how to run PySpark locally without any Kubernetes cluster.
Perfect for development, testing, and learning PySpark before scaling to Kubernetes.

Prerequisites:
- Python 3.8+
- PySpark installed: pip install pyspark

What You'll Learn:
- How to create a local SparkSession
- Basic DataFrame operations
- SQL queries with Spark
- When to use local mode vs Kubernetes

No Kubernetes Required!
"""

import sys


def main():
    """Run local Spark example."""
    print("=" * 80)
    print("EXAMPLE 00: Local Spark Quickstart")
    print("=" * 80)
    print()
    print("This example runs Spark locally - no Kubernetes needed!")
    print()

    # Check PySpark installation
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import avg, col, count, sum as _sum
    except ImportError:
        print("ERROR: PySpark not installed.")
        print("Install with: pip install pyspark")
        sys.exit(1)

    # Step 1: Create local SparkSession
    print("Step 1: Creating local SparkSession...")
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("LocalQuickstart") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    print(f"  Spark version: {spark.version}")
    print(f"  Master: {spark.sparkContext.master}")
    print()

    # Step 2: Create sample data
    print("Step 2: Creating sample sales data...")
    sales_data = [
        ("2024-01-01", "Product A", "Electronics", 10, 99.99, "North"),
        ("2024-01-01", "Product B", "Electronics", 5, 149.99, "South"),
        ("2024-01-02", "Product A", "Electronics", 8, 99.99, "East"),
        ("2024-01-02", "Product C", "Furniture", 3, 299.99, "West"),
        ("2024-01-03", "Product B", "Electronics", 12, 149.99, "North"),
        ("2024-01-03", "Product D", "Furniture", 2, 499.99, "South"),
    ]
    columns = ["date", "product", "category", "quantity", "price", "region"]
    df = spark.createDataFrame(sales_data, columns)

    print(f"  Created DataFrame with {df.count()} rows")
    print()
    print("Sample Data:")
    df.show()

    # Step 3: DataFrame operations
    print("Step 3: Running DataFrame operations...")

    # Add revenue column
    df_with_revenue = df.withColumn("revenue", col("quantity") * col("price"))

    # Group by category
    category_summary = df_with_revenue.groupBy("category").agg(
        count("*").alias("transactions"),
        _sum("quantity").alias("total_units"),
        _sum("revenue").alias("total_revenue"),
        avg("revenue").alias("avg_revenue")
    ).orderBy(col("total_revenue").desc())

    print()
    print("=" * 60)
    print("SALES SUMMARY BY CATEGORY")
    print("=" * 60)
    category_summary.show()

    # Step 4: SQL queries
    print("Step 4: Running SQL queries...")
    df.createOrReplaceTempView("sales")

    sql_result = spark.sql("""
        SELECT
            region,
            COUNT(*) as num_sales,
            SUM(quantity * price) as total_revenue
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC
    """)

    print()
    print("=" * 60)
    print("SALES BY REGION (SQL)")
    print("=" * 60)
    sql_result.show()

    # Step 5: Show insights
    print("Step 5: Generating insights...")
    total_revenue = df_with_revenue.agg(_sum("revenue")).collect()[0][0]
    total_units = df.agg(_sum("quantity")).collect()[0][0]

    print()
    print("=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print(f"  Total Revenue: ${total_revenue:,.2f}")
    print(f"  Total Units Sold: {total_units}")
    print(f"  Unique Products: {df.select('product').distinct().count()}")
    print(f"  Categories: {df.select('category').distinct().count()}")
    print(f"  Regions: {df.select('region').distinct().count()}")

    # Cleanup
    print()
    print("Step 6: Stopping SparkSession...")
    spark.stop()
    print("  Done!")

    print()
    print("=" * 80)
    print("LOCAL QUICKSTART COMPLETED!")
    print("=" * 80)
    print()
    print("What you learned:")
    print("  - Creating a local SparkSession")
    print("  - DataFrame creation and operations")
    print("  - SQL queries with Spark")
    print("  - Basic aggregations and transformations")
    print()
    print("When to use Local Mode:")
    print("  ✓ Development and testing")
    print("  ✓ Learning PySpark")
    print("  ✓ Small datasets (<1GB)")
    print("  ✓ Quick prototyping")
    print()
    print("When to use Kubernetes:")
    print("  ✓ Large datasets (>1GB)")
    print("  ✓ Production workloads")
    print("  ✓ Multi-user environments")
    print("  ✓ Need for scaling")
    print()
    print("Next steps:")
    print("  - Set up Kubernetes: ./setup_test_environment.sh")
    print("  - Try example 01: python 01_hello_spark_pi.py")
    print("  - Try interactive mode: python 08_auto_provision_spark_connect.py")
    print()


if __name__ == "__main__":
    main()

