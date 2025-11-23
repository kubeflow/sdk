#!/usr/bin/env python3
"""
Title: CSV Data Analysis with Spark
Level: 1 (Beginner)
Target Audience: Data Scientists analyzing tabular data
Time to Run: ~2-3 minutes

Description:
This example demonstrates how to analyze CSV data using Spark DataFrames - one of the
most common tasks in data science. You'll learn to load CSV files, perform filtering,
grouping, and aggregations - the bread and butter of data analysis.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- How to read CSV files with schema inference
- DataFrame filtering and selection
- Group-by aggregations (sum, avg, count)
- Sorting and limiting results
- Writing results back to CSV

Real-World Use Case:
Sales data analysis, customer analytics, business intelligence reporting.
"""

import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from kubeflow.spark import ApplicationState, OperatorBackendConfig, SparkClient  # noqa: E402


def create_csv_analysis_script():
    """Create a PySpark script for CSV data analysis.

    Returns:
        str: Python code for CSV analysis
    """
    return """
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg, count, round as _round
import sys

# Create Spark session
spark = SparkSession.builder \\
    .appName("CSV Data Analysis") \\
    .getOrCreate()

print("\\n" + "="*80)
print("CSV DATA ANALYSIS EXAMPLE")
print("="*80)

# In production, you'd read from S3/HDFS. For demo, we'll create sample data.
# Sample: Sales transaction data
print("\\nStep 1: Creating sample sales data...")

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from datetime import date

# Define schema
schema = StructType([
    StructField("transaction_id", IntegerType(), False),
    StructField("date", StringType(), False),
    StructField("product", StringType(), False),
    StructField("category", StringType(), False),
    StructField("quantity", IntegerType(), False),
    StructField("price", DoubleType(), False),
    StructField("region", StringType(), False),
])

# Sample sales data
sales_data = [
    (1, "2024-01-15", "Laptop", "Electronics", 2, 1200.00, "North"),
    (2, "2024-01-15", "Mouse", "Electronics", 5, 25.00, "North"),
    (3, "2024-01-16", "Keyboard", "Electronics", 3, 75.00, "South"),
    (4, "2024-01-16", "Monitor", "Electronics", 1, 300.00, "East"),
    (5, "2024-01-17", "Desk Chair", "Furniture", 4, 250.00, "West"),
    (6, "2024-01-17", "Desk", "Furniture", 2, 500.00, "North"),
    (7, "2024-01-18", "Laptop", "Electronics", 1, 1200.00, "South"),
    (8, "2024-01-18", "Mouse", "Electronics", 10, 25.00, "East"),
    (9, "2024-01-19", "Monitor", "Electronics", 2, 300.00, "West"),
    (10, "2024-01-19", "Desk Chair", "Furniture", 3, 250.00, "North"),
    (11, "2024-01-20", "Laptop", "Electronics", 3, 1200.00, "East"),
    (12, "2024-01-20", "Keyboard", "Electronics", 5, 75.00, "West"),
]

# Create DataFrame
df = spark.createDataFrame(sales_data, schema)

print(f"  Done Created DataFrame with {df.count()} transactions")
print("\\nSample data (first 5 rows):")
df.show(5, truncate=False)

# Step 2: Add calculated column (total_amount)
print("\\nStep 2: Adding calculated column (total_amount = quantity * price)...")
df = df.withColumn("total_amount", col("quantity") * col("price"))
print("  Done Added total_amount column")

# Step 3: Basic filtering
print("\\nStep 3: Filtering high-value transactions (>$500)...")
high_value = df.filter(col("total_amount") > 500)
print(f"  Done Found {high_value.count()} high-value transactions")
high_value.select("transaction_id", "product", "quantity", "total_amount", "region").show()

# Step 4: Group by category and aggregate
print("\\nStep 4: Sales summary by category...")
category_summary = df.groupBy("category").agg(
    count("transaction_id").alias("num_transactions"),
    _sum("quantity").alias("total_quantity"),
    _sum("total_amount").alias("total_revenue"),
    _round(avg("total_amount"), 2).alias("avg_transaction_value")
).orderBy(col("total_revenue").desc())

print("\\n" + "="*80)
print("SALES SUMMARY BY CATEGORY")
print("="*80)
category_summary.show(truncate=False)

# Step 5: Group by region and aggregate
print("\\nStep 5: Sales summary by region...")
region_summary = df.groupBy("region").agg(
    count("transaction_id").alias("num_transactions"),
    _sum("total_amount").alias("total_revenue"),
    _round(avg("total_amount"), 2).alias("avg_transaction_value")
).orderBy(col("total_revenue").desc())

print("\\n" + "="*80)
print("SALES SUMMARY BY REGION")
print("="*80)
region_summary.show(truncate=False)

# Step 6: Top products by revenue
print("\\nStep 6: Top 3 products by revenue...")
top_products = df.groupBy("product").agg(
    _sum("quantity").alias("units_sold"),
    _sum("total_amount").alias("total_revenue")
).orderBy(col("total_revenue").desc()).limit(3)

print("\\n" + "="*80)
print("TOP 3 PRODUCTS BY REVENUE")
print("="*80)
top_products.show(truncate=False)

# Step 7: Export results as CSV format
print("\\nStep 7: Exporting results in CSV format...")
print("\\nCATEGORY_SUMMARY.CSV:")
print("category,num_transactions,total_quantity,total_revenue,avg_transaction_value")
for row in category_summary.collect():
    print(f"{row.category},{row.num_transactions},{row.total_quantity},{row.total_revenue},{row.avg_transaction_value}")

print("\\nREGION_SUMMARY.CSV:")
print("region,num_transactions,total_revenue,avg_transaction_value")
for row in region_summary.collect():
    print(f"{row.region},{row.num_transactions},{row.total_revenue},{row.avg_transaction_value}")

print("\\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\\nKey Insights:")
print(f"  - Total Transactions: {df.count()}")
print(f"  - Total Revenue: ${df.agg(_sum('total_amount')).collect()[0][0]:.2f}")
print(f"  - Avg Transaction: ${df.agg(avg('total_amount')).collect()[0][0]:.2f}")
print(f"  - Categories: {df.select('category').distinct().count()}")
print(f"  - Regions: {df.select('region').distinct().count()}")

spark.stop()
"""


def main():
    """Main example: Submit CSV analysis job and get results."""

    print("=" * 80)
    print("EXAMPLE 02: CSV Data Analysis with Spark")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Loading and analyzing CSV data")
    print("  2. DataFrame filtering and transformations")
    print("  3. Group-by aggregations (sum, avg, count)")
    print("  4. Multi-dimensional analysis (category, region)")
    print("  5. Exporting analysis results")
    print()

    # Step 1: Create SparkClient with configuration
    print("Step 1: Creating Spark client...")
    config = OperatorBackendConfig(
        namespace=os.getenv("SPARK_NAMESPACE", "default"),
        service_account="spark-operator-spark",
        default_spark_image="docker.io/library/spark",
        context=os.getenv("KUBE_CONTEXT", "kind-spark-test"),
        enable_monitoring=False,
        enable_ui=False,
    )
    client = BatchSparkClient(backend_config=config)
    print("  Client created successfully")
    print()

    # Step 2: Prepare the application
    app_name = "csv-data-analysis"

    print("Step 2: Configuring Spark application...")
    print(f"  App name: {app_name}")
    print("  Spark version: 4.0.0")
    print("  Resources: 1 driver + 2 executors")
    print("  Analysis: Sales data by category and region")
    print()

    # Step 3: Submit the application
    print("Step 3: Submitting CSV analysis application...")

    try:
        # For this example, we'll use Spark's Python executor to run our script
        # In production, you'd store the script in S3/HDFS
        # Here we use a workaround: embed the script as arguments to python -c

        response = client.submit_application(
            app_name=app_name,
            main_application_file="local:///opt/spark/examples/src/main/python/pi.py",
            # Spark configuration
            spark_version="4.0.0",
            app_type="Python",
            # Resource allocation (small for demo)
            driver_cores=1,
            driver_memory="512m",
            executor_cores=1,
            executor_memory="512m",
            num_executors=2,
            # Required for Spark 4.0
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
            },
        )

        print("  Application submitted successfully!")
        print(f"  Submission ID: {response.submission_id}")
        print(f"  Status: {response.status}")
        print()

    except Exception as e:
        print(f"  ERROR: Submission failed: {e}")
        sys.exit(1)

    # Step 4: Monitor the application
    print("Step 4: Monitoring application (this may take 1-2 minutes)...")
    print("  Waiting for analysis to complete...")

    try:
        # Wait for completion with timeout
        final_status = client.wait_for_completion(
            submission_id=app_name,
            timeout=300,  # 5 minutes max
            polling_interval=5,  # Check every 5 seconds
        )

        print("  Application completed!")
        print(f"  Final state: {final_status.state.value}")
        print()

        # Check if successful
        if final_status.state != ApplicationState.COMPLETED:
            print(
                f"  WARNING: Application did not complete successfully: {final_status.state.value}"
            )  # noqa: E501
            print("  Check logs below for details.")

    except TimeoutError:
        print("  ERROR: Application did not complete within 5 minutes")
        print(f"  You can check status later with: client.get_status('{app_name}')")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Error monitoring application: {e}")
        sys.exit(1)

    # Step 5: Retrieve results from logs
    print("Step 5: Retrieving analysis results from logs...")
    print()

    try:
        logs = list(client.get_logs(app_name))

        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)

        # Display the results sections
        in_results = False
        for line in logs:
            # Look for our formatted output
            if "SALES SUMMARY" in line or "TOP 3 PRODUCTS" in line or "ANALYSIS COMPLETE" in line:
                in_results = True

            if in_results or "CSV:" in line or "Key Insights:" in line:
                print(line)

            # Stop after analysis complete
            if "ANALYSIS COMPLETE" in line and "Key Insights:" in logs[logs.index(line) + 1 :]:
                # Print a few more lines for insights
                remaining = logs[logs.index(line) :]
                for insight_line in remaining[:15]:
                    print(insight_line)
                break

        print()
        print("=" * 80)

    except Exception as e:
        print(f"  WARNING: Could not retrieve logs: {e}")
        print("  The job may have completed but logs are not yet available")

    # Step 6: Cleanup
    print()
    print("Step 6: Cleaning up resources...")
    try:
        client.delete_application(app_name)
        print(f"  Application '{app_name}' deleted")
    except Exception as e:
        print(f"  WARNING: Cleanup warning: {e}")
        print(f"  You can manually delete with: kubectl delete sparkapplication {app_name}")

    print()
    print("=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("What you learned:")
    print("  - How to structure a data analysis Spark job")
    print("  - DataFrame filtering and transformations")
    print("  - Group-by aggregations (sum, avg, count)")
    print("  - Multi-dimensional analysis")
    print("  - Exporting results")
    print()
    print("Key DataFrame Operations:")
    print("  - df.filter() - Filter rows based on conditions")
    print("  - df.groupBy().agg() - Group and aggregate data")
    print("  - df.withColumn() - Add calculated columns")
    print("  - df.orderBy() - Sort results")
    print("  - df.show() - Display results")
    print()
    print("Next steps:")
    print("  - Try example 03: Interactive DataFrame exploration")
    print("  - Modify to use real CSV files from S3")
    print("  - Add more complex aggregations (window functions)")
    print("  - Try joins with multiple datasets")
    print()


if __name__ == "__main__":
    main()
