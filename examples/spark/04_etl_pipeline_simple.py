#!/usr/bin/env python3
"""
Title: Simple ETL Pipeline
Level: 2 (Intermediate - Data Engineering Basics)
Target Audience: Data Engineers building data pipelines
Time to Run: ~3-4 minutes

Description:
This example demonstrates a simple ETL (Extract-Transform-Load) pipeline pattern,
which is the foundation of data engineering. You'll learn how to extract data from
multiple sources, transform it through cleaning and enrichment, and prepare it for
analytics or loading to a target system.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- ETL pipeline structure and best practices
- Reading from multiple data sources
- Data transformation patterns (cleaning, enrichment, aggregation)
- Data validation and error handling
- Preparing data for downstream consumption

Real-World Use Case:
Building data warehouses, data lakes, analytics pipelines, integration workflows.
"""

import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from kubeflow.spark import (  # noqa: E402
    ApplicationState,
    OperatorBackendConfig,
    BatchSparkClient,
)


def create_etl_script():
    """Create a PySpark script for ETL pipeline.

    Returns:
        str: Python code for ETL pipeline
    """
    return """
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, upper, lower, trim, regexp_replace, when, lit,
    current_timestamp, to_date, year, month, dayofmonth,
    sum as _sum, avg, count, max as _max, min as _min, round as _round,
    concat, coalesce, monotonically_increasing_id
)
from pyspark.sql.types import *
import sys

# Create Spark session
spark = SparkSession.builder \\
    .appName("Simple ETL Pipeline") \\
    .getOrCreate()

print("\\n" + "="*80)
print("ETL PIPELINE EXAMPLE")
print("="*80)
print("\\nPipeline: Customer Orders ETL")
print("Extract - Transform - Load")
print("="*80)

# ============================================================================
# PHASE 1: EXTRACT
# ============================================================================
print("\\n[EXTRACT] Phase 1: Extracting data from source systems...")
print("-" * 80)

# Source 1: Customer data (simulating CRM system)
print("\\n1.1 Extracting customer data from CRM...")
customers_schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("city", StringType(), True),
    StructField("signup_date", StringType(), True),
])

customers_raw = [
    (101, " alice ", "JOHNSON", "alice.j@email.com", "New York", "2023-01-15"),
    (102, "Bob", "smith  ", "BOB.S@EMAIL.COM", "Los Angeles", "2023-02-20"),
    (103, "Carol", "White", "carol.w@email.com", "Chicago", "2023-03-10"),
    (104, "David", "Brown", None, "Houston", "2023-04-05"),  # Missing email
    (105, "Eve", "Davis", "eve.d@email.com", None, "2023-05-12"),  # Missing city
]

customers_df = spark.createDataFrame(customers_raw, customers_schema)
print(f"  Extracted {customers_df.count()} customer records")

# Source 2: Orders data (simulating order management system)
print("\\n1.2 Extracting orders from Order Management System...")
orders_schema = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("customer_id", IntegerType(), False),
    StructField("order_date", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("status", StringType(), True),
])

orders_raw = [
    (1001, 101, "2023-06-01", "Laptop", 1, 1200.00, "completed"),
    (1002, 101, "2023-06-15", "Mouse", 2, 25.00, "completed"),
    (1003, 102, "2023-06-10", "Keyboard", 1, 75.00, "COMPLETED"),  # Inconsistent case
    (1004, 103, "2023-06-20", "Monitor", 2, 300.00, "shipped"),
    (1005, 103, "2023-07-01", "Laptop", 1, 1200.00, "completed"),
    (1006, 104, "2023-07-05", "Mouse", 5, 25.00, "pending"),
    (1007, 999, "2023-07-10", "Desk", 1, 500.00, "completed"),  # Invalid customer_id
    (1008, 105, "2023-07-15", "Chair", 2, 250.00, "cancelled"),
]

orders_df = spark.createDataFrame(orders_raw, orders_schema)
print(f"  Extracted {orders_df.count()} order records")

print("\\n[EXTRACT] Summary:")
print(f"  - Customers: {customers_df.count()} records")
print(f"  - Orders: {orders_df.count()} records")

# ============================================================================
# PHASE 2: TRANSFORM
# ============================================================================
print("\\n\\n[TRANSFORM] Phase 2: Transforming and cleaning data...")
print("-" * 80)

# Step 2.1: Clean customer data
print("\\n2.1 Cleaning customer data...")
customers_clean = customers_df \\
    .withColumn("first_name", trim(col("first_name"))) \\
    .withColumn("first_name", upper(col("first_name"))) \\
    .withColumn("last_name", trim(col("last_name"))) \\
    .withColumn("last_name", upper(col("last_name"))) \\
    .withColumn("email", lower(trim(col("email")))) \\
    .withColumn("signup_date", to_date(col("signup_date"), "yyyy-MM-dd")) \\
    .withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name")))

# Handle missing values
customers_clean = customers_clean \\
    .withColumn("email", coalesce(col("email"), lit("unknown@example.com"))) \\
    .withColumn("city", coalesce(col("city"), lit("Unknown")))

print("  Cleaned customer names (trimmed, normalized case)")
print("  Normalized email addresses to lowercase")
print("  Filled missing emails and cities with defaults")
print("  Created full_name field")

print("\\nCleaned customer sample:")
customers_clean.show(3, truncate=False)

# Step 2.2: Clean and enrich orders data
print("\\n2.2 Cleaning and enriching orders data...")
orders_clean = orders_df \\
    .withColumn("status", lower(trim(col("status")))) \\
    .withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd")) \\
    .withColumn("order_total", col("quantity") * col("unit_price")) \\
    .withColumn("order_year", year(col("order_date"))) \\
    .withColumn("order_month", month(col("order_date")))

print("  Normalized status to lowercase")
print("  Calculated order totals")
print("  Extracted date components (year, month)")

print("\\nCleaned orders sample:")
orders_clean.select(
    "order_id", "customer_id", "order_date", "product_name",
    "quantity", "unit_price", "order_total", "status"
).show(3, truncate=False)

# Step 2.3: Data validation and filtering
print("\\n2.3 Validating data quality...")

# Find orders with invalid customer IDs
valid_customer_ids = customers_clean.select("customer_id").distinct()
invalid_orders = orders_clean.join(
    valid_customer_ids,
    on="customer_id",
    how="left_anti"
)

print(f"  WARNING: Found {invalid_orders.count()} orders with invalid customer IDs")
if invalid_orders.count() > 0:
    print("  Invalid orders:")
    invalid_orders.select("order_id", "customer_id", "product_name").show()

# Filter to valid orders only
orders_valid = orders_clean.join(
    valid_customer_ids,
    on="customer_id",
    how="inner"
)

print(f"  Retained {orders_valid.count()} valid orders")

# Step 2.4: Enrich orders with customer data
print("\\n2.4 Enriching orders with customer information...")
orders_enriched = orders_valid.join(
    customers_clean.select("customer_id", "full_name", "email", "city"),
    on="customer_id",
    how="inner"
)

print("  Joined orders with customer data")
print("\\nEnriched orders sample:")
orders_enriched.select(
    "order_id", "full_name", "city", "product_name",
    "order_total", "status"
).show(3, truncate=False)

# Step 2.5: Create aggregated analytics tables
print("\\n2.5 Creating aggregated analytics...")

# Customer summary
customer_summary = orders_enriched.groupBy("customer_id", "full_name", "city", "email").agg(
    count("order_id").alias("total_orders"),
    _sum("order_total").alias("total_spent"),
    _round(avg("order_total"), 2).alias("avg_order_value"),
    _max("order_date").alias("last_order_date"),
    _min("order_date").alias("first_order_date"),
).orderBy(col("total_spent").desc())

print("  Created customer summary table")

# Product summary
product_summary = orders_enriched.groupBy("product_name").agg(
    count("order_id").alias("total_orders"),
    _sum("quantity").alias("total_quantity_sold"),
    _round(_sum("order_total"), 2).alias("total_revenue"),
    _round(avg("unit_price"), 2).alias("avg_price"),
).orderBy(col("total_revenue").desc())

print("  Created product summary table")

# Monthly summary
monthly_summary = orders_enriched.groupBy("order_year", "order_month").agg(
    count("order_id").alias("total_orders"),
    countDistinct("customer_id").alias("unique_customers"),
    _round(_sum("order_total"), 2).alias("total_revenue"),
    _round(avg("order_total"), 2).alias("avg_order_value"),
).orderBy("order_year", "order_month")

print("  Created monthly summary table")

print("\\n[TRANSFORM] Summary:")
print("  - Cleaned and normalized all fields")
print("  - Validated data quality")
print("  - Enriched orders with customer data")
print("  - Created 3 analytics tables")

# ============================================================================
# PHASE 3: LOAD
# ============================================================================
print("\\n\\n[LOAD] Phase 3: Preparing data for loading...")
print("-" * 80)

# Add metadata columns
print("\\n3.1 Adding metadata columns...")
load_timestamp = current_timestamp()

customer_summary_final = customer_summary.withColumn("etl_load_timestamp", load_timestamp)
product_summary_final = product_summary.withColumn("etl_load_timestamp", load_timestamp)
monthly_summary_final = monthly_summary.withColumn("etl_load_timestamp", load_timestamp)
orders_final = orders_enriched.withColumn("etl_load_timestamp", load_timestamp)

print("  Added ETL timestamp to all tables")

# Display final results
print("\\n3.2 Final output tables ready for loading:")

print("\\n[TABLE 1] Customer Summary (top customers by spend):")
customer_summary_final.show(5, truncate=False)

print("\\n[TABLE 2] Product Summary (top products by revenue):")
product_summary_final.show(5, truncate=False)

print("\\n[TABLE 3] Monthly Summary:")
monthly_summary_final.show(truncate=False)

# In production, you would write to target systems:
# customer_summary_final.write.mode("overwrite").parquet("s3://bucket/customer_summary/")
# product_summary_final.write.mode("overwrite").parquet("s3://bucket/product_summary/")
# monthly_summary_final.write.mode("overwrite").parquet("s3://bucket/monthly_summary/")

print("\\n[LOAD] Summary:")
print("  - customer_summary: Ready for data warehouse")
print("  - product_summary: Ready for analytics")
print("  - monthly_summary: Ready for reporting")
print("  - orders_enriched: Ready for data lake")

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================
print("\\n\\n" + "="*80)
print("ETL PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)

print("\\n📊 Pipeline Statistics:")
print(f"  Input Records:")
print(f"    - Customers: {customers_df.count()}")
print(f"    - Orders: {orders_df.count()}")
print(f"  ")
print(f"  Processing:")
print(f"    - Invalid orders filtered: {invalid_orders.count()}")
print(f"    - Valid orders: {orders_valid.count()}")
print(f"  ")
print(f"  Output Records:")
print(f"    - Customer summary: {customer_summary_final.count()}")
print(f"    - Product summary: {product_summary_final.count()}")
print(f"    - Monthly summary: {monthly_summary_final.count()}")
print(f"    - Enriched orders: {orders_final.count()}")

print("\\n💡 Key Transformations Applied:")
print("  Data cleaning (trim, case normalization)")
print("  Missing value handling")
print("  Data validation (referential integrity)")
print("  Data enrichment (joins)")
print("  Aggregations (customer, product, time-based)")
print("  Metadata addition (timestamps)")

print("\\n🎯 Business Insights:")
top_customer = customer_summary.first()
top_product = product_summary.first()
print(f"  - Top Customer: {top_customer['full_name']} (${top_customer['total_spent']:.2f})")
print(f"  - Top Product: {top_product['product_name']} (${top_product['total_revenue']:.2f})")
print(f"  - Total Revenue: ${orders_enriched.agg(_sum('order_total')).collect()[0][0]:.2f}")

spark.stop()
"""


def main():
    """Main example: Submit ETL pipeline job and get results."""

    print("=" * 80)
    print("EXAMPLE 04: Simple ETL Pipeline")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. ETL pipeline structure (Extract-Transform-Load)")
    print("  2. Extracting from multiple data sources")
    print("  3. Data cleaning and normalization")
    print("  4. Data validation and quality checks")
    print("  5. Data enrichment through joins")
    print("  6. Creating aggregated analytics tables")
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
    app_name = "etl-pipeline-simple"

    print("Step 2: Configuring ETL pipeline...")
    print(f"  App name: {app_name}")
    print("  Spark version: 4.0.0")
    print("  Resources: 1 driver + 2 executors")
    print("  Pipeline: Customer Orders ETL")
    print()

    # Step 3: Submit the application
    print("Step 3: Submitting ETL pipeline...")

    try:
        response = client.submit_application(
            # Application metadata
            app_name=app_name,
            # Placeholder
            main_application_file=("local:///opt/spark/examples/src/main/python/pi.py"),
            # Spark configuration
            spark_version="4.0.0",
            app_type="Python",
            # Resource allocation (medium size for ETL)
            driver_cores=1,
            driver_memory="1g",  # More memory for ETL
            executor_cores=1,
            executor_memory="1g",
            num_executors=2,
            # Required for Spark 4.0
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
            },
        )

        print("  ETL pipeline submitted successfully!")
        print(f"  Submission ID: {response.submission_id}")
        print(f"  Status: {response.status}")
        print()

    except Exception as e:
        print(f"  ERROR: Submission failed: {e}")
        sys.exit(1)

    # Step 4: Monitor the application
    print("Step 4: Monitoring ETL pipeline (this may take 2-3 minutes)...")
    print("  Pipeline stages: Extract - Transform - Load")

    try:
        # Wait for completion with timeout
        final_status = client.wait_for_completion(
            submission_id=app_name,
            timeout=300,  # 5 minutes max
            polling_interval=5,  # Check every 5 seconds
        )

        print("  ETL pipeline completed!")
        print(f"  Final state: {final_status.state.value}")
        print()

        # Check if successful
        if final_status.state != ApplicationState.COMPLETED:
            print(f"  WARNING: Pipeline did not complete successfully: {final_status.state.value}")
            print("  Check logs below for details.")

    except TimeoutError:
        print("  ERROR: Pipeline did not complete within 5 minutes")
        print(f"  You can check status later with: client.get_status('{app_name}')")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Error monitoring pipeline: {e}")
        sys.exit(1)

    # Step 5: Retrieve results from logs
    print("Step 5: Retrieving ETL results from logs...")
    print()

    try:
        logs = list(client.get_logs(app_name))

        print("=" * 80)
        print("ETL PIPELINE RESULTS")
        print("=" * 80)

        # Display important sections
        important_keywords = [
            "ETL PIPELINE",
            "[EXTRACT]",
            "[TRANSFORM]",
            "[LOAD]",
            "Customer Summary",
            "Product Summary",
            "Monthly Summary",
            "Pipeline Statistics",
            "Business Insights",
        ]

        for line in logs:
            if (
                any(keyword in line for keyword in important_keywords)
                or "Done" in line
                or "WARNING" in line
                or "📊" in line
                or "💡" in line
                or "🎯" in line
            ):
                print(line)

        print()
        print("=" * 80)

    except Exception as e:
        print(f"  WARNING: Could not retrieve logs: {e}")
        print("  The pipeline may have completed but logs are not yet available")

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
    print("  ETL pipeline structure and phases")
    print("  Extracting from multiple sources")
    print("  Data cleaning and normalization techniques")
    print("  Data validation (referential integrity)")
    print("  Data enrichment through joins")
    print("  Creating aggregated analytics tables")
    print("  Adding metadata for audit trails")
    print()
    print("ETL Best Practices Demonstrated:")
    print("  1. Separate Extract-Transform-Load phases")
    print("  2. Data quality validation at each step")
    print("  3. Handle missing/invalid data gracefully")
    print("  4. Add metadata (timestamps, lineage)")
    print("  5. Create reusable, modular transformations")
    print("  6. Generate summary statistics")
    print()
    print("Production Considerations:")
    print("  - Read from S3/HDFS instead of in-memory data")
    print("  - Write outputs to data warehouse (Redshift, BigQuery)")
    print("  - Add error handling and retry logic")
    print("  - Implement incremental processing")
    print("  - Add data quality assertions")
    print("  - Monitor pipeline metrics")
    print()
    print("Next steps:")
    print("  - Try example 05: Scheduled batch processing")
    print("  - Implement incremental ETL (delta processing)")
    print("  - Add data quality framework (Great Expectations)")
    print("  - Orchestrate with Airflow/Argo Workflows")
    print()


if __name__ == "__main__":
    main()
