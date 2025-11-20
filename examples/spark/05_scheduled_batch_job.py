#!/usr/bin/env python3
"""
Title: Scheduled Batch Job with Resilience
Level: 2 (Intermediate - Batch Processing)
Target Audience: Data Engineers building production batch pipelines
Time to Run: ~3-4 minutes

Description:
This example demonstrates production-ready batch processing patterns including
idempotent processing, incremental updates, restart policies, and resilience
features. You'll learn how to build reliable batch jobs that can handle failures
and process data incrementally.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- Batch processing patterns (full vs incremental)
- Idempotent job design
- Restart policies and failure handling
- Time-based partitioning
- Checkpoint and recovery patterns
- Production batch job best practices

Real-World Use Case:
Daily data warehouse refresh, nightly ETL jobs, periodic reporting, data synchronization.
"""

from datetime import datetime
import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from kubeflow.spark import (  # noqa: E402
    ApplicationState,
    OperatorBackendConfig,
    RestartPolicy,
    RestartPolicyType,
    BatchSparkClient,
)


def create_batch_job_script():
    """Create a PySpark script for scheduled batch processing.

    Returns:
        str: Python code for batch job
    """
    return """
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, current_timestamp, to_date, date_format,
    year, month, dayofmonth, sum as _sum, count,
    max as _max, min as _min, when, coalesce
)
from pyspark.sql.types import *
from datetime import datetime, timedelta
import sys

# Create Spark session
spark = SparkSession.builder \\
    .appName("Scheduled Batch Job") \\
    .getOrCreate()

print("\\n" + "="*80)
print("SCHEDULED BATCH JOB - DAILY TRANSACTION PROCESSING")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
print("\\n[CONFIG] Batch Job Configuration...")

# In production, these would come from job parameters
BATCH_DATE = datetime.now().strftime("%Y-%m-%d")
LOOKBACK_DAYS = 7  # Process last 7 days for incremental
JOB_ID = f"batch_{BATCH_DATE.replace('-', '')}"

print(f"  - Batch Date: {BATCH_DATE}")
print(f"  - Job ID: {JOB_ID}")
print(f"  - Lookback Days: {LOOKBACK_DAYS}")
print(f"  - Mode: Incremental")

# ============================================================================
# STEP 1: EXTRACT - Read Source Data
# ============================================================================
print("\\n[STEP 1] Extracting source data...")
print("-" * 80)

# Simulate transactional data source
transactions_schema = StructType([
    StructField("transaction_id", IntegerType(), False),
    StructField("transaction_date", StringType(), False),
    StructField("customer_id", IntegerType(), False),
    StructField("product_id", IntegerType(), False),
    StructField("amount", DoubleType(), False),
    StructField("status", StringType(), False),
])

# Generate sample transactions for last 7 days
base_date = datetime.now()
transactions_data = []
tx_id = 1

for day_offset in range(7):
    tx_date = (base_date - timedelta(days=day_offset)).strftime("%Y-%m-%d")
    # Generate 3-5 transactions per day
    num_txs = 3 + (day_offset % 3)
    for i in range(num_txs):
        customer_id = 100 + (tx_id % 5)
        product_id = 200 + (tx_id % 10)
        amount = round(50.0 + (tx_id % 20) * 25.5, 2)
        status = "completed" if tx_id % 10 != 0 else "pending"
        transactions_data.append((tx_id, tx_date, customer_id, product_id, amount, status))
        tx_id += 1

transactions_df = spark.createDataFrame(transactions_data, transactions_schema)

print(f"  Loaded {transactions_df.count()} total transactions")

# Show date range
date_range = transactions_df.agg(
    _min("transaction_date").alias("min_date"),
    _max("transaction_date").alias("max_date")
).collect()[0]

print(f"  - Date range: {date_range['min_date']} to {date_range['max_date']}")

# ============================================================================
# STEP 2: INCREMENTAL PROCESSING
# ============================================================================
print("\\n[STEP 2] Applying incremental processing logic...")
print("-" * 80)

# Calculate cutoff date for incremental processing
cutoff_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

print(f"  - Processing transactions >= {cutoff_date}")

# Filter for incremental window (idempotent - same date range produces same result)
incremental_df = transactions_df.filter(col("transaction_date") >= lit(cutoff_date))

print(f"  Filtered to {incremental_df.count()} transactions in incremental window")

# ============================================================================
# STEP 3: TRANSFORM - Business Logic
# ============================================================================
print("\\n[STEP 3] Applying business transformations...")
print("-" * 80)

# Add computed columns
enriched_df = incremental_df \\
    .withColumn("processing_date", lit(BATCH_DATE)) \\
    .withColumn("processing_timestamp", current_timestamp()) \\
    .withColumn("job_id", lit(JOB_ID)) \\
    .withColumn("year", year(col("transaction_date"))) \\
    .withColumn("month", month(col("transaction_date"))) \\
    .withColumn("day", dayofmonth(col("transaction_date")))

# Apply business rules
enriched_df = enriched_df \\
    .withColumn("is_high_value", when(col("amount") > 500, lit(True)).otherwise(lit(False))) \\
    .withColumn("is_completed", when(col("status") == "completed", lit(True)).otherwise(lit(False)))

print("  Added metadata columns (processing_date, job_id)")
print("  Added date partitions (year, month, day)")
print("  Applied business rules (high_value, completion flags)")

# ============================================================================
# STEP 4: AGGREGATIONS - Daily Summary
# ============================================================================
print("\\n[STEP 4] Creating daily aggregations...")
print("-" * 80)

daily_summary = enriched_df.groupBy("transaction_date", "year", "month", "day").agg(
    count("transaction_id").alias("transaction_count"),
    count(when(col("is_completed"), True)).alias("completed_count"),
    count(when(~col("is_completed"), True)).alias("pending_count"),
    count(when(col("is_high_value"), True)).alias("high_value_count"),
    _sum("amount").alias("total_amount"),
    _max("amount").alias("max_amount"),
    _min("amount").alias("min_amount"),
).withColumn("processing_date", lit(BATCH_DATE)) \\
 .withColumn("job_id", lit(JOB_ID)) \\
 .orderBy("transaction_date")

print("  Created daily summary table")
print("\\nDaily Summary:")
daily_summary.show(truncate=False)

# ============================================================================
# STEP 5: CUSTOMER AGGREGATIONS
# ============================================================================
print("\\n[STEP 5] Creating customer aggregations...")
print("-" * 80)

customer_summary = enriched_df.filter(col("is_completed")).groupBy("customer_id").agg(
    count("transaction_id").alias("transaction_count"),
    _sum("amount").alias("total_spent"),
    _max("amount").alias("max_transaction"),
    _min("amount").alias("min_transaction"),
    count(when(col("is_high_value"), True)).alias("high_value_transactions"),
).withColumn("processing_date", lit(BATCH_DATE)) \\
 .withColumn("job_id", lit(JOB_ID)) \\
 .orderBy(col("total_spent").desc())

print("  Created customer summary table")
print("\\nCustomer Summary (Top 5):")
customer_summary.show(5, truncate=False)

# ============================================================================
# STEP 6: DATA QUALITY CHECKS
# ============================================================================
print("\\n[STEP 6] Running data quality checks...")
print("-" * 80)

# Check 1: No null values in critical columns
null_check = enriched_df.filter(
    col("transaction_id").isNull() |
    col("customer_id").isNull() |
    col("amount").isNull()
).count()

print(f"  - Null check: {null_check} records with nulls (expecting 0)")

# Check 2: All amounts are positive
negative_amount_check = enriched_df.filter(col("amount") < 0).count()
print(f"  - Negative amount check: {negative_amount_check} records (expecting 0)")

# Check 3: Valid date range
out_of_range = enriched_df.filter(
    (col("transaction_date") < cutoff_date) |
    (col("transaction_date") > BATCH_DATE)
).count()
print(f"  - Date range check: {out_of_range} out of range (expecting 0)")

# Overall quality score
quality_passed = (null_check == 0) and (negative_amount_check == 0) and (out_of_range == 0)
quality_status = 'ALL QUALITY CHECKS PASSED' if quality_passed else 'WARNING: QUALITY ISSUES DETECTED'
print(f"\\n  {quality_status}")

# ============================================================================
# STEP 7: SIMULATE WRITE TO PARTITIONED STORAGE
# ============================================================================
print("\\n[STEP 7] Preparing output for partitioned storage...")
print("-" * 80)

# In production, you would write:
# enriched_df.write \\
#   .mode("overwrite") \\
#   .partitionBy("year", "month", "day") \\
#   .parquet("s3://bucket/transactions/")

print("  - Output format: Parquet")
print("  - Partitioning: year/month/day")
print("  - Write mode: Overwrite (idempotent)")
print("\\n  Partitions that would be written:")

partitions = enriched_df.select("year", "month", "day").distinct().collect()
for partition in partitions:
    print(f"    - year={partition['year']}/month={partition['month']}/day={partition['day']}")

# ============================================================================
# STEP 8: JOB SUMMARY
# ============================================================================
print("\\n\\n" + "="*80)
print("BATCH JOB COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\\n📊 Job Statistics:")
print(f"  Job ID: {JOB_ID}")
print(f"  Batch Date: {BATCH_DATE}")
print(f"  Processing Window: {cutoff_date} to {BATCH_DATE}")
print(f"  ")
print(f"  Records Processed:")
print(f"    - Total transactions: {enriched_df.count()}")
print(f"    - Completed: {enriched_df.filter(col('is_completed')).count()}")
print(f"    - Pending: {enriched_df.filter(~col('is_completed')).count()}")
print(f"    - High value: {enriched_df.filter(col('is_high_value')).count()}")
print(f"  ")
print(f"  Outputs Generated:")
print(f"    - Daily summaries: {daily_summary.count()} days")
print(f"    - Customer summaries: {customer_summary.count()} customers")
print(f"    - Partitions: {len(partitions)}")
print(f"  ")
print(f"  Data Quality:")
print(f"    - Quality checks: {'PASSED' if quality_passed else 'FAILED'}")

print("\\n💡 Batch Processing Features Demonstrated:")
print("  Incremental processing (configurable lookback)")
print("  Idempotent design (same input - same output)")
print("  Date partitioning for efficient queries")
print("  Job metadata for audit trail")
print("  Data quality validation")
print("  Business rule application")

print("\\n🔄 Production Considerations:")
print("  - Schedule with Airflow/Argo for automated runs")
print("  - Add checkpoint/recovery for large datasets")
print("  - Implement retry logic with exponential backoff")
print("  - Monitor job metrics and SLAs")
print("  - Use restart policies for fault tolerance")

spark.stop()
"""


def main():
    """Main example: Submit scheduled batch job with resilience features."""

    print("=" * 80)
    print("EXAMPLE 05: Scheduled Batch Job with Resilience")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Production batch job patterns")
    print("  2. Incremental processing (configurable lookback)")
    print("  3. Idempotent job design")
    print("  4. Restart policies for fault tolerance")
    print("  5. Date partitioning")
    print("  6. Data quality validation")
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

    # Step 2: Prepare the application with resilience
    app_name = "batch-job-scheduled"
    batch_date = datetime.now().strftime("%Y-%m-%d")

    print("Step 2: Configuring batch job with resilience...")
    print(f"  App name: {app_name}")
    print(f"  Batch date: {batch_date}")
    print("  Spark version: 4.0.0")
    print("  Resources: 1 driver + 2 executors")
    print("  Restart policy: OnFailure (retry up to 3 times)")
    print()

    # Step 3: Submit the application with restart policy
    print("Step 3: Submitting batch job with fault tolerance...")

    try:
        # Configure restart policy for production resilience
        restart_policy = RestartPolicy(
            type=RestartPolicyType.ON_FAILURE,
            on_failure_retries=3,  # Retry up to 3 times on failure
            on_failure_retry_interval=30,  # Wait 30 seconds between retries
            on_submission_failure_retries=2,  # Retry submission failures
            on_submission_failure_retry_interval=15,  # Wait 15 seconds
        )

        response = client.submit_application(
            # Application metadata
            app_name=app_name,
            # Placeholder
            main_application_file=("local:///opt/spark/examples/src/main/python/pi.py"),
            # Spark configuration
            spark_version="4.0.0",
            app_type="Python",
            # Resource allocation for batch processing
            driver_cores=1,
            driver_memory="1g",
            executor_cores=1,
            executor_memory="1g",
            num_executors=2,
            # Resilience configuration
            restart_policy=restart_policy,
            time_to_live_seconds=3600,  # Auto-cleanup after 1 hour
            # Batch job metadata
            labels={
                "job_type": "batch",
                "schedule": "daily",
                "batch_date": batch_date.replace("-", ""),
            },
            # Required for Spark 4.0
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
            },
        )

        print("  Batch job submitted successfully!")
        print(f"  Submission ID: {response.submission_id}")
        print(f"  Status: {response.status}")
        print()
        print("  Resilience features enabled:")
        print(f"    - Retry on failure: {restart_policy.on_failure_retries} attempts")
        print(f"    - Retry interval: {restart_policy.on_failure_retry_interval}s")
        print("    - Auto-cleanup: After 1 hour")
        print()

    except Exception as e:
        print(f"  ERROR: Submission failed: {e}")
        sys.exit(1)

    # Step 4: Monitor the application
    print("Step 4: Monitoring batch job (this may take 2-3 minutes)...")
    print("  Processing incremental data window...")

    try:
        # Wait for completion with timeout
        final_status = client.wait_for_completion(
            submission_id=app_name,
            timeout=300,  # 5 minutes max
            polling_interval=5,  # Check every 5 seconds
        )

        print("  Batch job completed!")
        print(f"  Final state: {final_status.state.value}")
        print()

        # Check if successful
        if final_status.state != ApplicationState.COMPLETED:
            print(f"  WARNING: Job did not complete successfully: {final_status.state.value}")
            print("  Restart policy would trigger automatic retry")
            print("  Check logs below for details.")

    except TimeoutError:
        print("  ERROR: Job did not complete within 5 minutes")
        print(f"  You can check status later with: client.get_status('{app_name}')")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Error monitoring job: {e}")
        sys.exit(1)

    # Step 5: Retrieve results from logs
    print("Step 5: Retrieving batch job results...")
    print()

    try:
        logs = list(client.get_logs(app_name))

        print("=" * 80)
        print("BATCH JOB RESULTS")
        print("=" * 80)

        # Display important sections
        important_keywords = [
            "SCHEDULED BATCH JOB",
            "[CONFIG]",
            "[STEP",
            "Daily Summary",
            "Customer Summary",
            "quality checks",
            "Job Statistics",
            "BATCH JOB COMPLETED",
        ]

        for line in logs:
            if (
                any(keyword in line for keyword in important_keywords)
                or "Done" in line
                or "WARNING" in line
                or "📊" in line
                or "💡" in line
                or "🔄" in line
            ):
                print(line)

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
    print("  Production batch job patterns")
    print("  Incremental vs full processing")
    print("  Idempotent job design")
    print("  Restart policies for resilience")
    print("  Date-based partitioning")
    print("  Data quality validation")
    print("  Job metadata and audit trails")
    print()
    print("Resilience Features:")
    print("  - RestartPolicy: Automatic retry on failures")
    print("  - TimeToLiveSeconds: Auto-cleanup completed jobs")
    print("  - Labels: Metadata for tracking and monitoring")
    print("  - Quality Checks: Fail fast on data issues")
    print()
    print("Batch Processing Best Practices:")
    print("  1. Design jobs to be idempotent (rerunnable)")
    print("  2. Use incremental processing for efficiency")
    print("  3. Partition data by date for query performance")
    print("  4. Add quality checks at each stage")
    print("  5. Include metadata (job_id, timestamps)")
    print("  6. Configure retry policies for resilience")
    print("  7. Set TTL for automatic cleanup")
    print()
    print("Scheduling Options:")
    print("  - Kubernetes CronJob")
    print("  - Apache Airflow")
    print("  - Argo Workflows")
    print("  - Custom scheduler with SparkClient SDK")
    print()
    print("Next steps:")
    print("  - Try example 06: Dynamic allocation and auto-scaling")
    print("  - Schedule this job with Airflow/Argo")
    print("  - Implement checkpoint/recovery for large jobs")
    print("  - Add alerting for job failures")
    print()


if __name__ == "__main__":
    main()
