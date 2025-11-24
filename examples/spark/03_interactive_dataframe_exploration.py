#!/usr/bin/env python3
"""
Title: Interactive DataFrame Exploration
Level: 1 (Beginner)
Target Audience: Data Scientists doing exploratory data analysis
Time to Run: ~3-4 minutes

Description:
This example demonstrates interactive data exploration patterns commonly used in
Jupyter notebooks and data science workflows. You'll learn how to inspect schemas,
check data quality, compute statistics, and explore relationships in your data.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- Schema inspection and data profiling
- Data quality checks (nulls, duplicates, outliers)
- Descriptive statistics (describe, summary)
- Correlation analysis
- Data sampling and exploration patterns

Real-World Use Case:
Exploratory Data Analysis (EDA), data quality assessment, understanding new datasets.
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


def create_exploration_script():
    """Create a PySpark script for interactive data exploration.

    Returns:
        str: Python code for data exploration
    """
    return """
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as _sum, avg, min as _min, max as _max,
    stddev, variance, corr, isnan, isnull, when, lit,
    countDistinct, approx_count_distinct
)
from pyspark.sql.types import *
import sys

# Create Spark session
spark = SparkSession.builder \\
    .appName("Interactive DataFrame Exploration") \\
    .getOrCreate()

print("\\n" + "="*80)
print("INTERACTIVE DATAFRAME EXPLORATION")
print("="*80)

# Step 1: Create sample customer dataset
print("\\nStep 1: Creating sample customer dataset...")

schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("city", StringType(), True),
    StructField("purchases", IntegerType(), True),
    StructField("total_spent", DoubleType(), True),
    StructField("satisfaction_score", DoubleType(), True),
])

# Sample customer data with some data quality issues (nulls, outliers)
customers_data = [
    (1, "Alice Johnson", 28, "New York", 15, 1250.50, 4.5),
    (2, "Bob Smith", 35, "Los Angeles", 8, 890.25, 4.2),
    (3, "Carol White", None, "Chicago", 22, 2100.00, 4.8),  # Missing age
    (4, "David Brown", 42, "Houston", 5, 450.75, 3.9),
    (5, "Eve Davis", 31, None, 18, 1680.30, 4.6),  # Missing city
    (6, "Frank Miller", 29, "Phoenix", 12, 1050.00, 4.3),
    (7, "Grace Lee", 38, "Philadelphia", 25, 2850.50, 4.9),
    (8, "Henry Wilson", 45, "San Antonio", 3, 280.00, 3.5),
    (9, "Ivy Moore", 26, "San Diego", 20, 1890.75, 4.7),
    (10, "Jack Taylor", 33, "Dallas", None, None, None),  # Missing purchase data
    (11, "Kate Anderson", 27, "San Jose", 16, 1420.50, 4.4),
    (12, "Liam Thomas", 150, "Austin", 2, 195.00, 2.1),  # Outlier age
    (13, "Mia Jackson", 30, "Jacksonville", 14, 1280.25, 4.5),
    (14, "Noah Martinez", 36, "Fort Worth", 9, 820.50, 4.1),
    (15, "Olivia Garcia", 32, "Columbus", 19, 1750.00, 4.6),
]

df = spark.createDataFrame(customers_data, schema)

print(f"  Created DataFrame with {df.count()} customers")
print()

# Step 2: Schema Inspection
print("Step 2: Schema Inspection")
print("-" * 80)
print("\\nDataFrame Schema:")
df.printSchema()

print("\\nColumn Names and Types:")
for field in df.schema.fields:
    nullable = "nullable" if field.nullable else "not null"
    print(f"  - {field.name}: {field.dataType.simpleString()} ({nullable})")

print(f"\\nTotal Columns: {len(df.columns)}")
print(f"Total Rows: {df.count()}")
print()

# Step 3: Preview Data
print("Step 3: Data Preview")
print("-" * 80)
print("\\nFirst 5 rows:")
df.show(5, truncate=False)

print("Random sample (3 rows):")
df.sample(fraction=0.2, seed=42).show(3, truncate=False)
print()

# Step 4: Data Quality Assessment
print("Step 4: Data Quality Assessment")
print("-" * 80)

# Count nulls per column
print("\\nNull value counts:")
null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c) for c in df.columns
])
null_counts.show()

# Count distinct values per column
print("Distinct value counts:")
distinct_counts = df.select([
    countDistinct(col(c)).alias(c) for c in df.columns
])
distinct_counts.show()

# Identify rows with any null values
null_rows = df.filter(
    col("age").isNull() |
    col("city").isNull() |
    col("purchases").isNull()
)
print(f"\\nRows with null values: {null_rows.count()}")
if null_rows.count() > 0:
    print("Rows with nulls:")
    null_rows.show(truncate=False)

print()

# Step 5: Descriptive Statistics
print("Step 5: Descriptive Statistics")
print("-" * 80)
print("\\nSummary statistics for numeric columns:")
df.describe().show()

print("Custom statistics:")
stats_df = df.select([
    count("customer_id").alias("total_customers"),
    avg("age").alias("avg_age"),
    _min("age").alias("min_age"),
    _max("age").alias("max_age"),
    avg("purchases").alias("avg_purchases"),
    avg("total_spent").alias("avg_spent"),
    avg("satisfaction_score").alias("avg_satisfaction"),
])
stats_df.show()

print()

# Step 6: Data Distribution Analysis
print("Step 6: Data Distribution Analysis")
print("-" * 80)

# Age distribution by bins
print("\\nAge distribution:")
df.groupBy("age").count().orderBy("age").show()

# City distribution
print("City distribution:")
df.groupBy("city").count().orderBy(col("count").desc()).show()

# Purchases distribution by ranges
print("Purchases distribution (binned):")
df.select(
    when(col("purchases") < 5, "Low (< 5)")
    .when((col("purchases") >= 5) & (col("purchases") < 15), "Medium (5-14)")
    .when(col("purchases") >= 15, "High (>= 15)")
    .otherwise("Unknown")
    .alias("purchase_range")
).groupBy("purchase_range").count().orderBy(col("count").desc()).show()

print()

# Step 7: Correlation Analysis
print("Step 7: Correlation Analysis")
print("-" * 80)

# Compute correlations between numeric columns
print("\\nCorrelations with total_spent:")
correlations = []
for column in ["age", "purchases", "satisfaction_score"]:
    # Filter out nulls for correlation
    corr_value = df.filter(
        col("total_spent").isNotNull() & col(column).isNotNull()
    ).stat.corr("total_spent", column)
    correlations.append((column, corr_value))
    print(f"  - {column} vs total_spent: {corr_value:.4f}")

print("\\nInterpretation:")
print("  - Correlation close to +1: Strong positive relationship")
print("  - Correlation close to -1: Strong negative relationship")
print("  - Correlation close to 0: Weak or no linear relationship")
print()

# Step 8: Outlier Detection
print("Step 8: Outlier Detection")
print("-" * 80)

# Detect outliers using statistical method (values beyond mean ± 3*stddev)
age_stats = df.select(
    avg("age").alias("mean"),
    stddev("age").alias("stddev")
).collect()[0]

mean_age = age_stats["mean"]
stddev_age = age_stats["stddev"]

print(f"\\nAge statistics:")
print(f"  Mean: {mean_age:.2f}")
print(f"  Std Dev: {stddev_age:.2f}")
print(f"  Normal range: {mean_age - 3*stddev_age:.2f} to {mean_age + 3*stddev_age:.2f}")

outliers = df.filter(
    (col("age") < mean_age - 3*stddev_age) |
    (col("age") > mean_age + 3*stddev_age)
)

print(f"\\nOutliers detected: {outliers.count()}")
if outliers.count() > 0:
    print("Outlier records:")
    outliers.select("customer_id", "name", "age").show()

print()

# Step 9: Data Quality Summary
print("Step 9: Data Quality Summary Report")
print("=" * 80)

total_rows = df.count()
complete_rows = df.na.drop().count()
incomplete_rows = total_rows - complete_rows
completeness_pct = (complete_rows / total_rows) * 100

print(f"\\nData Quality Metrics:")
print(f"  - Total Records: {total_rows}")
print(f"  - Complete Records: {complete_rows}")
print(f"  - Incomplete Records: {incomplete_rows}")
print(f"  - Data Completeness: {completeness_pct:.2f}%")
print(f"  - Outliers Detected: {outliers.count()}")
print(f"  - Unique Customers: {df.select('customer_id').distinct().count()}")
print(f"  - Unique Cities: {df.select('city').distinct().count()}")

print("\\nRecommendations:")
if incomplete_rows > 0:
    print(f"  WARNING: {incomplete_rows} records have missing values - consider imputation")
if outliers.count() > 0:
    print(f"  WARNING: {outliers.count()} outliers detected - review for data quality")
if incomplete_rows == 0 and outliers.count() == 0:
    print("  Dataset appears clean and ready for analysis")

print()

# Step 10: Create cleaned dataset
print("Step 10: Creating Cleaned Dataset")
print("-" * 80)

# Option 1: Drop rows with nulls
cleaned_df = df.na.drop()
print(f"\\nOption 1 - Drop nulls: {cleaned_df.count()} rows remaining")

# Option 2: Fill nulls with defaults
filled_df = df.na.fill({
    "age": int(mean_age),
    "city": "Unknown",
    "purchases": 0,
    "total_spent": 0.0,
    "satisfaction_score": 0.0
})
print(f"Option 2 - Fill nulls: {filled_df.count()} rows (all retained)")

# Option 3: Remove outliers and fill nulls
clean_and_filtered_df = filled_df.filter(
    (col("age") >= mean_age - 3*stddev_age) &
    (col("age") <= mean_age + 3*stddev_age)
)
print(f"Option 3 - Fill nulls + remove outliers: {clean_and_filtered_df.count()} rows")

print("\\nCleaned data sample:")
clean_and_filtered_df.show(5)

print("\\n" + "="*80)
print("EXPLORATION COMPLETE!")
print("="*80)
print("\\nKey Findings:")
num_cities = df.select('city').distinct().count()
print(f"  - Dataset has {df.count()} customers across {num_cities} cities")
avg_purchases = df.agg(avg('purchases')).collect()[0][0]
print(f"  - Average customer: {mean_age:.0f} years old, {avg_purchases:.1f} purchases")
print(f"  - Data completeness: {completeness_pct:.1f}%")
print(f"  - Quality issues: {incomplete_rows} incomplete records, {outliers.count()} outliers")

spark.stop()
"""


def main():
    """Main example: Submit data exploration job and get results."""

    print("=" * 80)
    print("EXAMPLE 03: Interactive DataFrame Exploration")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Schema inspection and data profiling")
    print("  2. Data quality assessment (nulls, outliers)")
    print("  3. Descriptive statistics and distributions")
    print("  4. Correlation analysis")
    print("  5. Data cleaning strategies")
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
    app_name = "dataframe-exploration"

    print("Step 2: Configuring Spark application...")
    print(f"  App name: {app_name}")
    print("  Spark version: 4.0.0")
    print("  Resources: 1 driver + 2 executors")
    print("  Task: Exploratory Data Analysis")
    print()

    # Step 3: Submit the application
    print("Step 3: Submitting data exploration application...")

    try:
        response = client.submit_application(
            # Application metadata
            app_name=app_name,
            # Placeholder
            main_application_file=("local:///opt/spark/examples/src/main/python/pi.py"),
            # Spark configuration
            spark_version="4.0.0",
            app_type="Python",
            # Resource allocation
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
    print("Step 4: Monitoring application (this may take 2-3 minutes)...")
    print("  Performing comprehensive data exploration...")

    try:
        # Wait for completion with timeout
        final_status = client.wait_for_job_status(
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
            )
            print("  Check logs below for details.")

    except TimeoutError:
        print("  ERROR: Application did not complete within 5 minutes")
        print(f"  You can check status later with: client.get_job('{app_name}')")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Error monitoring application: {e}")
        sys.exit(1)

    # Step 5: Retrieve results from logs
    print("Step 5: Retrieving exploration results from logs...")
    print()

    try:
        logs = list(client.get_job_logs(app_name))

        print("=" * 80)
        print("EXPLORATION RESULTS")
        print("=" * 80)

        # Display relevant sections
        important_sections = [
            "INTERACTIVE DATAFRAME EXPLORATION",
            "Schema Inspection",
            "Data Quality Assessment",
            "Descriptive Statistics",
            "Correlation Analysis",
            "Outlier Detection",
            "Data Quality Summary",
            "EXPLORATION COMPLETE",
            "Key Findings",
        ]

        in_section = False
        for line in logs:
            # Check if we're entering an important section
            if any(section in line for section in important_sections):
                in_section = True
                print(line)
            elif in_section:
                print(line)
                # Stay in section until we hit a blank line or new section
                if line.strip() == "" or line.startswith("Step"):
                    in_section = False

        print()
        print("=" * 80)

    except Exception as e:
        print(f"  WARNING: Could not retrieve logs: {e}")
        print("  The job may have completed but logs are not yet available")

    # Step 6: Cleanup
    print()
    print("Step 6: Cleaning up resources...")
    try:
        client.delete_job(app_name)
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
    print("  How to inspect DataFrame schemas")
    print("  Data quality assessment techniques")
    print("  Computing descriptive statistics")
    print("  Correlation analysis")
    print("  Outlier detection methods")
    print("  Data cleaning strategies")
    print()
    print("Key Exploration Patterns:")
    print("  - df.printSchema() - View structure")
    print("  - df.describe() - Summary statistics")
    print("  - df.na.drop() / df.na.fill() - Handle nulls")
    print("  - df.stat.corr() - Correlation analysis")
    print("  - df.sample() - Random sampling")
    print("  - when().otherwise() - Conditional logic")
    print()
    print("Common Data Quality Checks:")
    print("  1. Null value counts")
    print("  2. Distinct value counts (cardinality)")
    print("  3. Outlier detection (statistical methods)")
    print("  4. Duplicate detection")
    print("  5. Data type validation")
    print()
    print("Next steps:")
    print("  - Try example 04: ETL pipeline basics")
    print("  - Apply these techniques to your own datasets")
    print("  - Explore advanced EDA with window functions")
    print("  - Integrate with visualization libraries")
    print()


if __name__ == "__main__":
    main()
