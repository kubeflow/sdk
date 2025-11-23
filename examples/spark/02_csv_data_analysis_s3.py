#!/usr/bin/env python3
"""
Title: CSV Data Analysis with Spark (using MinIO S3)
Level: 1 (Beginner)
Target Audience: Data Scientists analyzing tabular data
Time to Run: ~2-3 minutes

Description:
This example demonstrates how to analyze CSV data using Spark DataFrames with scripts
stored in S3-compatible storage (MinIO). You'll learn to load scripts from object storage
and perform filtering, grouping, and aggregations - the bread and butter of data analysis.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- MinIO deployed (run ./setup_minio.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- Loading PySpark scripts from S3/MinIO
- DataFrame filtering and selection
- Group-by aggregations (sum, avg, count)
- Sorting and limiting results
- Production pattern with object storage

Real-World Use Case:
Sales data analysis, customer analytics, business intelligence reporting with scripts
stored in version-controlled S3 buckets.
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
    BatchSparkClient,
)

# Import MinIO configuration
try:
    from minio_config import S3_PATHS, get_s3_spark_conf, print_minio_info
except ImportError:
    print("ERROR: minio_config.py not found!")
    print("Please ensure you're running from the examples/spark directory")
    sys.exit(1)


def main():
    """Main example: Submit CSV analysis job from S3."""

    print("=" * 80)
    print("EXAMPLE 02: CSV Data Analysis (with MinIO S3)")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Storing PySpark scripts in S3 (MinIO)")
    print("  2. Loading CSV data and performing analysis")
    print("  3. DataFrame filtering and transformations")
    print("  4. Group-by aggregations (sum, avg, count)")
    print("  5. Multi-dimensional analysis (category, region)")
    print()

    # Show MinIO configuration
    print_minio_info()

    # Step 1: Create SparkClient with configuration
    print("Step 1: Creating Spark client...")
    config = OperatorBackendConfig(
        namespace=os.getenv("SPARK_NAMESPACE", "default"),
        service_account="spark-operator-spark",
        default_spark_image="docker.io/library/spark",
        context=os.getenv("KUBE_CONTEXT", "kind-spark-test"),
        enable_monitoring=False,
        enable_ui=True,  # Enable Spark UI
    )
    client = BatchSparkClient(backend_config=config)
    print("  Client created successfully")
    print("  Spark UI enabled")
    print()

    # Step 2: Prepare the application
    timestamp = datetime.now().strftime("%H%M%S")
    app_name = f"csv-analysis-{timestamp}"

    # Get S3 path for the CSV analysis script
    script_path = S3_PATHS["csv_analysis_script"]

    print("Step 2: Configuring Spark application with S3 storage...")
    print(f"  App name: {app_name}")
    print("  Spark version: 4.0.0")
    print(f"  Script location: {script_path}")
    print("  Resources: 1 driver + 2 executors")
    print()

    # Step 3: Submit the application
    print("Step 3: Submitting application from S3...")

    try:
        # Get S3-enabled Spark configuration
        spark_conf = get_s3_spark_conf()

        response = client.submit_application(
            # Application metadata
            app_name=app_name,
            main_application_file=script_path,  # S3 path!
            # Spark configuration
            spark_version="4.0.0",
            app_type="Python",
            # Resource allocation
            driver_cores=1,
            driver_memory="1g",  # More memory for JAR downloads
            executor_cores=1,
            executor_memory="1g",
            num_executors=2,
            # Keep pods for debugging (30 minutes)
            time_to_live_seconds=1800,
            # S3 configuration for MinIO
            spark_conf=spark_conf,
        )

        print("  Application submitted successfully!")
        print(f"  Submission ID: {response.submission_id}")
        print(f"  Status: {response.status}")
        print("  Script loaded from S3: Done")
        print()
        print("  🌐 Spark UI Access (choose one):")
        print("     Option 1 - Direct to driver pod:")
        print(f"       kubectl port-forward pod/{app_name}-driver 4040:4040")
        print("     Option 2 - Via service (if created by operator):")
        print(f"       kubectl port-forward svc/{app_name}-ui-svc 4040:4040")
        print("     Then open: http://localhost:4040")
        print()
        print("  💡 Tip: Use Option 1 if service doesn't exist")
        print()

    except Exception as e:
        print(f"  ERROR: Submission failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure MinIO is running:")
        print("     kubectl get pods -l app=minio")
        print("  2. Verify scripts are uploaded:")
        print("     kubectl exec minio-client -- mc ls myminio/spark-scripts/")
        print("  3. Check if setup_minio.sh was run successfully")
        sys.exit(1)

    # Step 4: Monitor the application
    print("Step 4: Monitoring application (this may take 1-2 minutes)...")
    print("  Executing CSV analysis from S3 script...")

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
        print("CSV ANALYSIS RESULTS (from S3 script)")
        print("=" * 80)

        # Display important sections from the analysis script
        important_keywords = [
            "CSV DATA ANALYSIS",
            "Sample Data:",
            "Sales by Category:",
            "category",
            "products",
        ]

        found_results = False
        for line in logs:
            if any(keyword in line for keyword in important_keywords):
                print(line)
                found_results = True
            elif found_results and ("+" in line or "|" in line):
                # Print table output
                print(line)

        if not found_results:
            print("Showing last 30 log lines:")
            for line in logs[-30:]:
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
    print("  How to store PySpark scripts in S3/MinIO")
    print("  How to configure Spark for S3 access")
    print("  How to submit applications from object storage")
    print("  DataFrame filtering and transformations")
    print("  Group-by aggregations")
    print()
    print("S3 Configuration Used:")
    print("  - spark.hadoop.fs.s3a.endpoint - MinIO endpoint")
    print("  - spark.hadoop.fs.s3a.access.key - Access credentials")
    print("  - spark.hadoop.fs.s3a.path.style.access - MinIO compatibility")
    print()
    print("Production Tips:")
    print("  - Store scripts in version-controlled S3 buckets")
    print("  - Use IAM roles instead of access keys (in AWS)")
    print("  - Enable S3 versioning for script history")
    print("  - Implement CI/CD pipeline for script deployment")
    print()
    print("Next steps:")
    print("  - Try example 03: Interactive DataFrame exploration (S3)")
    print("  - Try example 04: ETL pipeline (S3)")
    print("  - Upload your own CSV data to MinIO")
    print("  - Read/write data from/to S3 in your scripts")
    print()


if __name__ == "__main__":
    main()
