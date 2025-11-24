#!/usr/bin/env python3
"""
Title: Scheduled Batch Job with Resilience (using MinIO S3)
Level: 2 (Intermediate - Batch Processing)
Target Audience: Data Engineers building production batch pipelines
Time to Run: ~3-4 minutes

Description:
This example demonstrates production-ready batch processing patterns with scripts
stored in S3-compatible storage (MinIO). You'll learn how to build reliable batch
jobs with versioned scripts in object storage, restart policies, and resilience features.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- MinIO deployed (run ./setup_minio.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- Batch processing with scripts in S3/MinIO
- Restart policies and failure handling
- Production pattern with versioned batch scripts
- Job metadata and audit trails

Real-World Use Case:
Daily data warehouse refresh, nightly ETL jobs with scripts managed in S3.
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

# Import MinIO configuration
try:
    from minio_config import S3_PATHS, get_s3_spark_conf, print_minio_info
except ImportError:
    print("ERROR: minio_config.py not found!")
    print("Please ensure you're running from the examples/spark directory")
    sys.exit(1)


def main():
    """Main example: Submit scheduled batch job from S3 with resilience."""

    print("=" * 80)
    print("EXAMPLE 05: Scheduled Batch Job with Resilience (MinIO S3)")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Production batch job patterns")
    print("  2. Storing batch scripts in S3 (MinIO)")
    print("  3. Restart policies for fault tolerance")
    print("  4. Job metadata and audit trails")
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

    # Step 2: Prepare the application with resilience
    timestamp = datetime.now().strftime("%H%M%S")
    app_name = f"batch-job-{timestamp}"
    batch_date = datetime.now().strftime("%Y-%m-%d")

    # Get S3 path for the batch job script
    script_path = S3_PATHS["batch_job_script"]

    print("Step 2: Configuring batch job with resilience...")
    print(f"  App name: {app_name}")
    print(f"  Batch date: {batch_date}")
    print(f"  Script location: {script_path}")
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

        # Get S3-enabled Spark configuration
        spark_conf = get_s3_spark_conf()

        response = client.submit_application(
            # Application metadata
            app_name=app_name,
            main_application_file=script_path,  # S3 path!
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
            # S3 configuration for MinIO
            spark_conf=spark_conf,
        )

        print("  Batch job submitted successfully!")
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
        print("  Resilience features enabled:")
        print(f"    - Retry on failure: {restart_policy.on_failure_retries} attempts")
        print(f"    - Retry interval: {restart_policy.on_failure_retry_interval}s")
        print("    - Auto-cleanup: After 1 hour")
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
    print("Step 4: Monitoring batch job (this may take 2-3 minutes)...")
    print("  Processing batch data from S3 script...")

    try:
        # Wait for completion with timeout
        final_status = client.wait_for_job_status(
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
        print(f"  You can check status later with: client.get_job('{app_name}')")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Error monitoring job: {e}")
        sys.exit(1)

    # Step 5: Retrieve results from logs
    print("Step 5: Retrieving batch job results...")
    print()

    try:
        logs = list(client.get_job_logs(app_name))

        print("=" * 80)
        print("BATCH JOB RESULTS (from S3 script)")
        print("=" * 80)

        # Display important sections from the batch job script
        important_keywords = [
            "SCHEDULED BATCH JOB",
            "[CONFIG]",
            "[EXTRACT]",
            "[TRANSFORM]",
            "[LOAD]",
            "[COMPLETE]",
            "Batch Configuration",
            "Customer Summary",
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
    print("  Production batch job patterns")
    print("  Storing batch scripts in S3/MinIO")
    print("  Restart policies for resilience")
    print("  Job metadata and audit trails")
    print("  Production pattern with versioned scripts")
    print()
    print("Resilience Features:")
    print("  - RestartPolicy: Automatic retry on failures")
    print("  - TimeToLiveSeconds: Auto-cleanup completed jobs")
    print("  - Labels: Metadata for tracking and monitoring")
    print("  - S3 Versioning: Script rollback capability")
    print()
    print("Batch Processing Best Practices:")
    print("  1. Store scripts in version-controlled S3")
    print("  2. Design jobs to be idempotent (rerunnable)")
    print("  3. Configure retry policies for resilience")
    print("  4. Set TTL for automatic cleanup")
    print("  5. Use labels for job tracking")
    print("  6. Enable S3 versioning for rollback")
    print()
    print("Scheduling Options:")
    print("  - Kubernetes CronJob")
    print("  - Apache Airflow")
    print("  - Argo Workflows")
    print("  - Custom scheduler with SparkClient SDK")
    print()
    print("Production Tips:")
    print("  - Implement CI/CD for batch script deployment")
    print("  - Use S3 versioning for script history")
    print("  - Monitor job metrics and SLAs")
    print("  - Add alerting for job failures")
    print("  - Store job metadata in data catalog")
    print()
    print("Next steps:")
    print("  - Try example 06: Dynamic allocation and auto-scaling")
    print("  - Schedule this job with Airflow/Argo")
    print("  - Implement checkpoint/recovery for large jobs")
    print("  - Read/write data from/to S3 buckets")
    print()


if __name__ == "__main__":
    main()
