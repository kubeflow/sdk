#!/usr/bin/env python3
"""
Long-Running Job for Spark UI Validation

This example submits a 10-minute Spark job specifically designed to test
and validate Spark UI access. The job performs various operations to showcase
all UI features.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- MinIO deployed (run ./setup_minio.sh)
- Long-running job script uploaded to MinIO

Time to Run: ~10 minutes

Usage:
    python run_long_job_ui_validation.py [--no-monitor]

Options:
    --no-monitor    Skip interactive monitoring (useful when called from scripts)
"""

from datetime import datetime
import os
import sys
import time

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
    from minio_config import get_s3_spark_conf, print_minio_info
except ImportError:
    print("ERROR: minio_config.py not found!")
    print("Please ensure you're running from the examples/spark directory")
    sys.exit(1)


def print_ui_instructions(app_name: str):
    """Print detailed UI access instructions."""
    print("=" * 80)
    print("🌐 SPARK UI ACCESS INSTRUCTIONS")
    print("=" * 80)
    print()
    print("The job will run for ~10 minutes. Follow these steps to access the UI:")
    print()
    print("STEP 1: Wait for driver pod to be Running")
    print("-" * 80)
    print(f"  kubectl get pod {app_name}-driver -w")
    print()
    print("  Wait until STATUS shows 'Running' (may take 1-2 minutes)")
    print()
    print("STEP 2: Port-forward to driver pod")
    print("-" * 80)
    print(f"  kubectl port-forward pod/{app_name}-driver 4040:4040")
    print()
    print("  Keep this terminal open!")
    print()
    print("STEP 3: Open Spark UI in browser")
    print("-" * 80)
    print("  http://localhost:4040")
    print()
    print("STEP 4: Explore UI features while job runs")
    print("-" * 80)
    print("  Jobs tab - See 6 jobs (one per stage)")
    print("  Stages tab - Monitor stage progress in real-time")
    print("  Storage tab - View cached DataFrame (after Stage 2)")
    print("  Executors tab - Check executor metrics and GC")
    print("  SQL tab - Inspect DataFrame query plans")
    print("  Environment tab - View Spark configuration")
    print()
    print("=" * 80)
    print()
    print("💡 TIPS:")
    print("  - Job progresses through 5 stages over 10 minutes")
    print("  - Each stage pauses briefly - perfect for exploring UI")
    print("  - Stage 2 caches data - check Storage tab!")
    print("  - Stage 3 does heavy shuffling - watch Executors tab")
    print("  - Click on job/stage names for detailed views")
    print()
    print("=" * 80)
    print()


def monitor_job_progress(client: BatchSparkClient, app_name: str):
    """Monitor and display job progress."""
    print("=" * 80)
    print("MONITORING JOB PROGRESS")
    print("=" * 80)
    print()
    print("Checking status every 30 seconds...")
    print("Press Ctrl+C to stop monitoring (job will continue running)")
    print()

    start_time = time.time()
    last_state = None

    try:
        while True:
            try:
                status = client.get_status(app_name)
                elapsed = int(time.time() - start_time)

                if status.state != last_state:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] [{elapsed:3d}s] State: {status.state.value}")
                    last_state = status.state

                if status.state in [ApplicationState.COMPLETED, ApplicationState.FAILED]:
                    print()
                    print(f"Job finished with state: {status.state.value}")
                    return status

                time.sleep(30)

            except Exception as e:
                print(f"  Warning: Could not get status: {e}")
                time.sleep(30)

    except KeyboardInterrupt:
        print()
        print("Stopped monitoring (job still running)")
        print(f"Check status with: kubectl get sparkapplication {app_name}")
        return None


def main():
    """Main example: Submit long-running job for UI validation."""

    # Check for --no-monitor flag
    no_monitor = "--no-monitor" in sys.argv

    print("=" * 80)
    print("LONG-RUNNING SPARK JOB FOR UI VALIDATION")
    print("=" * 80)
    print()
    print("This example submits a 10-minute job designed to showcase")
    print("all Spark UI features. Perfect for testing UI access!")
    print()
    print("Job stages:")
    print("  Stage 1: Generate 100M rows (~2 min)")
    print("  Stage 2: Cache and aggregate (~2 min)")
    print("  Stage 3: Shuffle-heavy joins (~3 min)")
    print("  Stage 4: Multi-dimensional analysis (~2 min)")
    print("  Stage 5: Window functions (~1 min)")
    print()
    print("Total duration: ~10 minutes")
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
    app_name = f"long-job-{timestamp}"

    # Get S3 path for the long-running job script
    script_path = "s3a://spark-scripts/long_running_job.py"

    print("Step 2: Configuring long-running job...")
    print(f"  App name: {app_name}")
    print("  Spark version: 4.0.0")
    print(f"  Script location: {script_path}")
    print("  Duration: ~10 minutes")
    print("  Resources: 1 driver + 2 executors (1 CPU, 2g RAM each)")
    print()

    # Step 3: Submit the application
    print("Step 3: Submitting long-running job...")
    print()

    try:
        # Get S3-enabled Spark configuration
        spark_conf = get_s3_spark_conf()

        response = client.submit_application(
            # Application metadata
            app_name=app_name,
            main_application_file=script_path,
            # Spark configuration
            spark_version="4.0.0",
            app_type="Python",
            # Resource allocation (more resources for better performance)
            driver_cores=1,
            driver_memory="2g",  # More memory for large datasets
            executor_cores=1,
            executor_memory="2g",  # More memory for shuffles
            num_executors=2,
            # Keep job running for debugging
            time_to_live_seconds=7200,  # 2 hours
            # Labels for tracking
            labels={
                "job_type": "ui-validation",
                "duration": "long",
            },
            # S3 configuration for MinIO
            spark_conf=spark_conf,
        )

        print("  Job submitted successfully!")
        print(f"  Submission ID: {response.submission_id}")
        print(f"  Status: {response.status}")
        print()

        # Print app name for automation scripts to capture
        print(f"APP_NAME={app_name}")
        print()

    except Exception as e:
        print(f"  ERROR: Submission failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure MinIO is running:")
        print("     kubectl get pods -l app=minio")
        print("  2. Verify script is uploaded:")
        print("     kubectl exec minio-client -- mc ls myminio/spark-scripts/")
        print("  3. Run: ./setup_minio.sh to upload scripts")
        sys.exit(1)

    # Step 4: Print UI access instructions
    print()
    print_ui_instructions(app_name)

    # Step 5: Ask if user wants to monitor
    print()

    # Check if monitoring was disabled via flag or non-interactive mode
    if no_monitor:
        print("Monitoring disabled (--no-monitor flag)")
        response = "n"
    elif not sys.stdin.isatty():
        # Non-interactive mode (running from automation script)
        print("Running in non-interactive mode. Skipping monitoring.")
        response = "n"
    else:
        # Interactive mode - ask user
        response = input("Monitor job progress? (y/n): ").strip().lower()

    if response == "y":
        print()
        monitor_job_progress(client, app_name)

        # Retrieve logs after completion
        print()
        print("Retrieving job logs...")
        try:
            logs = list(client.get_logs(app_name))
            print()
            print("=" * 80)
            print("JOB OUTPUT (Last 50 lines)")
            print("=" * 80)
            for line in logs[-50:]:
                print(line)
            print("=" * 80)
        except Exception as e:
            print(f"  WARNING: Could not retrieve logs: {e}")
    else:
        print()
        print("Skipping monitoring.")
        print()
        print("Check job status anytime with:")
        print(f"  kubectl get sparkapplication {app_name}")
        print()
        print("View logs with:")
        print(f"  kubectl logs {app_name}-driver")
        print()

    print()
    print("=" * 80)
    print("QUICK REFERENCE")
    print("=" * 80)
    print()
    print("Port-forward to UI:")
    print(f"  kubectl port-forward pod/{app_name}-driver 4040:4040")
    print()
    print("Open UI:")
    print("  http://localhost:4040")
    print()
    print("Check status:")
    print(f"  kubectl get sparkapplication {app_name} -w")
    print()
    print("View driver logs:")
    print(f"  kubectl logs {app_name}-driver -f")
    print()
    print("Delete when done:")
    print(f"  kubectl delete sparkapplication {app_name}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
