#!/usr/bin/env python3
"""
Title: Hello Spark - Calculate Pi
Level: 1 (Beginner)
Target Audience: Data Scientists new to Spark
Time to Run: ~2-3 minutes

Description:
Your first Spark job! This example demonstrates how to submit a simple PySpark application
that calculates Pi using the Monte Carlo method - a classic distributed computing example
that shows how Spark distributes work across executors.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- How to create a SparkClient
- Submit a PySpark application
- Wait for job completion
- Retrieve and parse job logs
- Clean up resources

Real-World Use Case:
Distributed computation, parallel processing, Monte Carlo simulations.
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


def main():
    """Main example: Submit Pi calculation job and get results."""

    print("=" * 80)
    print("EXAMPLE 01: Hello Spark - Calculate Pi")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Creating a Spark client")
    print("  2. Submitting a PySpark application (Calculate Pi)")
    print("  3. Monitoring job progress")
    print("  4. Retrieving results from logs")
    print()

    # Step 1: Create SparkClient with configuration
    print("Step 1: Creating Spark client...")
    config = OperatorBackendConfig(
        namespace=os.getenv("SPARK_NAMESPACE", "default"),
        service_account="spark-operator-spark",
        default_spark_image="docker.io/library/spark",
        context=os.getenv("KUBE_CONTEXT", "kind-spark-test"),
        enable_monitoring=False,  # Keep it simple for beginners
        enable_ui=False,  # We'll enable this in later examples
    )
    client = BatchSparkClient(backend_config=config)
    print("  Client created successfully")
    print()

    # Step 2: Prepare the application
    # Use timestamp to ensure unique name each run
    timestamp = datetime.now().strftime("%H%M%S")
    app_name = f"hello-spark-{timestamp}"

    print("Step 2: Configuring Spark application...")
    print(f"  App name: {app_name}")
    print("  Spark version: 4.0.0")
    print("  Resources: 1 driver + 2 executors")
    print("  Memory: 512m per container")
    print("  Example: Calculate Pi using Monte Carlo method")
    print()

    # Step 3: Submit the application
    print("Step 3: Submitting application to Kubernetes...")

    try:
        response = client.submit_application(
            # Application metadata
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
            # Arguments for pi calculation (number of samples)
            arguments=["10"],  # Calculate Pi with 10 partitions
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
    print("  Waiting for pods to start...")

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
        print("  You can check status later with: client.get_status('{app_name}')")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Error monitoring application: {e}")
        sys.exit(1)

    # Step 5: Retrieve results from logs
    print("Step 5: Retrieving application logs and results...")
    print()

    try:
        logs = list(client.get_logs(app_name))

        # Parse and display results
        print("=" * 80)
        print("CALCULATION RESULTS:")
        print("=" * 80)

        # Find the Pi calculation result
        pi_found = False
        for line in logs:
            if "Pi is roughly" in line:
                print(f"\n{line}\n")
                pi_found = True
                break

        if not pi_found:
            # Show last 20 lines if Pi result not found
            print("Recent log lines:")
            for line in logs[-20:]:
                print(line)

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
    print("  - How to create a SparkClient")
    print("  - How to submit a PySpark application")
    print("  - How to wait for completion")
    print("  - How to retrieve logs")
    print("  - How to clean up resources")
    print()
    print("Key SDK Methods:")
    print("  - BatchSparkClient(backend_config=config) - Create client")
    print("  - client.submit_application(...) - Submit Spark job")
    print("  - client.wait_for_completion(...) - Monitor job")
    print("  - client.get_logs(...) - Retrieve logs")
    print("  - client.delete_application(...) - Cleanup")
    print()
    print("Next steps:")
    print("  - Try example 02: CSV data analysis")
    print("  - Try example 03: Interactive DataFrame exploration")
    print("  - Modify driver/executor resources")
    print("  - Try with different Spark versions")
    print()


if __name__ == "__main__":
    main()
