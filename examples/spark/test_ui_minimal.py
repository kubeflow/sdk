#!/usr/bin/env python3
"""
Minimal test for Spark UI service creation without S3.
Uses local:// path and simple SparkPi example.
"""

from datetime import datetime
import os
import sys
import time

# Add SDK to path
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from kubeflow.spark import OperatorBackendConfig, SparkClient

print("=" * 80)
print("MINIMAL TEST: Spark UI Service Creation")
print("=" * 80)
print()
print("This test submits a simple Spark application and checks if")
print("the UI service is created by the Spark Operator.")
print()

# Create client with UI enabled
config = OperatorBackendConfig(
    namespace="default",
    service_account="spark-operator-spark",
    default_spark_image="docker.io/apache/spark",  # Use official image
    context=os.getenv("KUBE_CONTEXT", "kind-spark-test"),
    enable_monitoring=False,
    enable_ui=True,  # Enable UI!
)

client = BatchSparkClient(backend_config=config)
print("Client created with enable_ui=True")
print()

# Submit a simple SparkPi example (built into Spark image)
timestamp = datetime.now().strftime("%H%M%S")
app_name = f"test-ui-{timestamp}"

print(f"Submitting test application: {app_name}")
print("-" * 80)

try:
    response = client.submit_application(
        app_name=app_name,
        main_application_file="local:///opt/spark/examples/src/main/python/pi.py",
        spark_version="3.5.0",
        app_type="Python",
        driver_cores=1,
        driver_memory="512m",
        executor_cores=1,
        executor_memory="512m",
        num_executors=1,
        arguments=["10"],  # Calculate pi with 10 partitions
    )

    print(f"Application submitted: {response.submission_id}")
    print(f"  Status: {response.status}")
    print()

except Exception as e:
    print(f"ERROR: Submission failed: {e}")
    sys.exit(1)

# Wait a few seconds for operator to process
print("Waiting 10 seconds for Spark Operator to create resources...")
time.sleep(10)
print()

# Instructions for checking
print("=" * 80)
print("Now check if the UI service was created:")
print("=" * 80)
print()
print("1. Check for the UI service:")
print(f"   kubectl get svc {app_name}-ui-svc -n default")
print()
print("2. If service exists, port-forward to access:")
print(f"   kubectl port-forward svc/{app_name}-ui-svc 4040:4040")
print("   Then open: http://localhost:4040")
print()
print("3. Check the SparkApplication YAML:")
print(f"   kubectl get sparkapplication {app_name} -o yaml | grep -A 5 sparkUIOptions")
print()
print("4. Check all services:")
print("   kubectl get svc -n default")
print()
print("5. View Spark Operator logs:")
print("   kubectl logs -n spark-operator deploy/spark-operator --tail=100")
print()
print("6. Watch application status:")
print(f"   kubectl get sparkapplication {app_name} -w")
print()
print("=" * 80)
print()
print(f"Application name: {app_name}")
print("The application will run for ~30 seconds.")
print("Check if the UI service exists while it's running!")
print()
