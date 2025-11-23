#!/usr/bin/env python3
"""
Simple test WITHOUT History Server to verify basic Spark works
"""

from datetime import datetime
import os
import sys

from kubernetes import client, config

# Add SDK to path
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

print("=" * 80)
print("SIMPLE SPARK TEST (No History Server)")
print("=" * 80)
print()

# Load kubeconfig
try:
    config.load_kube_config(context="kind-spark-test")
except:
    config.load_incluster_config()

# Create API client
api_client = client.CustomObjectsApi()

# Create minimal SparkApplication
app_name = f"simple-test-{datetime.now().strftime('%H%M%S')}"

spark_app = {
    "apiVersion": "sparkoperator.k8s.io/v1beta2",
    "kind": "SparkApplication",
    "metadata": {
        "name": app_name,
        "namespace": "default",
    },
    "spec": {
        "type": "Python",
        "mode": "cluster",
        "image": "docker.io/library/spark:4.0.0",
        "imagePullPolicy": "IfNotPresent",
        "mainApplicationFile": "local:///opt/spark/examples/src/main/python/pi.py",
        "arguments": ["10"],
        "sparkVersion": "4.0.0",
        "restartPolicy": {"type": "Never"},
        "timeToLiveSeconds": 1800,
        "driver": {
            "cores": 1,
            "memory": "512m",
            "serviceAccount": "spark-operator-spark",
            "labels": {
                "version": "4.0.0",
            },
        },
        "executor": {
            "cores": 1,
            "instances": 2,
            "memory": "512m",
            "labels": {
                "version": "4.0.0",
            },
        },
        "sparkConf": {
            "spark.kubernetes.file.upload.path": "/tmp",
        },
    },
}

print(f"Submitting simple Spark application: {app_name}")
print("This test has NO volume mounts - just basic Pi calculation")
print()

try:
    # Create the SparkApplication
    response = api_client.create_namespaced_custom_object(
        group="sparkoperator.k8s.io",
        version="v1beta2",
        namespace="default",
        plural="sparkapplications",
        body=spark_app,
    )

    print("Application submitted successfully!")
    print(f"  Name: {app_name}")
    print()
    print("Monitor:")
    print(f"  kubectl get sparkapplication {app_name} -w")
    print()
    print("View logs:")
    print(f"  kubectl logs {app_name}-driver -f")
    print()
    print("Describe:")
    print(f"  kubectl describe sparkapplication {app_name}")
    print()

    # Wait and show status
    import time

    print("Waiting for completion...")
    for i in range(60):
        time.sleep(2)
        try:
            app_status = api_client.get_namespaced_custom_object(
                group="sparkoperator.k8s.io",
                version="v1beta2",
                namespace="default",
                plural="sparkapplications",
                name=app_name,
            )

            state = app_status.get("status", {}).get("applicationState", {}).get("state", "UNKNOWN")
            print(f"  Status: {state}", end="\r")

            if state in ["COMPLETED", "FAILED"]:
                print()
                print(f"\nApplication {state}")

                if state == "COMPLETED":
                    print("\nSUCCESS! Basic Spark works without volumes.")
                    print("\nNow we can test with History Server volumes.")
                else:
                    error_msg = (
                        app_status.get("status", {})
                        .get("applicationState", {})
                        .get("errorMessage", "Unknown error")
                    )
                    print(f"\nERROR: FAILED: {error_msg}")
                    print("\nCheck logs:")
                    print(f"  kubectl logs {app_name}-driver")
                    break

        except Exception:
            continue

except Exception as e:
    print(f"ERROR: Failed to submit application: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
