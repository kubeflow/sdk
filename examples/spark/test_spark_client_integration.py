"""
Integration tests for Kubeflow Spark Client with Operator Backend.

These tests require:
1. A Kubernetes cluster with Spark Operator installed
2. kubectl configured with proper context
3. Service account 'spark-operator-spark' with proper permissions

Setup:
    Run ./setup_test_environment.sh to create a Kind cluster with Spark Operator

Usage:
    python test_spark_client_integration.py
"""

import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

import time
import unittest

from kubeflow.spark import ApplicationState, OperatorBackendConfig, SparkClient  # noqa: E402


class TestSparkClientIntegration(unittest.TestCase):
    """Integration tests for SparkClient with Operator backend."""

    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        config = OperatorBackendConfig(
            namespace=os.getenv("SPARK_NAMESPACE", "default"),
            service_account="spark-operator-spark",
            default_spark_image="docker.io/library/spark",
            # Explicitly set context
            context=os.getenv("KUBE_CONTEXT", "kind-spark-test"),
            # Disable to avoid JMX agent issue with Spark 4.0
            enable_monitoring=False,
            # Disable UI for simpler testing
            enable_ui=False,
        )
        cls.client = BatchSparkClient(backend_config=config)
        cls.submitted_apps = []

    @classmethod
    def tearDownClass(cls):
        """Clean up submitted applications."""
        print("\nCleaning up test applications...")
        for app_name in cls.submitted_apps:
            try:
                cls.client.delete_application(app_name)
                print(f"  Deleted {app_name}")
            except Exception as e:
                print(f"  ✗ Failed to delete {app_name}: {e}")

    def test_01_submit_spark_pi(self):
        """Test submitting a simple Spark Pi application."""
        print("\n" + "=" * 80)
        print("TEST: Submit Spark Pi Application")
        print("=" * 80)

        app_name = "test-spark-pi"

        response = self.client.submit_application(
            app_name=app_name,
            main_application_file=(
                "local:///opt/spark/examples/jars/spark-examples_2.13-4.0.0.jar"
            ),
            main_class="org.apache.spark.examples.SparkPi",
            spark_version="4.0.0",
            app_type="Scala",
            driver_cores=1,
            driver_memory="512m",
            executor_cores=1,
            executor_memory="512m",
            num_executors=1,
            arguments=["100"],
            spark_conf={
                # Required for Spark 4.0
                "spark.kubernetes.file.upload.path": "/tmp",
            },
        )

        self.submitted_apps.append(app_name)

        self.assertEqual(response.submission_id, app_name)
        self.assertEqual(response.status, "SUBMITTED")

        print(f"Application submitted: {app_name}")
        print(f"  Status: {response.status}")

    def test_02_get_status(self):
        """Test getting application status."""
        print("\n" + "=" * 80)
        print("TEST: Get Application Status")
        print("=" * 80)

        if not self.submitted_apps:
            self.skipTest("No applications to check status")

        app_name = self.submitted_apps[0]
        status = self.client.get_status(app_name)

        self.assertIsNotNone(status)
        self.assertEqual(status.submission_id, app_name)
        self.assertIn(status.state, list(ApplicationState))

        print(f"Got status for {app_name}")
        print(f"  State: {status.state.value}")
        print(f"  App ID: {status.app_id}")

    def test_03_list_applications(self):
        """Test listing applications."""
        print("\n" + "=" * 80)
        print("TEST: List Applications")
        print("=" * 80)

        apps = self.client.list_applications()

        self.assertIsInstance(apps, list)
        print(f"Listed {len(apps)} applications")

        for app in apps[:5]:  # Show first 5
            print(f"  - {app.app_name}: {app.state.value}")

    def test_04_get_logs(self):
        """Test getting application logs."""
        print("\n" + "=" * 80)
        print("TEST: Get Application Logs")
        print("=" * 80)

        if not self.submitted_apps:
            self.skipTest("No applications to get logs from")

        app_name = self.submitted_apps[0]

        # Wait for driver pod to be ready before fetching logs
        print("Waiting for driver pod to be ready...")
        is_ready = self.client.wait_for_pod_ready(app_name, timeout=120)

        if not is_ready:
            print("WARNING: Driver pod not ready within timeout, logs may be empty")

        logs = list(self.client.get_logs(app_name))

        # Logs might be empty if pod not started yet
        print(f"Retrieved {len(logs)} log lines from {app_name}")
        if logs:
            print("\n  First 5 lines:")
            for line in logs[:5]:
                print(f"    {line}")
        else:
            print("  (No logs available yet - pod may still be starting)")

    def test_05_wait_for_completion(self):
        """Test waiting for application completion."""
        print("\n" + "=" * 80)
        print("TEST: Wait for Completion")
        print("=" * 80)

        app_name = "test-spark-pi-completion"

        response = self.client.submit_application(
            app_name=app_name,
            main_application_file=(
                "local:///opt/spark/examples/jars/spark-examples_2.13-4.0.0.jar"
            ),
            main_class="org.apache.spark.examples.SparkPi",
            spark_version="4.0.0",
            app_type="Scala",
            driver_cores=1,
            driver_memory="512m",
            executor_cores=1,
            executor_memory="512m",
            num_executors=1,
            arguments=["10"],  # Small workload
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
            },
        )

        self.submitted_apps.append(app_name)

        print(f"Submitted {app_name}")
        print("  Waiting for completion (timeout: 300s)...")

        final_status = self.client.wait_for_completion(app_name, timeout=300, polling_interval=5)

        print("Application completed")
        print(f"  Final state: {final_status.state.value}")

        self.assertIn(
            final_status.state,
            [ApplicationState.COMPLETED, ApplicationState.FAILED],
        )

    def test_06_delete_application(self):
        """Test deleting an application."""
        print("\n" + "=" * 80)
        print("TEST: Delete Application")
        print("=" * 80)

        # Submit a temporary application
        app_name = "test-spark-delete"

        response = self.client.submit_application(
            app_name=app_name,
            main_application_file=(
                "local:///opt/spark/examples/jars/spark-examples_2.13-4.0.0.jar"
            ),
            main_class="org.apache.spark.examples.SparkPi",
            spark_version="4.0.0",
            app_type="Scala",
            driver_cores=1,
            driver_memory="512m",
            executor_cores=1,
            executor_memory="512m",
            num_executors=1,
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
            },
        )

        print(f"Submitted {app_name}")

        # Delete immediately
        result = self.client.delete_application(app_name)

        self.assertIsInstance(result, dict)
        print(f"Deleted {app_name}")
        print(f"  Result: {result}")

    def test_07_dynamic_allocation(self):
        """Test application with dynamic allocation."""
        print("\n" + "=" * 80)
        print("TEST: Dynamic Allocation")
        print("=" * 80)

        app_name = "test-dynamic-allocation"

        response = self.client.submit_application(
            app_name=app_name,
            main_application_file=(
                "local:///opt/spark/examples/jars/spark-examples_2.13-4.0.0.jar"
            ),
            main_class="org.apache.spark.examples.SparkPi",
            spark_version="4.0.0",
            app_type="Scala",
            driver_cores=1,
            driver_memory="512m",
            executor_cores=1,
            executor_memory="512m",
            num_executors=2,
            arguments=["1000"],
            enable_dynamic_allocation=True,
            initial_executors=1,
            min_executors=1,
            max_executors=5,
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
            },
        )

        self.submitted_apps.append(app_name)

        print(f"Submitted {app_name} with dynamic allocation")
        print("  Config: min=1, max=5, initial=1")

        # Check status after a bit
        time.sleep(10)
        status = self.client.get_status(app_name)

        print(f"  Current state: {status.state.value}")
        if status.executor_state:
            print(f"  Executors: {len(status.executor_state)}")


def run_tests():
    """Run integration tests."""
    print("=" * 80)
    print(" Kubeflow Spark Client - Integration Tests")
    print("=" * 80)
    print()
    print("Prerequisites:")
    print("  - Kubernetes cluster with Spark Operator")
    print("  - kubectl configured with proper context")
    print("  - Service account 'spark-operator-spark'")
    print()
    print("Run ./setup_test_environment.sh if not already done")
    print("=" * 80)
    print()

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSparkClientIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\nAll tests passed! 🎉")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(run_tests())
