#!/usr/bin/env python3
"""
SparkClient Examples - KEP-107 Three-Level API Pattern

This example demonstrates the three usage levels of SparkClient,
following the KEP-107 specification:

1. Level 1 (Minimal): Use all defaults
2. Level 2 (Simple): Configure with simple parameters
3. Level 3 (Advanced): Use Driver/Executor objects for full control

Usage:
    # Run directly:
    python examples/spark_connect_simple.py

    # Or in IPython:
    %run examples/spark_connect_simple.py

    # Or interactive mode:
    python -i examples/spark_connect_simple.py
"""

import os
import uuid

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import Driver, Executor, Name, SparkClient


def _e2e_session_name(base: str) -> str:
    """In E2E in-cluster runs use a unique name to avoid conflicts with existing resources."""
    if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") == "1":
        return f"{base}-{uuid.uuid4().hex[:8]}"
    return base


def _backend_config(namespace_default: str = "default"):
    """Backend config; uses SPARK_TEST_NAMESPACE in CI."""
    return KubernetesBackendConfig(
        namespace=os.environ.get("SPARK_TEST_NAMESPACE", namespace_default)
    )


def example_level1_minimal():
    """
    Level 1: Minimal Usage (KEP-107 lines 104-109)

    Uses all default values:
    - Default namespace
    - Default executor count
    - Default resource allocations
    - Auto-generated session name (spark-connect-{uuid})
    """
    print("=" * 70)
    print("LEVEL 1: MINIMAL USAGE (Auto-generated name)")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())
    # Use Name option to track session for cleanup
    session_name = _e2e_session_name("spark-connect-minimal")
    spark = client.connect(
        spark_conf={"spark.serializer": "org.apache.spark.serializer.KryoSerializer"},
        options=[Name(session_name)],
    )

    # Use Spark - simple data operations
    df = spark.range(10)
    print(f"\nGenerated range with {df.count()} rows")
    print(f"Session name: {session_name}")
    df.show()

    spark.stop()
    # Clean up K8s resources to free cluster capacity for next example
    client.delete_session(session_name)
    print("\nLevel 1 complete.\n")


def example_level2_simple():
    """
    Level 2: Simple Parameters (KEP-107 lines 54-67)

    Configure executors and resources using simple parameters:
    - num_executors: Number of executor instances
    - resources_per_executor: Resource dict with cpu, memory
    - spark_conf: Spark configuration properties
    - options: Use Name option for custom session name
    """
    print("=" * 70)
    print("LEVEL 2: SIMPLE CONFIGURATION (With custom name)")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())
    session_name = _e2e_session_name("my-simple-session")
    spark = client.connect(
        num_executors=5,
        resources_per_executor={"cpu": "2", "memory": "4Gi"},
        spark_conf={
            "spark.sql.adaptive.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        },
        options=[Name(session_name)],
    )

    # Use Spark - more data operations
    df = spark.range(100)
    print(f"\nGenerated range with {df.count()} rows across 5 executors")
    print(f"Session name: {session_name}")
    df.show(10)

    spark.stop()
    # Clean up K8s resources to free cluster capacity for next example
    client.delete_session(session_name)
    print("\nLevel 2 complete.\n")


def example_level3_advanced():
    """
    Level 3: Advanced Configuration (KEP-107 advanced example)

    Full control using Driver and Executor objects:
    - Driver: Configure driver pod resources and service account
    - Executor: Configure executor instances and resources per executor
    - Supports GPUs and custom service accounts
    - options: Use Name option for custom session name
    """
    print("=" * 70)
    print("LEVEL 3: ADVANCED CONFIGURATION (With custom name)")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())
    session_name = _e2e_session_name("advanced-session-name")

    spark = client.connect(
        driver=Driver(
            resources={"cpu": "1", "memory": "2Gi"},
            # service_account="spark-driver",  # Optional: custom service account
        ),
        executor=Executor(
            num_instances=2,  # Reduced for Kind cluster compatibility
            resources_per_executor={
                "cpu": "1",
                "memory": "2Gi",
                # Add GPU if needed: "nvidia.com/gpu": "1"
            },
        ),
        spark_conf={
            "spark.app.name": "advanced-spark-app",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        },
        options=[Name(session_name)],
    )

    # Use Spark - complex data operations
    df = spark.range(1000)
    print(f"\nGenerated range with {df.count()} rows across 2 executors")
    print(f"Session name: {session_name}")
    df.show(10)

    spark.stop()
    # Clean up K8s resources to free cluster capacity for next example
    client.delete_session(session_name)
    print("\nLevel 3 complete.\n")


def example_connect_existing():
    """
    Connect to Existing Spark Connect Server (KEP-107 lines 88-97)

    Connect to a server that's already running using its URL.
    """
    print("=" * 70)
    print("CONNECT TO EXISTING SERVER")
    print("=" * 70)

    SparkClient(backend_config=_backend_config())

    # Example URL - replace with your actual server URL
    # spark = client.connect(base_url="sc://spark-server:15002")

    print("\nTo connect to existing server, use:")
    print('spark = client.connect(base_url="sc://spark-server:15002")')
    print("\nExample shown.\n")


def example_with_namespace():
    """
    Configure with Custom Namespace (KEP-107 lines 69-85)

    Deploy Spark sessions to a specific Kubernetes namespace.
    """
    print("=" * 70)
    print("CUSTOM NAMESPACE CONFIGURATION")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config("spark-jobs"))

    spark = client.connect(
        num_executors=5,
        resources_per_executor={"cpu": "4", "memory": "8Gi"},
        spark_conf={"spark.serializer": "org.apache.spark.serializer.KryoSerializer"},
    )

    df = spark.range(50)
    print("\nSession running in 'spark-jobs' namespace")
    print(f"Generated {df.count()} rows")

    spark.stop()
    print("\nCustom namespace example complete.\n")


def main():
    """Run all examples sequentially."""
    print("E2E: Starting spark_connect_simple.py", flush=True)
    print("\n")
    print("=" * 70)
    print("KUBEFLOW SPARKCLIENT - KEP-107 API EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating three levels of API usage:\n")

    # Run examples (E2E in-cluster runs only Level 1+2 to stay within timeout)
    try:
        example_level1_minimal()
        example_level2_simple()
        if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") != "1":
            example_level3_advanced()
            example_connect_existing()
            example_with_namespace()

        print("=" * 70)
        print("ALL EXAMPLES COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("  1. Spark Operator installed in your cluster")
        print("  2. Proper RBAC permissions configured")
        print("  3. kubectl configured to access the cluster")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
