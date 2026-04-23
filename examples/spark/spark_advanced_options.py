#!/usr/bin/env python3
"""
Advanced SparkClient Options Examples (KEP-107 Options Pattern)

This example demonstrates the options pattern for advanced Kubernetes configurations.
The options pattern provides extensibility without API changes - new option types
can be added in the future without modifying the core SparkClient API.

KEP-107 Reference: Lines 180-192

Usage:
    python examples/spark_advanced_options.py
"""

import os
import uuid

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import (
    Annotations,
    Driver,
    Executor,
    # Options for advanced Kubernetes configuration
    Labels,
    Name,
    NodeSelector,
    PodTemplateOverride,
    SparkClient,
    Toleration,
)


def _e2e_session_name(base: str) -> str:
    """In E2E in-cluster runs use a unique name to avoid conflicts."""
    if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") == "1":
        return f"{base}-{uuid.uuid4().hex[:8]}"
    return base


def _backend_config():
    """Backend config; uses SPARK_TEST_NAMESPACE in CI."""
    return KubernetesBackendConfig(namespace=os.environ.get("SPARK_TEST_NAMESPACE", "default"))


def example_labels_and_annotations():
    """Example 1: Add labels and annotations for organization and tooling.

    Labels are used for selection and grouping in Kubernetes.
    Annotations store arbitrary metadata for tools and documentation.
    """
    print("=" * 70)
    print("EXAMPLE 1: Labels and Annotations")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())

    spark = client.connect(
        num_executors=3,
        resources_per_executor={"cpu": "2", "memory": "4Gi"},
        spark_conf={"spark.serializer": "org.apache.spark.serializer.KryoSerializer"},
        options=[
            Labels(
                {
                    "app": "spark",
                    "team": "data-engineering",
                    "environment": "production",
                    "cost-center": "analytics",
                }
            ),
            Annotations(
                {
                    "description": "Daily ETL pipeline for customer data",
                    "owner": "data-team@company.com",
                    "runbook": "https://wiki.company.com/runbooks/etl",
                    "pagerduty": "spark-oncall",
                }
            ),
        ],
    )

    print("\nSpark session created with labels and annotations.")
    print("  Labels help with resource organization and cost tracking")
    print("  Annotations provide context for monitoring and alerting tools")

    df = spark.range(100)
    print(f"\nProcessed {df.count()} rows")

    spark.stop()
    print("\nExample complete.\n")


def example_node_selection():
    """Example 2: Schedule Spark pods on specific node types.

    Node selectors constrain pods to run only on nodes with matching labels.
    Useful for dedicated Spark clusters or GPU nodes.
    """
    print("=" * 70)
    print("EXAMPLE 2: Node Selection")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())

    spark = client.connect(
        num_executors=5,
        resources_per_executor={
            "cpu": "4",
            "memory": "16Gi",
            "nvidia.com/gpu": "1",  # Request GPU
        },
        spark_conf={"spark.serializer": "org.apache.spark.serializer.KryoSerializer"},
        options=[
            NodeSelector(
                {
                    "node-type": "spark-gpu",  # Only schedule on GPU nodes
                    "workload": "ml",
                }
            ),
        ],
    )

    print("\nSpark session scheduled on GPU nodes.")
    print("  All pods will only run on nodes labeled with node-type=spark-gpu")

    df = spark.range(1000)
    print(f"\nProcessed {df.count()} rows on GPU-enabled executors")

    spark.stop()
    print("\nExample complete.\n")


def example_tolerations():
    """Example 3: Tolerate node taints for dedicated workloads.

    Tolerations allow pods to schedule on tainted nodes.
    Useful for dedicated Spark infrastructure or spot instances.
    """
    print("=" * 70)
    print("EXAMPLE 3: Node Taints and Tolerations")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())

    spark = client.connect(
        num_executors=10,
        resources_per_executor={"cpu": "8", "memory": "32Gi"},
        spark_conf={"spark.serializer": "org.apache.spark.serializer.KryoSerializer"},
        options=[
            # Tolerate nodes tainted for Spark workloads
            Toleration(
                key="spark-workload",
                operator="Equal",
                value="true",
                effect="NoSchedule",
            ),
            # Tolerate spot instance taints
            Toleration(
                key="spot-instance",
                operator="Exists",
                effect="NoSchedule",
            ),
        ],
    )

    print("\nSpark session can run on tainted nodes.")
    print("  Executors can use cost-effective spot instances")

    df = spark.range(10000)
    print(f"\nProcessed {df.count()} rows on spot instances")

    spark.stop()
    print("\nExample complete.\n")


def example_pod_template_override():
    """Example 4: Full control with pod template overrides.

    Pod template overrides provide complete control over pod specifications.
    Use for advanced cases like security contexts, volumes, or sidecars.

    Warning: Can conflict with SDK-managed settings. Use with caution.
    """
    print("=" * 70)
    print("EXAMPLE 4: Pod Template Override (Advanced)")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())

    spark = client.connect(
        driver=Driver(resources={"cpu": "2", "memory": "4Gi"}),
        executor=Executor(
            num_instances=5,
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
        ),
        spark_conf={"spark.serializer": "org.apache.spark.serializer.KryoSerializer"},
        options=[
            # Add security context to executors
            PodTemplateOverride(
                role="executor",
                template={
                    "spec": {
                        "securityContext": {
                            "runAsUser": 1000,
                            "runAsGroup": 1000,
                            "fsGroup": 1000,
                            "runAsNonRoot": True,
                        },
                        "containers": [
                            {
                                "name": "spark-executor",
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                            }
                        ],
                    }
                },
            ),
        ],
    )

    print("\nSpark executors running with restricted security context.")
    print("  Meets security requirements for sensitive data workloads")

    df = spark.range(5000)
    print(f"\nProcessed {df.count()} rows with enhanced security")

    spark.stop()
    print("\nExample complete.\n")


def example_name_option():
    """Example: Set session name via Name option.

    The Name option allows you to specify a custom name for your Spark session.
    This name becomes the Kubernetes resource name and can be used for management.
    If not specified, a name is auto-generated with format: spark-connect-{uuid}
    """
    print("=" * 70)
    print("EXAMPLE: Custom Session Name with Name Option")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())

    spark = client.connect(
        num_executors=3,
        resources_per_executor={"cpu": "2", "memory": "4Gi"},
        spark_conf={"spark.serializer": "org.apache.spark.serializer.KryoSerializer"},
        options=[
            Name(_e2e_session_name("custom-session-name")),
            Labels({"app": "spark", "team": "data-eng"}),
        ],
    )

    print("\nSpark session created with custom name.")
    print("  Session name: custom-session-name")

    df = spark.range(100)
    print(f"\nProcessed {df.count()} rows")

    spark.stop()
    print("\nExample complete.\n")


def example_combined_options():
    """Example 5: Combine multiple options for production workloads.

    Real-world production deployments often need multiple configurations.
    The options pattern makes complex setups composable and maintainable.
    """
    print("=" * 70)
    print("EXAMPLE 5: Combined Options (Production Pattern)")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config("spark-production"))

    spark = client.connect(
        driver=Driver(
            resources={"cpu": "4", "memory": "8Gi"},
            service_account="spark-driver-prod",
        ),
        executor=Executor(
            num_instances=20,
            resources_per_executor={"cpu": "8", "memory": "32Gi"},
        ),
        spark_conf={
            "spark.app.name": "prod-etl-pipeline",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.dynamicAllocation.enabled": "false",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        },
        options=[
            # Custom session name
            Name(_e2e_session_name("prod-etl-session")),
            # Organization and cost tracking
            Labels(
                {
                    "app": "etl-pipeline",
                    "team": "data-platform",
                    "environment": "production",
                    "cost-center": "analytics",
                    "version": "v2.1.0",
                }
            ),
            # Monitoring and alerting metadata
            Annotations(
                {
                    "description": "Production ETL pipeline for customer analytics",
                    "owner": "data-platform@company.com",
                    "slack-channel": "#data-platform-alerts",
                    "pagerduty-service": "spark-prod",
                    "runbook": "https://wiki.company.com/spark-etl",
                }
            ),
            # Schedule on dedicated Spark nodes
            NodeSelector(
                {
                    "node-type": "spark",
                    "workload": "etl",
                    "disk-type": "ssd",
                }
            ),
            # Tolerate spot instances for cost savings
            Toleration(
                key="spot-instance",
                operator="Exists",
                effect="NoSchedule",
            ),
        ],
    )

    print("\nProduction Spark cluster configured with:")
    print("  - Custom resource allocations (20 executors x 8 cores)")
    print("  - Labels for cost tracking and organization")
    print("  - Annotations for monitoring and alerting")
    print("  - Node selection for dedicated infrastructure")
    print("  - Spot instance tolerations for cost optimization")

    df = spark.range(100000)
    print(f"\nProcessing {df.count()} rows in production environment")

    spark.stop()
    print("\nExample complete.\n")


def main():
    """Run all option pattern examples."""
    print("E2E: Starting spark_advanced_options.py", flush=True)
    print("\n")
    print("=" * 70)
    print("SPARK OPTIONS PATTERN EXAMPLES (KEP-107)")
    print("=" * 70)
    print("\nDemonstrating advanced Kubernetes configuration options\n")

    try:
        example_labels_and_annotations()
        if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") != "1":
            example_node_selection()
            example_tolerations()
            example_pod_template_override()
            example_name_option()
            example_combined_options()

        print("=" * 70)
        print("ALL EXAMPLES COMPLETE!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  - Options pattern provides extensibility without API changes")
        print("  - New option types can be added in future SDK versions")
        print("  - Options are composable for complex production setups")
        print("  - Backwards compatible - old code continues to work")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("  1. Spark Operator installed with SparkConnect CRD")
        print("  2. Proper RBAC permissions configured")
        print("  3. kubectl configured to access the cluster")
        print("  4. Nodes labeled/tainted as referenced in examples")


if __name__ == "__main__":
    main()
