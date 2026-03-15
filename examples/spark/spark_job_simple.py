#!/usr/bin/env python3
"""
SparkClient Batch Job Examples

Demonstrates submitting batch Spark jobs via the SparkApplication CRD.
The SDK auto-builds a Docker image from the local script and loads it
into the Kind cluster before submitting.

Usage:
    python examples/spark/spark_job_simple.py

    # Run a specific example:
    SPARK_TEST_NAMESPACE=spark-test python examples/spark/spark_job_simple.py
"""

import os
from pathlib import Path
import uuid

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import SparkClient
from kubeflow.spark.types.types import SparkJobStatus

EXAMPLES_DIR = Path(__file__).parent
WORDCOUNT_SCRIPT = str(EXAMPLES_DIR / "wordcount.py")


def _e2e_job_name(base: str) -> str:
    """In E2E in-cluster runs use a unique name to avoid conflicts."""
    if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") == "1":
        return f"{base}-{uuid.uuid4().hex[:8]}"
    return base


def _backend_config(namespace_default: str = "spark-test"):
    """Backend config; uses SPARK_TEST_NAMESPACE in CI."""
    return KubernetesBackendConfig(
        namespace=os.environ.get("SPARK_TEST_NAMESPACE", namespace_default)
    )


def example_level1_minimal():
    """
    Level 1: Minimal Usage

    Submit a local Python script with all defaults:
    - Auto-generated job name
    - Default namespace (spark-test or SPARK_TEST_NAMESPACE)
    - SDK builds Docker image and loads it into Kind automatically
    """
    print("=" * 70)
    print("LEVEL 1: MINIMAL USAGE (Auto-generated name)")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())
    job_name = client.submit_job(main_file=WORDCOUNT_SCRIPT)
    print(f"\nSubmitted job: {job_name}")

    job = client.wait_for_job_status(job_name, status={SparkJobStatus.COMPLETED, SparkJobStatus.FAILED}, timeout=300)
    print(f"Job status: {job.status}")
    print(f"Driver pod: {job.driver_pod_name}")

    for line in client.get_job_logs(job_name):
        print(line)

    client.delete_job(job_name)
    print("\nLevel 1 complete.\n")


def example_level2_named_with_arguments():
    """
    Level 2: Named Job with Arguments

    Submit with a custom job name and pass command-line arguments
    to the Spark application. Lists all jobs before deleting.
    """
    print("=" * 70)
    print("LEVEL 2: NAMED JOB WITH ARGUMENTS")
    print("=" * 70)

    client = SparkClient(backend_config=_backend_config())
    job_name = _e2e_job_name("wordcount-args")

    job_name = client.submit_job(
        main_file=WORDCOUNT_SCRIPT,
        name=job_name,
        arguments=["--verbose"],
    )
    print(f"\nSubmitted job: {job_name}")

    job = client.wait_for_job_status(job_name)
    print(f"Job status: {job.status}")
    print(f"Driver pod: {job.driver_pod_name}")

    for line in client.get_job_logs(job_name):
        print(line)

    jobs = client.list_jobs()
    print(f"\nFound {len(jobs)} job(s) in namespace:")
    for j in jobs:
        print(f"  {j.name}: status={j.status} driver={j.driver_pod_name}")

    client.delete_job(job_name)
    print("\nLevel 2 complete.\n")


def main():
    """Run all examples sequentially."""
    print("E2E: Starting spark_job_simple.py", flush=True)
    print("\n")
    print("=" * 70)
    print("KUBEFLOW SPARKCLIENT - BATCH JOB EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating batch job submission:\n")

    try:
        example_level1_minimal()
        example_level2_named_with_arguments()

        print("=" * 70)
        print("ALL EXAMPLES COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("  1. Spark Operator installed and watching your namespace")
        print("  2. Docker daemon running (needed to build job images)")
        print("  3. Kind CLI installed (needed to load images into cluster)")
        print("  4. kubectl configured to access the cluster")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
