#!/usr/bin/env python3
# Copyright The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import FileJob, SparkClient, SparkJobStatus

REMOTE_PI = "https://raw.githubusercontent.com/apache/spark/master/examples/src/main/python/pi.py"

JOB_NAME: str | None = None


def _backend_config(namespace_default: str = "default"):
    """Backend config; uses SPARK_TEST_NAMESPACE in CI."""
    return KubernetesBackendConfig(
        namespace=os.environ.get("SPARK_TEST_NAMESPACE", namespace_default)
    )


def _client() -> SparkClient:
    """Create SparkClient."""
    return SparkClient(backend_config=_backend_config())


def example_submit_and_wait():
    """Submit a Spark batch job expected to fail."""
    global JOB_NAME

    print("=" * 70)
    print("SUBMIT FAILED SPARK BATCH JOB")
    print("=" * 70)

    client = _client()

    print("\nSubmitting Spark job expected to fail...")

    JOB_NAME = client.submit_job(
        job=FileJob(
            file_source=REMOTE_PI,
            args=["hello"],
        ),
        num_executors=1,
        resources_per_executor={
            "cpu": "1",
            "memory": "512Mi",
        },
    )

    print(f"Job submitted successfully: {JOB_NAME}")

    print("\nWaiting for job to fail...")

    job = client.wait_for_job_status(
        JOB_NAME,
        status={SparkJobStatus.FAILED},
        timeout=300,
    )

    print("Job failed as expected.")
    print(f"Status: {job.status}")
    print(f"Driver Pod: {job.driver_pod_name}")
    print(f"Namespace: {job.namespace}")

    print("\nSubmit and wait example complete.\n")


def example_get_job():
    """Get information about the failed Spark batch job."""
    print("=" * 70)
    print("GET FAILED SPARK BATCH JOB")
    print("=" * 70)

    if JOB_NAME is None:
        raise RuntimeError("No job has been submitted.")

    client = _client()

    print(f"\nRetrieving job: {JOB_NAME}")

    job = client.get_job(JOB_NAME)

    print("Job retrieved successfully.")
    print(f"Name: {job.name}")
    print(f"Namespace: {job.namespace}")
    print(f"Status: {job.status}")
    print(f"Driver Pod: {job.driver_pod_name}")
    print(f"Executors: {job.num_executors}")

    if job.status != SparkJobStatus.FAILED:
        raise RuntimeError("Expected job to be in FAILED state.")

    print("\nGet job example complete.\n")


def example_get_job_logs():
    """Get logs from the failed Spark batch job."""
    print("=" * 70)
    print("GET FAILED SPARK BATCH JOB LOGS")
    print("=" * 70)

    if JOB_NAME is None:
        raise RuntimeError("No job has been submitted.")

    client = _client()

    print(f"\nRetrieving logs for: {JOB_NAME}")

    print("\nDriver logs (first 20 lines):")
    print("-" * 70)

    line_count = 0
    for line in client.get_job_logs(JOB_NAME):
        print(line.rstrip())
        line_count += 1

        if line_count >= 20:
            print("...")
            break

    print("-" * 70)
    print(f"Displayed {line_count} log lines.")

    print("\nGet job logs example complete.\n")


def example_delete_job():
    """Delete the failed Spark batch job."""
    print("=" * 70)
    print("DELETE FAILED SPARK BATCH JOB")
    print("=" * 70)

    global JOB_NAME

    if JOB_NAME is None:
        raise RuntimeError("No job has been submitted.")

    client = _client()

    print(f"\nDeleting job: {JOB_NAME}")

    client.delete_job(JOB_NAME)

    print("Job deleted.")

    try:
        client.get_job(JOB_NAME)
    except RuntimeError as e:
        if "Spark job not found" not in str(e):
            raise
    else:
        raise RuntimeError("Job still exists after deletion.")

    print("Verified job has been deleted.")

    JOB_NAME = None

    print("\nDelete job example complete.\n")


def main():
    """Run failed batch job examples."""
    print("E2E: Starting batch_failed_job.py", flush=True)
    print()
    print("=" * 70)
    print("KUBEFLOW SPARKCLIENT - FAILED BATCH JOB")
    print("=" * 70)

    try:
        example_submit_and_wait()
        example_get_job()
        example_get_job_logs()
        example_delete_job()

        print("=" * 70)
        print("FAILED BATCH JOB EXAMPLE COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
