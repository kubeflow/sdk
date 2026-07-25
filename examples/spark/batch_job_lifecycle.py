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
    """Submit a Spark batch job and wait for completion."""
    global JOB_NAME

    print("=" * 70)
    print("SUBMIT SPARK BATCH JOB")
    print("=" * 70)

    client = _client()

    print("\nSubmitting Spark job...")

    JOB_NAME = client.submit_job(
        job=FileJob(
            file_source=REMOTE_PI,
            args=["10"],
        ),
        num_executors=1,
        resources_per_executor={
            "cpu": "1",
            "memory": "512Mi",
        },
    )

    print(f"Job submitted successfully: {JOB_NAME}")

    print("\nWaiting for job to complete...")

    job = client.wait_for_job_status(
        JOB_NAME,
        status={SparkJobStatus.COMPLETED},
        timeout=300,
    )

    print("Job completed successfully.")
    print(f"Status: {job.status}")
    print(f"Driver Pod: {job.driver_pod_name}")
    print(f"Namespace: {job.namespace}")
    print("\nSubmit and wait example complete.\n")


def example_get_job():
    """Get information about a Spark batch job."""
    print("=" * 70)
    print("GET SPARK BATCH JOB")
    print("=" * 70)

    if JOB_NAME is None:
        raise RuntimeError("No job has been submitted.")

    client = _client()

    print(f"\nRetrieving job: {JOB_NAME}")

    job = client.get_job(JOB_NAME)

    if job.status != SparkJobStatus.COMPLETED:
        raise RuntimeError(f"Expected COMPLETED status, got {job.status}.")

    if not job.driver_pod_name:
        raise RuntimeError("Expected driver pod name to be populated.")

    print("Job retrieved successfully.")
    print(f"Name: {job.name}")
    print(f"Namespace: {job.namespace}")
    print(f"Status: {job.status}")
    print(f"Driver Pod: {job.driver_pod_name}")
    print(f"Executors: {job.num_executors}")

    print("\nGet job example complete.\n")


def example_list_jobs():
    """List Spark batch jobs."""
    print("=" * 70)
    print("LIST SPARK BATCH JOBS")
    print("=" * 70)

    if JOB_NAME is None:
        raise RuntimeError("No job has been submitted.")

    client = _client()

    print("\nListing Spark jobs...")

    jobs = client.list_jobs()

    print(f"Found {len(jobs)} Spark job(s).\n")

    job_found = False

    for job in jobs:
        print(f"- {job.name} | Status: {job.status} | Namespace: {job.namespace}")

        if job.name == JOB_NAME:
            job_found = True

    if not job_found:
        raise RuntimeError(f"Submitted job '{JOB_NAME}' not found in job list.")

    print("\nSubmitted job found in job list.")

    print("\nListing completed Spark jobs...")

    completed_jobs = client.list_jobs(
        status={SparkJobStatus.COMPLETED},
    )

    print(f"Found {len(completed_jobs)} completed Spark job(s).\n")

    completed_job_found = False

    for job in completed_jobs:
        if job.status != SparkJobStatus.COMPLETED:
            raise RuntimeError(f"Expected COMPLETED status, got {job.status}.")

        print(f"- {job.name} | Status: {job.status} | Namespace: {job.namespace}")

        if job.name == JOB_NAME:
            completed_job_found = True

    if not completed_job_found:
        raise RuntimeError(f"Submitted completed job '{JOB_NAME}' not found in filtered job list.")

    print("\nCompleted job filter verified.")
    print("\nList jobs example complete.\n")


def example_get_job_logs():
    """Get logs from a Spark batch job."""
    print("=" * 70)
    print("GET SPARK BATCH JOB LOGS")
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
    """Delete a Spark batch job."""
    print("=" * 70)
    print("DELETE SPARK BATCH JOB")
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
    """Run all batch job lifecycle examples."""
    print("E2E: Starting batch_job_lifecycle.py", flush=True)
    print()
    print("=" * 70)
    print("KUBEFLOW SPARKCLIENT - BATCH JOB LIFECYCLE")
    print("=" * 70)

    try:
        example_submit_and_wait()
        example_get_job()
        example_list_jobs()
        example_get_job_logs()
        example_delete_job()

        print("=" * 70)
        print("BATCH JOB LIFECYCLE COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
