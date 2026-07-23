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
from kubeflow.spark import FuncJob, SparkClient, SparkJobStatus


def estimate_pi(samples: int) -> None:
    """Simple function executed as a Spark FuncJob."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("estimate-pi").getOrCreate()
    count = spark.range(samples).count()
    print(f"Estimated Pi over {count} samples.")
    spark.stop()


def main():
    """Run all batch job lifecycle examples."""
    print("E2E: Starting batch_func_job_lifecycle.py", flush=True)
    print()
    print("=" * 70)
    print("KUBEFLOW SPARKCLIENT - FUNCJOB LIFECYCLE")
    print("=" * 70)

    try:
        client = SparkClient(
            backend_config=KubernetesBackendConfig(
                namespace=os.environ.get(
                    "SPARK_TEST_NAMESPACE",
                    "default",
                )
            )
        )
        print("=" * 70)
        print("SUBMIT SPARK BATCH JOB")
        print("=" * 70)

        print("\nSubmitting Spark job...")

        job_name = client.submit_job(
            job=FuncJob(
                func=estimate_pi,
                func_args={
                    "samples": 10,
                },
            ),
            num_executors=1,
            resources_per_executor={
                "cpu": "1",
                "memory": "512Mi",
            },
        )

        print(f"Job submitted successfully: {job_name}")

        print("\nWaiting for job to complete...")

        job = client.wait_for_job_status(
            job_name,
            timeout=300,
        )

        print("Job completed successfully.")
        print(f"Status: {job.status}")
        print(f"Driver Pod: {job.driver_pod_name}")
        print(f"Namespace: {job.namespace}")
        print("\nSubmit and wait example complete.\n")

        print("=" * 70)
        print("GET SPARK BATCH JOB")
        print("=" * 70)

        print(f"\nRetrieving job: {job_name}")

        job = client.get_job(job_name)

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

        print("=" * 70)
        print("LIST SPARK BATCH JOBS")
        print("=" * 70)

        print("\nListing Spark jobs...")

        jobs = client.list_jobs()

        print(f"Found {len(jobs)} Spark job(s).\n")

        job_found = False

        for job in jobs:
            print(f"- {job.name} | Status: {job.status} | Namespace: {job.namespace}")

            if job.name == job_name:
                job_found = True

        if not job_found:
            raise RuntimeError(f"Submitted job '{job_name}' not found in job list.")

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

            if job.name == job_name:
                completed_job_found = True

        if not completed_job_found:
            raise RuntimeError(
                f"Submitted completed job '{job_name}' not found in filtered job list."
            )

        print("\nCompleted job filter verified.")
        print("\nList jobs example complete.\n")

        print("=" * 70)
        print("GET SPARK BATCH JOB LOGS")
        print("=" * 70)

        print(f"\nRetrieving logs for: {job_name}")

        print("\nDriver logs (first 20 lines):")
        print("-" * 70)

        line_count = 0
        for line in client.get_job_logs(job_name):
            print(line.rstrip())
            line_count += 1

            if line_count >= 20:
                print("...")
                break

        print("-" * 70)
        print(f"Displayed {line_count} log lines.")

        print("\nGet job logs example complete.\n")

        print("=" * 70)
        print("DELETE SPARK BATCH JOB")
        print("=" * 70)

        print(f"\nDeleting job: {job_name}")

        client.delete_job(job_name)

        print("Job deleted.")

        try:
            client.get_job(job_name)
        except RuntimeError as e:
            if "Spark job not found" not in str(e):
                raise
        else:
            raise RuntimeError("Job still exists after deletion.")

        print("Verified job has been deleted.")

        print("\nDelete job example complete.\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
