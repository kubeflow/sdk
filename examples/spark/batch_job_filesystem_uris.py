#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors.
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

"""Submit Spark batch jobs using remote filesystem URIs.

This example covers KEP-107 remote ``file_source`` schemes such as ``s3a://``,
``gs://``, and ``hdfs://``. Set one or more of the following environment
variables to a cluster-accessible Spark application URI:

- ``SPARK_E2E_S3A_URI``
- ``SPARK_E2E_GS_URI``
- ``SPARK_E2E_HDFS_URI``

The Spark image / cluster must already be configured for the target filesystem
(for example Hadoop S3A settings for MinIO). Phase 1 ``submit_job`` does not yet
accept ``spark_conf`` for per-job filesystem credentials.
"""

from __future__ import annotations

import os

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import FileJob, SparkClient, SparkJobStatus

URI_ENV_VARS = (
    ("s3a", "SPARK_E2E_S3A_URI"),
    ("gs", "SPARK_E2E_GS_URI"),
    ("hdfs", "SPARK_E2E_HDFS_URI"),
)


def _backend_config(namespace_default: str = "default") -> KubernetesBackendConfig:
    return KubernetesBackendConfig(
        namespace=os.environ.get("SPARK_TEST_NAMESPACE", namespace_default)
    )


def _client() -> SparkClient:
    return SparkClient(backend_config=_backend_config())


def _configured_uris() -> list[tuple[str, str]]:
    uris: list[tuple[str, str]] = []
    for scheme, env_name in URI_ENV_VARS:
        value = os.environ.get(env_name, "").strip()
        if value:
            uris.append((scheme, value))
    return uris


def example_submit_filesystem_uri_jobs() -> None:
    """Submit and wait for jobs for each configured filesystem URI."""
    print("=" * 70)
    print("SUBMIT SPARK BATCH JOBS WITH FILESYSTEM URIS")
    print("=" * 70)

    configured = _configured_uris()
    if not configured:
        raise RuntimeError(
            "No filesystem URI configured. Set one or more of: "
            + ", ".join(env_name for _, env_name in URI_ENV_VARS)
        )

    client = _client()
    submitted: list[str] = []

    try:
        for scheme, file_source in configured:
            print(f"\nSubmitting {scheme} job: {file_source}")
            job_name = client.submit_job(
                job=FileJob(file_source=file_source),
                num_executors=1,
                resources_per_executor={
                    "cpu": "1",
                    "memory": "512Mi",
                },
            )
            submitted.append(job_name)
            print(f"Job submitted: {job_name}")

            job = client.wait_for_job_status(
                job_name,
                status={SparkJobStatus.COMPLETED},
                timeout=600,
            )
            print(f"Job completed: {job.name} status={job.status}")
    finally:
        for job_name in submitted:
            client.delete_job(job_name)
            print(f"Deleted job: {job_name}")

    print("\nFilesystem URI job submission complete.\n")


def main() -> None:
    print("E2E: Starting batch_job_filesystem_uris.py", flush=True)
    print()
    print("=" * 70)
    print("KUBEFLOW SPARKCLIENT - FILESYSTEM URI BATCH JOBS")
    print("=" * 70)

    try:
        example_submit_filesystem_uri_jobs()
        print("=" * 70)
        print("FILESYSTEM URI BATCH JOBS COMPLETE!")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
