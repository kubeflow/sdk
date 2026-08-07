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
import uuid

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import (
    Annotations,
    FileJob,
    Labels,
    Name,
    NodeSelector,
    SparkClient,
    SparkJobStatus,
    Toleration,
)

REMOTE_JOB = "https://raw.githubusercontent.com/kubeflow/sdk/main/examples/spark/spark_job.py"

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
            file_source=REMOTE_JOB,
            args=["10"],
        ),
        options=[
            Name(f"batch-job-options-{uuid.uuid4().hex[:8]}"),
            Labels(
                {
                    "app": "spark",
                    "team": "ml",
                }
            ),
            Annotations(
                {
                    "owner": "kubeflow",
                    "environment": "dev",
                }
            ),
            NodeSelector(
                {
                    "kubernetes.io/os": "linux",
                }
            ),
            Toleration(
                key="dedicated",
                operator="Equal",
                value="spark",
                effect="NoSchedule",
            ),
        ],
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


def example_verify_options():
    """Verify that Spark options were applied to the SparkApplication."""
    print("=" * 70)
    print("VERIFY SPARK BATCH JOB OPTIONS")
    print("=" * 70)

    if JOB_NAME is None:
        raise RuntimeError("No job has been submitted.")

    client = _client()

    response = client.backend.custom_api.get_namespaced_custom_object(
        group="sparkoperator.k8s.io",
        version="v1beta2",
        namespace=client.backend.namespace,
        plural="sparkapplications",
        name=JOB_NAME,
    )

    metadata = response["metadata"]
    spec = response["spec"]

    resource_name = metadata.get("name")

    if resource_name != JOB_NAME:
        raise RuntimeError(f"Expected resource name '{JOB_NAME}', got '{resource_name}'.")

    labels = metadata.get("labels", {})
    annotations = metadata.get("annotations", {})

    if labels.get("app") != "spark":
        raise RuntimeError("Expected label app=spark.")

    if labels.get("team") != "ml":
        raise RuntimeError("Expected label team=ml.")

    if annotations.get("owner") != "kubeflow":
        raise RuntimeError("Expected annotation owner=kubeflow.")

    if annotations.get("environment") != "dev":
        raise RuntimeError("Expected annotation environment=dev.")

    driver = spec.get("driver", {})

    node_selector = driver.get("nodeSelector", {})

    if node_selector.get("kubernetes.io/os") != "linux":
        raise RuntimeError("Expected driver nodeSelector to be applied.")

    tolerations = driver.get("tolerations", [])

    if not any(
        t.get("key") == "dedicated"
        and t.get("operator") == "Equal"
        and t.get("value") == "spark"
        and t.get("effect") == "NoSchedule"
        for t in tolerations
    ):
        raise RuntimeError("Expected driver toleration to be applied.")

    executor = spec.get("executor", {})

    executor_node_selector = executor.get("nodeSelector", {})

    if executor_node_selector.get("kubernetes.io/os") != "linux":
        raise RuntimeError("Expected executor nodeSelector to be applied.")

    executor_tolerations = executor.get("tolerations", [])

    if not any(
        t.get("key") == "dedicated"
        and t.get("operator") == "Equal"
        and t.get("value") == "spark"
        and t.get("effect") == "NoSchedule"
        for t in executor_tolerations
    ):
        raise RuntimeError("Expected executor toleration to be applied.")

    print("✓ Name verified.")
    print("✓ Labels verified.")
    print("✓ Annotations verified.")
    print("✓ Driver NodeSelector verified.")
    print("✓ Driver Toleration verified.")
    print("✓ Executor NodeSelector verified.")
    print("✓ Executor Toleration verified.")

    print("\nOptions verification complete.\n")


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
    """Run the batch job options examples."""
    print("E2E: Starting batch_job_options.py", flush=True)
    print()
    print("=" * 70)
    print("KUBEFLOW SPARKCLIENT - BATCH JOB OPTIONS")
    print("=" * 70)

    try:
        example_submit_and_wait()
        example_verify_options()
        example_delete_job()

        print("=" * 70)
        print("BATCH JOB OPTIONS COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
