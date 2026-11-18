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

"""Kubeflow Pipelines (KFP) example integrating SparkClient.

This example demonstrates orchestrating distributed Spark batch jobs inside a
Kubeflow Pipeline using the Kubeflow Spark SDK.

The pipeline comprises two sequential tasks:
1. `submit_spark_job`: Connects to the Kubernetes cluster using `SparkClient`,
   submits a batch Spark application (via FileJob), and polls until completion.
2. `inspect_and_cleanup_spark_job`: Retrieves logs from the driver pod for auditing
   and observability, then deletes the completed Spark job resources.
"""

import argparse

from kubeflow.pipelines import compiler, dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["kubeflow[spark]"],
)
def submit_spark_job(
    file_source: str,
    namespace: str = "default",
    num_executors: int = 1,
    executor_cpu: str = "1",
    executor_memory: str = "512Mi",
    timeout: int = 600,
) -> str:
    """Submit a Spark batch job and wait for completion.

    Args:
        file_source: Remote or local URI of the Spark application script.
        namespace: Target Kubernetes namespace for the Spark application.
        num_executors: Number of executor instances to allocate.
        executor_cpu: CPU request/limit per executor pod.
        executor_memory: Memory request/limit per executor pod.
        timeout: Maximum time in seconds to wait for completion.

    Returns:
        The name of the submitted Spark job.
    """
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.spark import FileJob, SparkClient, SparkJobStatus

    backend_config = KubernetesBackendConfig(namespace=namespace)
    client = SparkClient(backend_config=backend_config)

    job_name = client.submit_job(
        job=FileJob(file_source=file_source),
        num_executors=num_executors,
        resources_per_executor={
            "cpu": executor_cpu,
            "memory": executor_memory,
        },
    )
    print(f"Submitted Spark batch job: {job_name}")

    completed_job = client.wait_for_job_status(
        name=job_name,
        status={SparkJobStatus.COMPLETED},
        timeout=timeout,
    )
    print(f"Spark job {job_name} finished with status: {completed_job.status}")
    return job_name


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["kubeflow[spark]"],
)
def inspect_and_cleanup_spark_job(
    job_name: str,
    namespace: str = "default",
) -> str:
    """Retrieve driver pod logs and delete the Spark batch job.

    Args:
        job_name: Name of the Spark job to inspect and delete.
        namespace: Target Kubernetes namespace.

    Returns:
        Log retrieval summary string.
    """
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.spark import SparkClient

    backend_config = KubernetesBackendConfig(namespace=namespace)
    client = SparkClient(backend_config=backend_config)

    logs = list(client.get_job_logs(name=job_name))
    log_summary = f"Retrieved {len(logs)} log lines from driver pod for job {job_name}."
    print(log_summary)

    client.delete_job(name=job_name)
    print(f"Cleaned up Spark batch job: {job_name}")
    return log_summary


@dsl.pipeline(
    name="spark-batch-pipeline",
    description="Kubeflow Pipeline orchestrating a distributed Spark batch job via SparkClient.",
)
def spark_pipeline(
    file_source: str = "https://raw.githubusercontent.com/kubeflow/sdk/main/examples/spark/spark_job.py",
    namespace: str = "default",
    num_executors: int = 1,
):
    """Pipeline definition composing Spark submission and cleanup tasks."""
    spark_task = submit_spark_job(
        file_source=file_source,
        namespace=namespace,
        num_executors=num_executors,
    )

    cleanup_task = inspect_and_cleanup_spark_job(
        job_name=spark_task.output,
        namespace=namespace,
    )
    cleanup_task.after(spark_task)


def compile_pipeline(output_path: str = "spark_batch_pipeline.yaml") -> str:
    """Compile the pipeline definition into a KFP IR YAML package.

    Args:
        output_path: Destination path for the compiled YAML file.

    Returns:
        Path to the compiled YAML package.
    """
    compiler.Compiler().compile(
        pipeline_func=spark_pipeline,
        package_path=output_path,
    )
    return output_path


def main() -> None:
    """CLI entrypoint for compiling or running the Spark pipeline."""
    parser = argparse.ArgumentParser(description="Kubeflow Spark Pipeline Example")
    parser.add_argument(
        "--output",
        "-o",
        default="spark_batch_pipeline.yaml",
        help="Target output YAML path for compiled pipeline IR",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Submit the compiled pipeline directly via PipelinesClient",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace for the pipeline run",
    )
    args = parser.parse_args()

    compiled_file = compile_pipeline(args.output)
    print(f"Pipeline compiled successfully to: {compiled_file}")

    if args.run:
        from kubeflow.pipelines import PipelinesClient

        client = PipelinesClient()
        run = client.run(
            pipeline=spark_pipeline,
            name="spark-pipeline-run",
            params={"namespace": args.namespace},
        )
        print(f"Pipeline run submitted: {run.id}")


if __name__ == "__main__":
    main()
