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

import logging
import os
import sys
import time

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import FileJob, SparkClient, SparkJobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

namespace = os.environ.get("SPARK_TEST_NAMESPACE", "default")
s3_endpoint = os.environ.get("S3_ENDPOINT", "http://seaweedfs-s3.seaweedfs.svc.cluster.local:8333")
s3_bucket = os.environ.get("S3_BUCKET", "mlpipeline")

# Default to standard KFP MinIO/SeaweedFS credentials if not explicitly overridden
s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

file_uri = os.environ.get("SPARK_JOB_URI", f"s3a://{s3_bucket}/spark_job.py")

backend_config = KubernetesBackendConfig(namespace=namespace)
client = SparkClient(backend_config=backend_config)

logger.info("SparkClient initialized for namespace: %s", namespace)

s3_spark_conf = {
    # Ivy cache location
    "spark.jars.ivy": "/tmp/.ivy2",
    "spark.driver.extraJavaOptions": "-Divy.home=/tmp/.ivy2",
    "spark.executor.extraJavaOptions": "-Divy.home=/tmp/.ivy2",
    # Hadoop S3A runtime packages
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.4.1,software.amazon.awssdk:s3:2.24.6,software.amazon.awssdk:s3-transfer-manager:2.24.6",
    "spark.jars.excludes": "software.amazon.awssdk:bundle",
    # S3A connector settings
    "spark.hadoop.fs.s3a.endpoint": s3_endpoint,
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "spark.hadoop.fs.s3a.connection.timeout": "10000",
    "spark.hadoop.fs.s3a.endpoint.region": "us-east-1",
}

if s3_access_key and s3_secret_key:
    s3_spark_conf["spark.hadoop.fs.s3a.access.key"] = s3_access_key
    s3_spark_conf["spark.hadoop.fs.s3a.secret.key"] = s3_secret_key
    s3_spark_conf["spark.hadoop.fs.s3a.aws.credentials.provider"] = (
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
    )
else:
    s3_spark_conf["spark.hadoop.fs.s3a.aws.credentials.provider"] = (
        "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
    )

logger.info("Submitting batch job using URI: %s", file_uri)
start_time = time.time()

job_name = client.submit_job(
    job=FileJob(
        file_source=file_uri,
        args=["10"],
    ),
    num_executors=1,
    resources_per_executor={
        "cpu": "1",
        "memory": "512Mi",
    },
    spark_conf=s3_spark_conf,
)

logger.info("Job submitted successfully. Generated Name: %s", job_name)
logger.info("Waiting for %s to complete execution...", job_name)

try:
    job = client.wait_for_job_status(
        job_name,
        status={SparkJobStatus.COMPLETED},
        timeout=600,
    )
except Exception as exc:
    duration = round(time.time() - start_time, 2)
    logger.error("Execution halted after %ss: %s", duration, exc)
    sys.exit(1)

duration = round(time.time() - start_time, 2)

if job is None or job.status != SparkJobStatus.COMPLETED:
    current_status = job.status if job else "None"
    logger.error("Failed after %ss! Job status: %s", duration, current_status)
    sys.exit(1)

logger.info("Job completed successfully in %ss. Status: %s", duration, job.status)
logger.info("Driver Pod: %s", job.driver_pod_name)

logger.info("Fetching execution logs for %s:", job_name)
print("-" * 60)
for line in client.get_job_logs(job_name):
    print(line.rstrip())
print("-" * 60)

logger.info("Cleaning up job %s...", job_name)
client.delete_job(job_name)
logger.info("Test finished successfully in %ss.", duration)
