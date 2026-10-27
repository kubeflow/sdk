import os
import sys
import time

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import FileJob, SparkClient, SparkJobStatus

namespace = os.environ.get("SPARK_TEST_NAMESPACE", "default")
s3_endpoint = os.environ.get("S3_ENDPOINT", "http://seaweedfs-s3.seaweedfs.svc.cluster.local:8333")

s3_bucket = os.environ.get("S3_BUCKET", "mlpipeline")
s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

backend_config = KubernetesBackendConfig(namespace=namespace)
client = SparkClient(backend_config=backend_config)

print(f"[1/5] SparkClient initialized for namespace: {namespace}")

REMOTE_S3_JOB = f"s3a://{s3_bucket}/spark_job.py"

s3_spark_conf = {
    # Ivy cache location
    "spark.jars.ivy": "/tmp/.ivy2",
    "spark.driver.extraJavaOptions": "-Divy.home=/tmp/.ivy2",
    "spark.executor.extraJavaOptions": "-Divy.home=/tmp/.ivy2",
    # Trimmed package downloads
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.4.1,software.amazon.awssdk:s3:2.24.6,software.amazon.awssdk:s3-transfer-manager:2.24.6",
    "spark.jars.excludes": "software.amazon.awssdk:bundle",
    # S3A / SeaweedFS parameters
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

print(f"[2/5] Submitting remote S3A batch job from: {REMOTE_S3_JOB}...")
start_time = time.time()

job_name = client.submit_job(
    job=FileJob(
        file_source=REMOTE_S3_JOB,
        args=["10"],
    ),
    num_executors=1,
    resources_per_executor={
        "cpu": "1",
        "memory": "512Mi",
    },
    spark_conf=s3_spark_conf,
)

print(f"Job submitted successfully! Generated Name: {job_name}")

print(f"[3/5] Waiting for {job_name} to finish execution (timeout: 600s)...")
try:
    job = client.wait_for_job_status(
        job_name,
        status={SparkJobStatus.COMPLETED},
        timeout=600,
    )
except Exception as e:
    duration = round(time.time() - start_time, 2)
    print(f"Execution halted or timed out after {duration}s: {e}")
    sys.exit(1)

duration = round(time.time() - start_time, 2)

if job is None or job.status != SparkJobStatus.COMPLETED:
    current_status = job.status if job else "None"
    print(f"Failed after {duration}s! Job status: {current_status}")
    sys.exit(1)

print(f"Success! Job Status: {job.status} (Total Duration: {duration}s)")
print(f"Driver Pod Name: {job.driver_pod_name}")

print(f"[4/5] Fetching execution logs for {job_name}...")
print("=" * 60)
for line in client.get_job_logs(job_name):
    print(line.rstrip())
print("=" * 60)

print(f"[5/5] Cleaning up job: {job_name}...")
client.delete_job(job_name)
print(f"E2E Test Passed Successfully in {duration} seconds!")
