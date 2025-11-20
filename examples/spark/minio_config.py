"""MinIO S3 Configuration Helper for Spark Examples.

This module provides utilities for configuring Spark to work with MinIO
(S3-compatible storage) running in the same Kubernetes cluster.

Usage:
    from minio_config import get_s3_spark_conf, S3_ENDPOINT

    spark_conf = get_s3_spark_conf()
    response = client.submit_application(
        app_name="my-app",
        main_application_file="s3a://spark-scripts/my_script.py",
        spark_conf=spark_conf,
        ...
    )
"""

import os

# MinIO Configuration (deployed via setup_minio.sh)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-service.default.svc.cluster.local:9000")

# S3 endpoint for Spark (use http:// for internal cluster access)
S3_ENDPOINT = f"http://{MINIO_ENDPOINT}"

# Buckets
SCRIPTS_BUCKET = "spark-scripts"
DATA_BUCKET = "spark-data"
OUTPUT_BUCKET = "spark-output"


def get_s3_spark_conf(additional_conf=None, enable_history=False):
    """Get Spark configuration for S3/MinIO access.

    Args:
        additional_conf: Optional dict of additional Spark configs
        enable_history: If True, enable event logging for Spark History Server

    Returns:
        Dict of Spark configuration properties for S3 access
    """
    conf = {
        # Required for Spark 4.0
        "spark.kubernetes.file.upload.path": "/tmp",
        # Download Hadoop AWS libraries at runtime (includes S3A filesystem)
        # Compatible with Spark 4.0.0 and Hadoop 3.4.0
        "spark.jars.packages": (
            "org.apache.hadoop:hadoop-aws:3.4.0,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        ),
        # Ivy cache location - use /tmp which is always writable
        # Fixes: java.io.FileNotFoundException: /home/spark/.ivy2.5.2/cache/...
        "spark.jars.ivy": "/tmp/.ivy2",
        # S3A Configuration for MinIO
        "spark.hadoop.fs.s3a.endpoint": S3_ENDPOINT,
        "spark.hadoop.fs.s3a.access.key": MINIO_ACCESS_KEY,
        "spark.hadoop.fs.s3a.secret.key": MINIO_SECRET_KEY,
        "spark.hadoop.fs.s3a.path.style.access": "true",  # Required for MinIO
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",  # HTTP for internal
        # Performance tuning
        "spark.hadoop.fs.s3a.fast.upload": "true",
        "spark.hadoop.fs.s3a.block.size": "128M",
        "spark.hadoop.fs.s3a.multipart.size": "104857600",  # 100MB
        # Connection settings
        "spark.hadoop.fs.s3a.connection.maximum": "100",
        "spark.hadoop.fs.s3a.threads.max": "20",
        "spark.hadoop.fs.s3a.connection.timeout": "200000",
        "spark.hadoop.fs.s3a.attempts.maximum": "3",
    }

    # Add event logging for History Server
    if enable_history:
        conf.update(
            {
                "spark.eventLog.enabled": "true",
                "spark.eventLog.dir": "file:///mnt/spark-events",
                "spark.eventLog.compress": "true",
            }
        )

    # Merge additional configuration
    if additional_conf:
        conf.update(additional_conf)

    return conf


def get_s3_path(bucket, key):
    """Build S3 path for MinIO.

    Args:
        bucket: Bucket name (e.g., 'spark-scripts')
        key: Object key (e.g., 'exploration.py')

    Returns:
        S3 URL (e.g., 's3a://spark-scripts/exploration.py')
    """
    return f"s3a://{bucket}/{key}"


# Common S3 paths for examples
S3_PATHS = {
    "exploration_script": get_s3_path(SCRIPTS_BUCKET, "exploration.py"),
    "csv_analysis_script": get_s3_path(SCRIPTS_BUCKET, "csv_analysis.py"),
    "etl_script": get_s3_path(SCRIPTS_BUCKET, "etl_pipeline.py"),
    "batch_job_script": get_s3_path(SCRIPTS_BUCKET, "batch_job.py"),
    "data_dir": f"s3a://{DATA_BUCKET}/",
    "output_dir": f"s3a://{OUTPUT_BUCKET}/",
}


def print_minio_info():
    """Print MinIO configuration information."""
    print("MinIO S3 Configuration:")
    print(f"  Endpoint: {S3_ENDPOINT}")
    print(f"  Access Key: {MINIO_ACCESS_KEY}")
    print("  Buckets:")
    print(f"    - {SCRIPTS_BUCKET}/ - Application scripts")
    print(f"    - {DATA_BUCKET}/ - Input data")
    print(f"    - {OUTPUT_BUCKET}/ - Output results")
    print()


if __name__ == "__main__":
    # Print configuration when run directly
    print_minio_info()
    print("Available S3 Paths:")
    for name, path in S3_PATHS.items():
        print(f"  {name}: {path}")
