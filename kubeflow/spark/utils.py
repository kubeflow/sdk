"""Utility functions for Spark client."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def format_memory(memory_mb: int) -> str:
    """Format memory in MB to Kubernetes format.

    Args:
        memory_mb: Memory in megabytes

    Returns:
        Formatted memory string (e.g., "4096m", "4g")

    Example:
        >>> format_memory(1024)
        '1g'
        >>> format_memory(512)
        '512m'
    """
    if memory_mb >= 1024 and memory_mb % 1024 == 0:
        return f"{memory_mb // 1024}g"
    return f"{memory_mb}m"


def parse_memory(memory_str: str) -> int:
    """Parse Kubernetes memory format to MB.

    Args:
        memory_str: Memory string (e.g., "4g", "512m")

    Returns:
        Memory in megabytes

    Example:
        >>> parse_memory("4g")
        4096
        >>> parse_memory("512m")
        512
    """
    memory_str = memory_str.lower().strip()

    if memory_str.endswith("g"):
        return int(memory_str[:-1]) * 1024
    elif memory_str.endswith("m"):
        return int(memory_str[:-1])
    elif memory_str.endswith("k"):
        return int(memory_str[:-1]) // 1024
    else:
        # Assume bytes
        return int(memory_str) // (1024 * 1024)


def validate_spark_config(config: dict[str, Any]) -> bool:
    """Validate Spark configuration.

    Args:
        config: Spark configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["app_name", "main_application_file"]

    for field in required_fields:
        if field not in config or not config[field]:
            raise ValueError(f"Required field '{field}' is missing or empty")

    # Validate resource specifications
    if "driver_memory" in config:
        try:
            parse_memory(config["driver_memory"])
        except Exception as e:
            raise ValueError(f"Invalid driver_memory format: {e}") from e

    if "executor_memory" in config:
        try:
            parse_memory(config["executor_memory"])
        except Exception as e:
            raise ValueError(f"Invalid executor_memory format: {e}") from e

    return True


def build_s3_path(bucket: str, prefix: str, filename: str) -> str:
    """Build S3 path for artifacts.

    Args:
        bucket: S3 bucket name
        prefix: Prefix/folder path
        filename: File name

    Returns:
        Complete S3 path

    Example:
        >>> build_s3_path("my-bucket", "artifacts/spark", "app.py")
        's3://my-bucket/artifacts/spark/app.py'
    """
    prefix = prefix.strip("/")
    if prefix:
        return f"s3://{bucket}/{prefix}/{filename}"
    return f"s3://{bucket}/{filename}"


def wait_for_completion(
    client: "BatchSparkClient",
    submission_id: str,
    timeout: int = 3600,
    poll_interval: int = 10,
) -> "ApplicationStatus":
    """Wait for Spark application to complete.

    Args:
        client: BatchSparkClient instance
        submission_id: Submission ID to monitor
        timeout: Maximum time to wait in seconds
        poll_interval: Polling interval in seconds

    Returns:
        Final ApplicationStatus

    Raises:
        TimeoutError: If application doesn't complete within timeout
    """
    import time

    from kubeflow.spark.models import ApplicationState

    start_time = time.time()

    while True:
        status = client.get_status(submission_id)

        if status.state in [
            ApplicationState.COMPLETED,
            ApplicationState.FAILED,
        ]:
            return status

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise TimeoutError(f"Application {submission_id} did not complete within {timeout}s")

        logger.info(
            f"Application {submission_id} status: {status.state.value}. Waiting {poll_interval}s..."
        )
        time.sleep(poll_interval)
