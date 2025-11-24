"""Example utilities for Kubeflow Spark SDK examples.

This module provides common utilities, helpers, and sample data generators
used across the Spark examples. It helps reduce code duplication and provides
a consistent interface for common operations.

Usage:
    from example_utils import (
        create_client,
        setup_logging,
        generate_sample_data,
    )

    # Create client with defaults
    client = create_client()

    # Or with custom configuration
    client = create_client(
        namespace="my-namespace",
        enable_ui=True,
    )
"""

from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Optional

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from kubeflow.spark import (  # noqa: E402
    ApplicationState,
    DynamicAllocation,
    OperatorBackendConfig,
    RestartPolicy,
    RestartPolicyType,
    BatchBatchSparkClient,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging for examples.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# CLIENT CREATION HELPERS
# ============================================================================


def create_client(
    namespace: Optional[str] = None,
    service_account: str = "spark-operator-spark",
    context: Optional[str] = None,
    enable_monitoring: bool = False,
    enable_ui: bool = False,
    default_spark_image: str = "docker.io/library/spark",
) -> BatchSparkClient:
    """Create a BatchSparkClient with sensible defaults for examples.

    Args:
        namespace: Kubernetes namespace (default: from SPARK_NAMESPACE env or 'default')
        service_account: Kubernetes service account
        context: Kubernetes context (default: from KUBE_CONTEXT env or 'kind-spark-test')
        enable_monitoring: Enable Prometheus monitoring
        enable_ui: Enable Spark UI
        default_spark_image: Default Spark image to use

    Returns:
        Configured BatchSparkClient instance

    Example:
        >>> client = create_client()
        >>> client = create_client(namespace="production", enable_ui=True)
    """
    config = OperatorBackendConfig(
        namespace=namespace or os.getenv("SPARK_NAMESPACE", "default"),
        service_account=service_account,
        default_spark_image=default_spark_image,
        context=context or os.getenv("KUBE_CONTEXT", "kind-spark-test"),
        enable_monitoring=enable_monitoring,
        enable_ui=enable_ui,
    )

    logger.info(f"Creating BatchSparkClient for namespace: {config.namespace}")
    return BatchSparkClient(backend_config=config)


# ============================================================================
# COMMON CONFIGURATIONS
# ============================================================================


def get_resilient_restart_policy() -> RestartPolicy:
    """Get a restart policy suitable for production batch jobs.

    Returns:
        RestartPolicy with retry configuration
    """
    return RestartPolicy(
        type=RestartPolicyType.ON_FAILURE,
        on_failure_retries=3,
        on_failure_retry_interval=30,
        on_submission_failure_retries=2,
        on_submission_failure_retry_interval=15,
    )


def get_dynamic_allocation_config(
    min_executors: int = 1,
    max_executors: int = 10,
    initial_executors: int = 2,
) -> DynamicAllocation:
    """Get a dynamic allocation configuration.

    Args:
        min_executors: Minimum number of executors
        max_executors: Maximum number of executors
        initial_executors: Initial number of executors

    Returns:
        DynamicAllocation configuration
    """
    return DynamicAllocation(
        enabled=True,
        initial_executors=initial_executors,
        min_executors=min_executors,
        max_executors=max_executors,
        shuffle_tracking_enabled=True,  # Required for K8s
    )


def get_spark_conf_defaults(spark_version: str = "4.0.0") -> dict[str, str]:
    """Get default Spark configuration suitable for examples.

    Args:
        spark_version: Spark version to configure for

    Returns:
        Dictionary of Spark configuration properties
    """
    conf = {
        "spark.kubernetes.file.upload.path": "/tmp",
    }

    # Spark 4.0+ specific configurations
    if spark_version.startswith("4."):
        conf.update(
            {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
            }
        )

    return conf


# ============================================================================
# SAMPLE DATA GENERATORS
# ============================================================================


def generate_customer_data(num_records: int = 100) -> list[tuple]:
    """Generate sample customer data.

    Args:
        num_records: Number of customer records to generate

    Returns:
        List of customer tuples (id, name, email, city, signup_date)
    """
    from datetime import date
    import random

    cities = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
    ]

    base_date = date.today() - timedelta(days=365)

    customers = []
    for i in range(1, num_records + 1):
        signup_date = base_date + timedelta(days=random.randint(0, 365))
        customers.append(
            (
                i,
                f"Customer{i}",
                f"customer{i}@example.com",
                random.choice(cities),
                signup_date.strftime("%Y-%m-%d"),
            )
        )

    return customers


def generate_transaction_data(
    num_transactions: int = 1000,
    num_customers: int = 100,
    days_back: int = 30,
) -> list[tuple]:
    """Generate sample transaction data.

    Args:
        num_transactions: Number of transactions to generate
        num_customers: Number of unique customers
        days_back: How many days back to generate data

    Returns:
        List of transaction tuples (tx_id, date, customer_id, amount, status)
    """
    import random

    base_date = datetime.now()
    statuses = ["completed", "pending", "cancelled"]

    transactions = []
    for i in range(1, num_transactions + 1):
        tx_date = (base_date - timedelta(days=random.randint(0, days_back))).strftime("%Y-%m-%d")
        customer_id = random.randint(1, num_customers)
        amount = round(random.uniform(10.0, 1000.0), 2)
        # 90% completed
        status = random.choice(statuses) if i % 10 != 0 else "completed"

        transactions.append((i, tx_date, customer_id, amount, status))

    return transactions


def generate_sales_data(
    num_records: int = 100,
    products: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
) -> list[tuple]:
    """Generate sample sales data.

    Args:
        num_records: Number of sales records to generate
        products: List of product names (default: common products)
        categories: List of categories (default: Electronics, Furniture, etc.)

    Returns:
        List of sales tuples (id, date, product, category, quantity, price, region)
    """
    import random

    if products is None:
        products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Desk", "Chair"]

    if categories is None:
        categories = ["Electronics", "Furniture", "Accessories"]

    regions = ["North", "South", "East", "West"]
    base_date = datetime.now()

    sales = []
    for i in range(1, num_records + 1):
        sale_date = (base_date - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
        product = random.choice(products)
        category = random.choice(categories)
        quantity = random.randint(1, 10)
        price = round(random.uniform(25.0, 1500.0), 2)
        region = random.choice(regions)

        sales.append((i, sale_date, product, category, quantity, price, region))

    return sales


# ============================================================================
# COMMON OPERATIONS
# ============================================================================


def wait_for_job(
    client: BatchBatchSparkClient,
    app_name: str,
    timeout: int = 300,
    polling_interval: int = 5,
) -> ApplicationState:
    """Wait for a Spark job to complete with proper error handling.

    Args:
        client: BatchSparkClient instance
        app_name: Application name
        timeout: Maximum time to wait in seconds
        polling_interval: Polling interval in seconds

    Returns:
        Final ApplicationState

    Raises:
        TimeoutError: If job doesn't complete within timeout
        RuntimeError: If job fails
    """
    logger.info(f"Waiting for job '{app_name}' to complete (timeout: {timeout}s)...")

    try:
        status = client.wait_for_job_status(
            submission_id=app_name,
            timeout=timeout,
            polling_interval=polling_interval,
        )

        if status.state == ApplicationState.COMPLETED:
            logger.info(f"Job '{app_name}' completed successfully")
        elif status.state == ApplicationState.FAILED:
            logger.error(f"Job '{app_name}' failed")
            raise RuntimeError(f"Job failed with state: {status.state.value}")
        else:
            logger.warning(f"Job '{app_name}' ended with unexpected state: {status.state.value}")

        return status.state

    except TimeoutError:
        logger.error(f"Job '{app_name}' timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"Error waiting for job '{app_name}': {e}")
        raise


def print_job_status(client: BatchBatchSparkClient, app_name: str):
    """Print current job status in a formatted way.

    Args:
        client: BatchSparkClient instance
        app_name: Application name
    """
    try:
        status = client.get_job(app_name)

        print("\n" + "=" * 60)
        print(f"JOB STATUS: {app_name}")
        print("=" * 60)
        print(f"State: {status.state.value}")
        if status.app_id:
            print(f"App ID: {status.app_id}")
        if status.submission_time:
            print(f"Submitted: {status.submission_time}")
        if status.start_time:
            print(f"Started: {status.start_time}")
        if status.completion_time:
            print(f"Completed: {status.completion_time}")
        print("=" * 60)
        print()

    except Exception as e:
        logger.error(f"Error getting status for '{app_name}': {e}")


def cleanup_job(client: BatchBatchSparkClient, app_name: str):
    """Clean up a Spark application with proper error handling.

    Args:
        client: BatchSparkClient instance
        app_name: Application name
    """
    try:
        client.delete_job(app_name)
        logger.info(f"Successfully deleted application '{app_name}'")
    except Exception as e:
        logger.warning(f"Failed to delete application '{app_name}': {e}")
        logger.warning(f"You can manually delete with: kubectl delete sparkapplication {app_name}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    if seconds > 0 or not parts:
        parts.append(f"{int(seconds)}s")

    return " ".join(parts)


def get_sample_spark_conf_for_use_case(use_case: str) -> dict[str, str]:
    """Get recommended Spark configuration for common use cases.

    Args:
        use_case: One of 'etl', 'ml', 'streaming', 'interactive'

    Returns:
        Dictionary of recommended Spark configuration
    """
    base_conf = get_spark_conf_defaults()

    use_case_configs = {
        "etl": {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.shuffle.partitions": "200",
        },
        "ml": {
            # Some ML libs prefer this off
            "spark.sql.adaptive.enabled": "false",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.kryoserializer.buffer.max": "512m",
        },
        "streaming": {
            "spark.streaming.backpressure.enabled": "true",
            "spark.streaming.receiver.maxRate": "10000",
        },
        "interactive": {
            "spark.sql.adaptive.enabled": "true",
            "spark.ui.enabled": "true",
            "spark.eventLog.enabled": "false",
        },
    }

    if use_case in use_case_configs:
        base_conf.update(use_case_configs[use_case])
    else:
        logger.warning(f"Unknown use case '{use_case}', using defaults")

    return base_conf


# ============================================================================
# EXAMPLE METADATA
# ============================================================================

EXAMPLES_METADATA = {
    "01_hello_spark_pi": {
        "title": "Hello Spark - Calculate Pi",
        "level": 1,
        "category": "Getting Started",
        "time": "2-3 minutes",
        "description": "Your first Spark job - calculate Pi using Monte Carlo method",
    },
    "02_csv_data_analysis": {
        "title": "CSV Data Analysis",
        "level": 1,
        "category": "Data Analysis Basics",
        "time": "2-3 minutes",
        "description": "Analyze CSV data with filtering and aggregations",
    },
    "03_interactive_dataframe_exploration": {
        "title": "Interactive DataFrame Exploration",
        "level": 1,
        "category": "Data Exploration",
        "time": "3-4 minutes",
        "description": "Exploratory data analysis patterns and data quality checks",
    },
    "04_etl_pipeline_simple": {
        "title": "Simple ETL Pipeline",
        "level": 2,
        "category": "Data Engineering",
        "time": "3-4 minutes",
        "description": "Extract-Transform-Load pipeline with data validation",
    },
    "05_scheduled_batch_job": {
        "title": "Scheduled Batch Job",
        "level": 2,
        "category": "Batch Processing",
        "time": "3-4 minutes",
        "description": "Production batch job with incremental processing and resilience",
    },
    "06_autoscaling_dynamic_allocation": {
        "title": "Dynamic Allocation",
        "level": 2,
        "category": "Auto-scaling",
        "time": "4-5 minutes",
        "description": "Automatic executor scaling based on workload",
    },
}


def print_examples_catalog():
    """Print a catalog of all available examples."""
    print("\n" + "=" * 80)
    print("KUBEFLOW SPARK SDK - EXAMPLES CATALOG")
    print("=" * 80)
    print()

    # Group by level
    by_level = {}
    for name, metadata in EXAMPLES_METADATA.items():
        level = metadata["level"]
        if level not in by_level:
            by_level[level] = []
        by_level[level].append((name, metadata))

    level_names = {
        1: "Level 1: Getting Started",
        2: "Level 2: Data Engineering Basics",
    }

    for level in sorted(by_level.keys()):
        print(f"\n{level_names.get(level, f'Level {level}')}")
        print("-" * 80)

        for name, metadata in sorted(by_level[level], key=lambda x: x[0]):
            print(f"\n{name}.py")
            print(f"  {metadata['title']}")
            print(f"  Category: {metadata['category']}")
            print(f"  Time: {metadata['time']}")
            print(f"  {metadata['description']}")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    # Print examples catalog when run directly
    print_examples_catalog()
