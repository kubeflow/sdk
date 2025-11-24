#!/usr/bin/env python3
"""
Title: Dynamic Allocation and Auto-scaling
Level: 2 (Intermediate - Auto-scaling)
Target Audience: Data Engineers optimizing resource usage
Time to Run: ~4-5 minutes

Description:
This example demonstrates Spark's dynamic allocation feature, which automatically
scales executors up and down based on workload. You'll learn when to use dynamic
allocation, how to configure it, and how it improves resource efficiency in
multi-tenant Kubernetes clusters.

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Default namespace with 'spark-operator-spark' service account
- Spark 3.0+ (required for dynamic allocation on Kubernetes)

What You'll Learn:
- Dynamic allocation configuration and tuning
- How Spark scales executors automatically
- Resource efficiency vs performance trade-offs
- Monitoring executor scaling behavior
- When to use dynamic vs fixed allocation

Real-World Use Case:
Multi-tenant clusters, variable workloads, cost optimization, shared resources.
"""

import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from kubeflow.spark import (  # noqa: E402
    ApplicationState,
    DynamicAllocation,
    OperatorBackendConfig,
    BatchSparkClient,
)


def create_dynamic_allocation_script():
    """Create a PySpark script demonstrating dynamic allocation.

    Returns:
        str: Python code for dynamic allocation demo
    """
    return """
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, sum as _sum, count, avg,
    monotonically_increasing_id, rand, when
)
from pyspark.sql.types import *
import time

# Create Spark session
spark = SparkSession.builder \\
    .appName("Dynamic Allocation Demo") \\
    .getOrCreate()

# Get dynamic allocation configuration
dyn_enabled = spark.conf.get("spark.dynamicAllocation.enabled", "false")
min_executors = spark.conf.get("spark.dynamicAllocation.minExecutors", "N/A")
max_executors = spark.conf.get("spark.dynamicAllocation.maxExecutors", "N/A")
initial_executors = spark.conf.get("spark.dynamicAllocation.initialExecutors", "N/A")

print("\\n" + "="*80)
print("DYNAMIC ALLOCATION DEMO")
print("="*80)
print("\\n📊 Dynamic Allocation Configuration:")
print(f"  - Enabled: {dyn_enabled}")
print(f"  - Min Executors: {min_executors}")
print(f"  - Max Executors: {max_executors}")
print(f"  - Initial Executors: {initial_executors}")
shuffle_tracking = spark.conf.get('spark.dynamicAllocation.shuffleTracking.enabled', 'N/A')
print(f"  - Shuffle Tracking: {shuffle_tracking}")

# ============================================================================
# PHASE 1: LIGHT WORKLOAD (should use minimal executors)
# ============================================================================
print("\\n\\n[PHASE 1] Light Workload - Testing Scale Down")
print("="*80)
print("Expected: Spark should use minimal executors for small dataset\\n")

# Small dataset
print("Creating small dataset (1,000 records)...")
small_data = spark.range(1000).select(
    col("id"),
    (col("id") * 2).alias("value"),
    (col("id") % 10).alias("category")
)

print(f"  Created {small_data.count()} records")

# Simple aggregation (low resource need)
result1 = small_data.groupBy("category").agg(
    count("id").alias("count"),
    _sum("value").alias("sum_value"),
    avg("value").alias("avg_value")
).orderBy("category")

print("\\nLight Workload Results:")
result1.show()

print("\\n⏱️  Waiting 10 seconds for executor scaling to stabilize...")
time.sleep(10)

# Check current executor count (approximation based on task distribution)
print("\\n📈 After light workload:")
print("  - Spark should have scaled down to minimum executors")
print("  - Check operator logs or Spark UI for exact executor count")

# ============================================================================
# PHASE 2: MEDIUM WORKLOAD (should scale up moderately)
# ============================================================================
print("\\n\\n[PHASE 2] Medium Workload - Testing Scale Up")
print("="*80)
print("Expected: Spark should add executors to handle increased load\\n")

# Medium dataset
print("Creating medium dataset (100,000 records)...")
medium_data = spark.range(100000).select(
    col("id"),
    (rand() * 1000).alias("value"),
    (col("id") % 100).alias("category"),
    (col("id") % 10).alias("partition_key")
)

print(f"  Created {medium_data.count()} records")

# More complex processing (triggers parallelism)
result2 = medium_data.groupBy("category").agg(
    count("id").alias("count"),
    _sum("value").alias("sum_value"),
    avg("value").alias("avg_value")
).filter(col("count") > 100).orderBy(col("sum_value").desc())

print(f"\\nMedium Workload Results (showing top 10):")
result2.show(10)

print("\\n⏱️  Waiting 10 seconds for executor scaling...")
time.sleep(10)

print("\\n📈 After medium workload:")
print("  - Spark should have scaled up executors")
print("  - More executors = better parallelism for aggregations")

# ============================================================================
# PHASE 3: HEAVY WORKLOAD (should scale to maximum)
# ============================================================================
print("\\n\\n[PHASE 3] Heavy Workload - Testing Maximum Scale")
print("="*80)
print("Expected: Spark should scale to max executors for heavy computation\\n")

# Large dataset with shuffle
print("Creating large dataset (500,000 records)...")
large_data = spark.range(500000).select(
    col("id"),
    (rand() * 10000).alias("value"),
    (col("id") % 1000).alias("category"),
    when(rand() > 0.5, "A").otherwise("B").alias("group")
)

print(f"  Created {large_data.count()} records")

# Heavy processing with shuffle (join + aggregation)
print("\\nPerforming heavy computation (join + aggregation)...")

# Self-join to increase workload
large_data_alias = large_data.alias("df1")
large_data2 = large_data.alias("df2")

result3 = large_data_alias.join(
    large_data2,
    col("df1.category") == col("df2.category"),
    "inner"
).groupBy("df1.category").agg(
    count("df1.id").alias("total_records"),
    _sum("df1.value").alias("sum_value1"),
    _sum("df2.value").alias("sum_value2")
).orderBy(col("total_records").desc())

print(f"\\nHeavy Workload Results (top 10 categories):")
result3.show(10)

print("\\n⏱️  Waiting 10 seconds for executor scaling...")
time.sleep(10)

print("\\n📈 After heavy workload:")
print("  - Spark should have scaled to maximum executors")
print("  - Shuffle operations triggered executor requests")
print("  - Join and aggregation required maximum parallelism")

# ============================================================================
# PHASE 4: COOL DOWN (should scale back down)
# ============================================================================
print("\\n\\n[PHASE 4] Cool Down - Testing Scale Down After Load")
print("="*80)
print("Expected: After workload completes, Spark should release idle executors\\n")

print("Performing final light operation...")
final_result = small_data.groupBy("category").count().orderBy("category")
final_result.show()

print("\\n⏱️  Waiting 15 seconds for idle executors to be released...")
time.sleep(15)

print("\\n📉 After cool down:")
print("  - Spark should release idle executors")
print("  - Only minimum executors retained")
print("  - Resources returned to cluster for other workloads")

# ============================================================================
# SUMMARY
# ============================================================================
print("\\n\\n" + "="*80)
print("DYNAMIC ALLOCATION DEMO COMPLETED!")
print("="*80)

print("\\n🎯 Key Observations:")
print("  1. Light workload - Minimal executors (resource efficient)")
print("  2. Medium workload - Moderate scale up (balanced)")
print("  3. Heavy workload - Maximum executors (performance optimized)")
print("  4. Cool down - Scale down (return resources)")

print("\\n💡 Dynamic Allocation Benefits:")
print("  Automatic resource optimization")
print("  Cost efficiency in multi-tenant clusters")
print("  No manual executor tuning needed")
print("  Better cluster utilization")

print("\\nWARNING: When NOT to Use Dynamic Allocation:")
print("  - Streaming jobs (need consistent executors)")
print("  - Very short-lived jobs (overhead of scaling)")
print("  - Dedicated clusters (fixed allocation is simpler)")
print("  - Jobs with strict latency SLAs")

print("\\n📊 Configuration Parameters Explained:")
print("  - minExecutors: Safety net, always available")
print("  - maxExecutors: Resource cap, prevents runaway scaling")
print("  - initialExecutors: Starting point, balances startup time")
print("  - shuffleTracking: Required for K8s, tracks shuffle data")

print("\\n🔧 Tuning Recommendations:")
print("  - Set min = 1-2 for cost efficiency")
print("  - Set max based on cluster capacity")
print("  - Set initial = expected average load")
print("  - Enable shuffleTracking (required for K8s)")
print("  - Monitor executor metrics in Spark UI")

spark.stop()
"""


def main():
    """Main example: Submit Spark job with dynamic allocation enabled."""

    print("=" * 80)
    print("EXAMPLE 06: Dynamic Allocation and Auto-scaling")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Configuring dynamic allocation")
    print("  2. Automatic executor scaling based on workload")
    print("  3. Resource efficiency in shared clusters")
    print("  4. Performance vs cost trade-offs")
    print("  5. When to use dynamic vs fixed allocation")
    print()

    # Step 1: Create SparkClient with configuration
    print("Step 1: Creating Spark client...")
    config = OperatorBackendConfig(
        namespace=os.getenv("SPARK_NAMESPACE", "default"),
        service_account="spark-operator-spark",
        default_spark_image="docker.io/library/spark",
        context=os.getenv("KUBE_CONTEXT", "kind-spark-test"),
        enable_monitoring=False,
        enable_ui=False,
    )
    client = BatchSparkClient(backend_config=config)
    print("  Client created successfully")
    print()

    # Step 2: Configure dynamic allocation
    app_name = "dynamic-allocation-demo"

    print("Step 2: Configuring dynamic allocation...")

    # Create dynamic allocation configuration
    dyn_alloc = DynamicAllocation(
        enabled=True,
        initial_executors=1,  # Start with 1 executor
        min_executors=1,  # Keep at least 1
        max_executors=5,  # Scale up to 5 max
        shuffle_tracking_enabled=True,  # Required for K8s
    )

    print(f"  App name: {app_name}")
    print("  Spark version: 4.0.0 (supports dynamic allocation)")
    print("  Dynamic Allocation Settings:")
    print(f"    - Initial executors: {dyn_alloc.initial_executors}")
    print(f"    - Min executors: {dyn_alloc.min_executors}")
    print(f"    - Max executors: {dyn_alloc.max_executors}")
    print(f"    - Shuffle tracking: {dyn_alloc.shuffle_tracking_enabled}")
    print()
    print("  How it works:")
    print("    - Starts with 1 executor (initial)")
    print("    - Scales up to 5 as workload increases")
    print("    - Scales down to 1 when idle")
    print()

    # Step 3: Submit the application
    print("Step 3: Submitting application with dynamic allocation...")

    try:
        response = client.submit_application(
            # Application metadata
            app_name=app_name,
            # Placeholder
            main_application_file=("local:///opt/spark/examples/src/main/python/pi.py"),
            # Spark configuration
            # Spark 3.0+ required for dynamic allocation on K8s
            spark_version="4.0.0",
            app_type="Python",
            # Resource allocation per executor
            driver_cores=1,
            driver_memory="1g",
            executor_cores=1,
            executor_memory="1g",
            # This will be overridden by dynamic allocation
            num_executors=1,
            # Dynamic Allocation Configuration
            dynamic_allocation=dyn_alloc,
            # Spark configuration
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
                # Additional tuning for dynamic allocation
                # Release idle executors after 30s
                "spark.dynamicAllocation.executorIdleTimeout": "30s",
                # Keep cached executors longer
                "spark.dynamicAllocation.cachedExecutorIdleTimeout": "60s",
                # Request executors quickly
                "spark.dynamicAllocation.schedulerBacklogTimeout": "5s",
            },
            # Labels for tracking
            labels={
                "feature": "dynamic-allocation",
                "workload": "variable",
            },
        )

        print("  Application submitted successfully!")
        print(f"  Submission ID: {response.submission_id}")
        print(f"  Status: {response.status}")
        print()
        print("  Dynamic allocation features enabled:")
        print("    Auto-scaling based on workload")
        print("    Shuffle tracking for K8s compatibility")
        print("    Optimized resource utilization")
        print()

    except Exception as e:
        print(f"  ERROR: Submission failed: {e}")
        sys.exit(1)

    # Step 4: Monitor the application
    print("Step 4: Monitoring application (this will take 4-5 minutes)...")
    print("  The job will demonstrate executor scaling through 4 phases:")
    print("    Phase 1: Light workload (scale down)")
    print("    Phase 2: Medium workload (scale up)")
    print("    Phase 3: Heavy workload (max scale)")
    print("    Phase 4: Cool down (scale down)")
    print()

    try:
        # Wait for completion with longer timeout for demo phases
        final_status = client.wait_for_job_status(
            submission_id=app_name,
            timeout=360,  # 6 minutes for all phases
            polling_interval=5,  # Check every 5 seconds
        )

        print("  Application completed!")
        print(f"  Final state: {final_status.state.value}")
        print()

        # Check if successful
        if final_status.state != ApplicationState.COMPLETED:
            print(
                f"  WARNING: Application did not complete successfully: {final_status.state.value}"
            )
            print("  Check logs below for details.")

    except TimeoutError:
        print("  ERROR: Application did not complete within 6 minutes")
        print(f"  You can check status later with: client.get_job('{app_name}')")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Error monitoring application: {e}")
        sys.exit(1)

    # Step 5: Retrieve results from logs
    print("Step 5: Retrieving dynamic allocation insights from logs...")
    print()

    try:
        logs = list(client.get_job_logs(app_name))

        print("=" * 80)
        print("DYNAMIC ALLOCATION RESULTS")
        print("=" * 80)

        # Display important sections
        important_keywords = [
            "DYNAMIC ALLOCATION",
            "Configuration:",
            "[PHASE",
            "Expected:",
            "After",
            "Key Observations",
            "Benefits:",
            "When NOT to Use",
            "Tuning Recommendations",
        ]

        for line in logs:
            if any(keyword in line for keyword in important_keywords) or any(
                emoji in line
                for emoji in [
                    "Done",
                    "WARNING",
                    "📊",
                    "📈",
                    "📉",
                    "💡",
                    "🎯",
                    "🔧",
                ]
            ):
                print(line)

        print()
        print("=" * 80)

    except Exception as e:
        print(f"  WARNING: Could not retrieve logs: {e}")
        print("  The job may have completed but logs are not yet available")

    # Step 6: Cleanup
    print()
    print("Step 6: Cleaning up resources...")
    try:
        client.delete_job(app_name)
        print(f"  Application '{app_name}' deleted")
        print("  All executors released back to cluster")
    except Exception as e:
        print(f"  WARNING: Cleanup warning: {e}")
        print(f"  You can manually delete with: kubectl delete sparkapplication {app_name}")

    print()
    print("=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("What you learned:")
    print("  How to configure dynamic allocation")
    print("  How Spark scales executors automatically")
    print("  Resource efficiency vs performance trade-offs")
    print("  Tuning parameters and their effects")
    print("  When to use dynamic vs fixed allocation")
    print()
    print("Dynamic Allocation Configuration:")
    print("  from kubeflow.spark import DynamicAllocation")
    print()
    print("  dyn_alloc = DynamicAllocation(")
    print("      enabled=True,")
    print("      initial_executors=2,  # Starting point")
    print("      min_executors=1,      # Always keep at least 1")
    print("      max_executors=10,     # Cap at 10")
    print("      shuffle_tracking_enabled=True  # Required for K8s")
    print("  )")
    print()
    print("  client.submit_application(")
    print("      app_name='my-app',")
    print("      dynamic_allocation=dyn_alloc,")
    print("      ...")
    print("  )")
    print()
    print("Key Scaling Triggers:")
    print("  - Scale Up: Pending tasks, shuffle writes, backlog")
    print("  - Scale Down: Idle executors, no shuffle data needed")
    print("  - Timing: Controlled by timeout configurations")
    print()
    print("Use Cases for Dynamic Allocation:")
    print("  Multi-tenant clusters (shared resources)")
    print("  Variable workloads (unpredictable load)")
    print("  Cost optimization (pay for what you use)")
    print("  Development/testing (efficient resource use)")
    print()
    print("Use Cases for Fixed Allocation:")
    print("  Streaming jobs (predictable, constant load)")
    print("  Short-lived jobs (scaling overhead too high)")
    print("  Strict SLAs (no scaling latency)")
    print("  Dedicated clusters (resources already allocated)")
    print()
    print("Next steps:")
    print("  - Experiment with different min/max settings")
    print("  - Monitor executor scaling in Spark UI")
    print("  - Compare costs: dynamic vs fixed allocation")
    print("  - Test with your own workloads")
    print()


if __name__ == "__main__":
    main()
