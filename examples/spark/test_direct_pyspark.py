#!/usr/bin/env python3
"""
Minimal PySpark Connect test - bypasses Kubeflow SDK completely
"""

import sys

print("Testing direct PySpark Connect (no Kubeflow SDK)...")
print("=" * 80)

try:
    import signal

    from pyspark.sql import SparkSession

    def timeout_handler(signum, frame):
        raise TimeoutError("Connection timed out")

    # Set a 15 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)

    print("Creating SparkSession.builder.remote('sc://localhost:30000')...")
    print("Timeout set to 15 seconds...")

    spark = (
        SparkSession.builder.remote("sc://localhost:30000")
        .appName("direct-test")
        .config("spark.connect.grpc.binding.port", "30000")
        .getOrCreate()
    )

    signal.alarm(0)  # Cancel timeout

    print("✓ Session created!")
    print(f"Session: {spark}")

    # Try a query
    print("\nTesting query...")
    df = spark.sql("SELECT 1 AS id, 'Hello' AS msg")
    result = df.collect()
    print(f"✓ Result: {result}")

    spark.stop()
    print("✓ Test passed!")

except TimeoutError:
    print("\n✗ Connection timed out after 15 seconds")
    print("\nThis means PySpark is not able to connect to the server.")
    print("The problem is NOT with Kubeflow SDK - it's with the basic connection.")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
