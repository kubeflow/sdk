#!/usr/bin/env python3
"""
Test that mimics exactly what the Kubeflow SDK does
"""

print("Testing SDK URL building logic...")

# Test 1: Build URL like SDK does
connect_url = "sc://localhost:30000"
use_ssl = False

url = connect_url
param_dict = {}

# This is what SDK does
if use_ssl:
    param_dict["use_ssl"] = "true"
else:
    param_dict["use_ssl"] = "false"  # New fix

# Build final URL
if param_dict:
    param_str = ";".join([f"{k}={v}" for k, v in param_dict.items()])
    final_url = f"{url}/;{param_str}"
else:
    final_url = url

print(f"SDK would build URL: {final_url}")

# Test 2: Try this URL with PySpark
print("\nTesting this URL with PySpark...")

import signal

from pyspark.sql import SparkSession


def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(15)

try:
    print(f"Connecting to: {final_url}")
    spark = SparkSession.builder.remote(final_url).appName("sdk-mimic-test").getOrCreate()
    signal.alarm(0)

    print("✓ Connection successful!")

    # Test query
    df = spark.sql("SELECT 1 AS id")
    print(f"✓ Query worked: {df.collect()}")

    spark.stop()
    print("✓ All good!")

except TimeoutError:
    print("✗ Timed out - the URL format might be wrong")
    print("\nTry this instead:")
    print(f"  SparkSession.builder.remote('{connect_url}').appName('test').getOrCreate()")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
