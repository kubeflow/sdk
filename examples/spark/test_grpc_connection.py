#!/usr/bin/env python3
"""
Minimal test to check if we can connect to Spark Connect via gRPC directly.
"""

import sys

print("Testing basic gRPC connection to Spark Connect...")
print("=" * 80)

# Test 1: Check if grpc is available
print("\n[1] Checking grpcio installation...")
try:
    import grpc

    print(f"✓ grpcio version: {grpc.__version__}")
except ImportError as e:
    print(f"✗ grpcio not installed: {e}")
    print("Install with: pip install grpcio")
    sys.exit(1)

# Test 2: Try to connect to the port
print("\n[2] Testing TCP connection to localhost:30000...")
import socket

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex(("localhost", 30000))
    sock.close()

    if result == 0:
        print("✓ TCP connection successful")
    else:
        print(f"✗ TCP connection failed with error code: {result}")
        sys.exit(1)
except Exception as e:
    print(f"✗ TCP connection error: {e}")
    sys.exit(1)

# Test 3: Try gRPC channel
print("\n[3] Creating gRPC channel to localhost:30000...")
try:
    channel = grpc.insecure_channel("localhost:30000")
    print("✓ gRPC channel created")

    # Test if channel is ready
    print("   Testing if channel becomes ready (5 second timeout)...")
    try:
        future = grpc.channel_ready_future(channel)
        future.result(timeout=5)
        print("✓ gRPC channel is ready!")
    except grpc.FutureTimeoutError:
        print("✗ gRPC channel timeout - server not responding on gRPC")
        print("   This suggests the server might not be accepting gRPC connections")
    except Exception as e:
        print(f"✗ gRPC channel error: {e}")
    finally:
        channel.close()

except Exception as e:
    print(f"✗ gRPC error: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Try with PySpark directly
print("\n[4] Testing PySpark Spark Connect...")
try:
    from pyspark.sql import SparkSession

    print("✓ PySpark imported")

    print("   Creating SparkSession with remote connection...")
    print("   This might take 10-30 seconds or hang if there's an issue...")

    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Session creation timed out after 20 seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)

    try:
        spark = (
            SparkSession.builder.remote("sc://localhost:30000").appName("grpc-test").getOrCreate()
        )

        signal.alarm(0)
        print("✓ SparkSession created!")

        # Try a simple operation
        print("   Testing simple query...")
        df = spark.sql("SELECT 1 AS id")
        result = df.collect()
        print(f"✓ Query executed successfully: {result}")

        spark.stop()
        print("✓ Session stopped")

        print("\n" + "=" * 80)
        print("SUCCESS! Everything is working.")
        print("=" * 80)

    except TimeoutError as e:
        signal.alarm(0)
        print(f"✗ {e}")
        print("\nThis means PySpark is hanging while trying to connect.")
        print("Possible causes:")
        print("  1. Spark Connect server not responding to gRPC requests")
        print("  2. Server bound to wrong address (IPv4 vs IPv6)")
        print("  3. Firewall or network policy blocking connection")

except KeyboardInterrupt:
    print("\n✗ Interrupted by user")
except Exception as e:
    print(f"✗ PySpark connection failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug test complete")
print("=" * 80)
