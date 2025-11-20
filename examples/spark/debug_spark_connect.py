#!/usr/bin/env python3
"""
Debug script for Spark Connect connection issues.

This script tests the connection step-by-step with verbose logging.
"""

import logging
import os
import signal
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add SDK to path
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, sdk_path)

print("=" * 80)
print("Spark Connect Connection Debugger")
print("=" * 80)

# Test 1: Check PySpark installation
print("\n[Test 1] Checking PySpark installation...")
try:
    import pyspark

    print(f"✓ PySpark version: {pyspark.__version__}")
except ImportError as e:
    print(f"✗ PySpark not installed: {e}")
    sys.exit(1)

# Test 2: Check Spark Connect support
print("\n[Test 2] Checking Spark Connect support...")
try:
    from pyspark.sql import SparkSession

    print("✓ SparkSession imported")

    # Check if remote() method exists
    if hasattr(SparkSession.builder, "remote"):
        print("✓ Spark Connect (remote) support available")
    else:
        print("✗ Spark Connect support not available - upgrade PySpark")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Test basic gRPC connectivity
print("\n[Test 3] Testing gRPC connectivity to localhost:30000...")
try:
    import grpc

    print("✓ grpc module available")

    # Try to create a channel
    channel = grpc.insecure_channel("localhost:30000")

    # Set a short timeout for connection test
    import grpc

    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        print("✓ gRPC channel ready")
    except grpc.FutureTimeoutError:
        print("✗ gRPC channel timeout - server may not be responding")
        print("  Check: kubectl logs -l app=spark-connect -n default")
    except Exception as e:
        print(f"✗ gRPC channel error: {e}")
    finally:
        channel.close()

except ImportError:
    print("⚠ grpcio not installed (will be used by pyspark)")
except Exception as e:
    print(f"⚠ gRPC test error: {e}")

# Test 4: Test Kubeflow SDK import
print("\n[Test 4] Testing Kubeflow SDK imports...")
try:
    from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

    print("✓ Kubeflow Spark imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 5: Create config (doesn't connect yet)
print("\n[Test 5] Creating ConnectBackendConfig...")
try:
    config = ConnectBackendConfig(
        connect_url="sc://localhost:30000",
        use_ssl=False,
        timeout=10,  # Short timeout for testing
    )
    print(f"✓ Config created: {config.connect_url}")
except Exception as e:
    print(f"✗ Config creation error: {e}")
    sys.exit(1)

# Test 6: Create client (doesn't connect yet)
print("\n[Test 6] Creating SparkSessionClient...")
try:
    client = SparkSessionClient(backend_config=config)
    print("✓ Client created")
except Exception as e:
    print(f"✗ Client creation error: {e}")
    sys.exit(1)

# Test 7: Try to create session with timeout
print("\n[Test 7] Creating Spark session (this may hang)...")
print("  If this hangs for more than 30 seconds, press Ctrl+C")
print("  Attempting connection to sc://localhost:30000...")


def timeout_handler(signum, frame):
    print("\n✗ Session creation timed out after 30 seconds")
    print("\nPossible issues:")
    print("  1. Spark Connect server not accessible")
    print("  2. Port forwarding not working correctly")
    print("  3. gRPC connection blocked")
    print("\nDebugging steps:")
    print("  - Check server logs: kubectl logs -l app=spark-connect -n default -f")
    print("  - Verify port forward: lsof -i :30000")
    print("  - Test connectivity: nc -zv localhost 30000")
    print(
        "  - Check server is listening: kubectl exec -it <pod-name> -- netstat -tlnp | grep 15002"
    )
    sys.exit(1)


# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

try:
    start_time = time.time()
    session = client.create_session(app_name="debug-test")
    elapsed = time.time() - start_time

    signal.alarm(0)  # Cancel timeout

    print(f"✓ Session created in {elapsed:.2f} seconds!")
    print(f"  Session ID: {session.session_id}")
    print(f"  App name: {session.app_name}")

    # Test 8: Try a simple query
    print("\n[Test 8] Testing simple SQL query...")
    try:
        df = session.sql("SELECT 1 AS id, 'test' AS message")
        result = df.collect()
        print(f"✓ Query executed: {result[0].message}")
        df.show()
    except Exception as e:
        print(f"✗ Query error: {e}")

    # Cleanup
    print("\n[Cleanup] Closing session...")
    session.close()
    client.close()
    print("✓ Session closed")

    print("\n" + "=" * 80)
    print("All tests passed! Connection is working.")
    print("=" * 80)

except KeyboardInterrupt:
    signal.alarm(0)
    print("\n\n✗ Interrupted by user")
    sys.exit(1)
except Exception as e:
    signal.alarm(0)
    print(f"\n✗ Session creation failed: {e}")
    print(f"\nError type: {type(e).__name__}")
    import traceback

    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
