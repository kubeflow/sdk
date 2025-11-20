#!/usr/bin/env python3
"""
Debug version of SDK client with verbose logging
"""

import logging
import os
import sys

# Setup very verbose logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add SDK to path
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, sdk_path)

print("=" * 80)
print("Testing Kubeflow SDK with DEBUG logging")
print("=" * 80)

# Import with logging
print("\n[1] Importing Kubeflow SDK...")
from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

print("\n[2] Creating config...")
config = ConnectBackendConfig(connect_url="sc://localhost:30000", use_ssl=False, timeout=60)
print(f"    Config: {config.connect_url}")

print("\n[3] Creating client...")
client = SparkSessionClient(backend_config=config)
print(f"    Client created: {client}")
print(f"    Backend: {client.backend}")

print("\n[4] Creating session (this is where it might hang)...")
print("    About to call client.create_session()...")

import signal


def timeout_handler(signum, frame):
    print("\n✗ Session creation timed out after 20 seconds")
    print("\nThe hang is in the SDK's create_session method.")
    print("Check: kubeflow/spark/backends/connect.py line ~241")
    sys.exit(1)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(20)

try:
    session = client.create_session(app_name="debug-test")
    signal.alarm(0)

    print("\n✓ Session created!")
    print(f"    Session ID: {session.session_id}")
    print(f"    App name: {session.app_name}")

    print("\n[5] Testing query...")
    df = session.sql("SELECT 1 AS id")
    result = df.collect()
    print(f"✓ Query result: {result}")

    session.close()
    print("✓ Test passed!")

except Exception as e:
    signal.alarm(0)
    print(f"\n✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
