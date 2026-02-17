#!/usr/bin/env python3
# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SparkClient Demo with Existing SparkConnect Cluster.

Prerequisites:
    1. Kind cluster with Spark Operator installed:
       ./scripts/spark/setup-kind.sh

    2. pyspark installed (optional, for connect() API):
       uv pip install pyspark

Run:
    uv run python examples/spark/demo_existing_sparkconnect.py
"""

import os
import subprocess
import sys
import time

# Configuration
NAMESPACE = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
KUBECONFIG = os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))

print("=" * 70)
print("SparkClient Real Cluster Demo")
print("=" * 70)
print(f"Namespace: {NAMESPACE}")
print(f"Kubeconfig: {KUBECONFIG}")
print("=" * 70)


def kubectl(cmd: str, check: bool = True) -> str:
    """Run kubectl command and return output."""
    full_cmd = f"kubectl -n {NAMESPACE} {cmd}"
    print(f"\n$ {full_cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"kubectl failed: {result.stderr}")
    return result.stdout


def wait_and_show(name: str, timeout: int = 120):
    """Wait for SparkConnect to be ready and show status."""
    print(f"\nWaiting for {name} to be Ready (timeout={timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            f"kubectl -n {NAMESPACE} get sparkconnect {name} -o jsonpath='{{.status.state}}'",
            shell=True,
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip().strip("'")
        if state == "Ready":
            print(f"{name} is Ready!")
            return True
        print(f"   State: {state or 'Pending'}...")
        time.sleep(5)
    print(f"Timeout waiting for {name}")
    return False


# ============================================================
# Check cluster connectivity
# ============================================================
print("\n" + "=" * 70)
print("Step 0: Check Cluster Connectivity")
print("=" * 70)

try:
    kubectl("get namespace spark-test", check=True)
    print("Cluster is accessible")
except Exception as e:
    print(f"Cannot connect to cluster: {e}")
    print("\nRun this first to setup the cluster:")
    print("  ./scripts/spark/setup-kind.sh")
    sys.exit(1)

# ============================================================
# Initialize SparkClient
# ============================================================
print("\n" + "=" * 70)
print("Step 1: Initialize SparkClient")
print("=" * 70)

from kubeflow.common.types import KubernetesBackendConfig  # noqa: E402
from kubeflow.spark import Driver, Executor, Name, SparkClient, SparkConnectState  # noqa: E402

client = SparkClient(backend_config=KubernetesBackendConfig(namespace=NAMESPACE))
print("SparkClient initialized")
print(f"   Backend: {type(client.backend).__name__}")
print(f"   Namespace: {client.backend.namespace}")

# ============================================================
# Example 1: Create SparkConnect Session with Defaults
# ============================================================
print("\n" + "=" * 70)
print("Example 1: Create SparkConnect Session with Defaults")
print("=" * 70)

print("\n# Python API:")
print("spark = client.connect(timeout=120)")
spark = client.connect(timeout=120)
session_name = None

# Get the session name from list
sessions = client.list_sessions()
if sessions:
    session_name = sessions[-1].name
    info = sessions[-1]
    print(f"\nSession created: {info.name}")
    print(f"   State: {info.state.value}")
    print(f"   Namespace: {info.namespace}")

    print("\n# Verify with kubectl:")
    kubectl(f"get sparkconnect {session_name} -o wide")

    # Session is ready because connect() waits
    print("\n# Session Details:")
    print(f"   Name: {info.name}")
    print(f"   State: {info.state.value}")
    print(f"   Server Pod: {info.pod_name}")
    print(f"   Service: {info.service_name}")
    print(f"   Service URL: {info.service_url}")

    print("\n# Verify pods:")
    kubectl("get pods -l app.kubernetes.io/component=server")

    print("\n# Verify service:")
    kubectl(f"get svc {info.service_name}")

spark.stop()

# ============================================================
# Example 2: Create Named Session with Custom Config
# ============================================================
print("\n" + "=" * 70)
print("Example 2: Create Named Session with Custom Config")
print("=" * 70)

print("\n# Python API:")
print("""
driver = Driver(cores=1, memory="512m")
executor = Executor(cores=1, memory="512m", num_instances=2)
spark2 = client.connect(
    options=[Name("my-spark-session")],
    driver=driver,
    executor=executor,
    timeout=120
)
""")

driver = Driver(cores=1, memory="512m")
executor = Executor(cores=1, memory="512m", num_instances=2)
spark2 = client.connect(
    options=[Name("my-spark-session")], driver=driver, executor=executor, timeout=120
)

info2 = client.get_session("my-spark-session")
print(f"\nSession created: {info2.name}")

print("\n# Verify with kubectl:")
kubectl("get sparkconnect my-spark-session -o wide")

print("\n# View SparkConnect CRD YAML:")
kubectl("get sparkconnect my-spark-session -o yaml | head -50")

spark2.stop()

# ============================================================
# Example 3: List All Sessions
# ============================================================
print("\n" + "=" * 70)
print("Example 3: List All Sessions")
print("=" * 70)

print("\n# Python API:")
print("sessions = client.list_sessions()")
sessions = client.list_sessions()

print(f"\nFound {len(sessions)} sessions:")
for s in sessions:
    print(f"   - {s.name}: {s.state.value}")

print("\n# Verify with kubectl:")
kubectl("get sparkconnect")

# ============================================================
# Example 4: Get Session Details
# ============================================================
print("\n" + "=" * 70)
print("Example 4: Get Session Details")
print("=" * 70)

print("\n# Python API:")
print(f'info = client.get_session("{session_name}")')
info = client.get_session(session_name)

print("\nSession details:")
print(f"   name: {info.name}")
print(f"   namespace: {info.namespace}")
print(f"   state: {info.state.value}")
print(f"   server_pod_name: {info.server_pod_name}")
print(f"   service_name: {info.service_name}")
print(f"   created_at: {info.created_at}")

# ============================================================
# Example 5: SparkClient.connect() - Get PySpark SparkSession
# ============================================================
print("\n" + "=" * 70)
print("Example 5: SparkClient.connect() - Get PySpark SparkSession")
print("=" * 70)

print("\n# Python API:")
print("""
# Option A: Create new session and connect
spark = client.connect(options=[Name("my-session")], timeout=120)

# Option B: Auto-create with defaults (auto-generated name)
spark = client.connect(timeout=120)

# Option C: Connect to external Spark Connect server
spark = client.connect(base_url="sc://localhost:15002")
""")

# Check if pyspark is installed
try:
    import pyspark

    print(f"\npyspark is installed (version {pyspark.__version__})")

    # Session is already ready from connect()
    if info and info.state == SparkConnectState.READY and info.service_url:
        print(f"\n# Connected to: {info.service_url}")
        print("# Note: To connect from outside cluster, use port-forward")
        print("\n# Run in another terminal:")
        print(f"  kubectl -n {NAMESPACE} port-forward svc/{info.service_name} 15002:15002")
        print("\n# Then connect with:")
        print('  spark = client.connect(base_url="sc://localhost:15002")')
except ImportError:
    print("\npyspark not installed. Install with: uv pip install pyspark")
    print("\n# Once installed, you can create and connect with:")
    print('spark = client.connect(options=[Name("my-session")], timeout=120)')

# ============================================================
# Example 6: Get Session Logs
# ============================================================
print("\n" + "=" * 70)
print("Example 6: Get Session Logs")
print("=" * 70)

print("\n# Python API:")
print(f'logs = client.get_session_logs("{session_name}")')

try:
    logs = list(client.get_session_logs(session_name))
    print("\nLogs (last 20 lines):")
    for line in logs[-20:]:
        print(f"   {line}")
except Exception as e:
    print(f"\nCould not get logs: {e}")

print("\n# Verify with kubectl:")
kubectl(f"logs -l app.kubernetes.io/name={session_name} --tail=10", check=False)

# ============================================================
# Example 7: Delete Sessions
# ============================================================
print("\n" + "=" * 70)
print("Example 7: Delete Sessions")
print("=" * 70)

print("\n# Python API:")
print('client.delete_session("my-spark-session")')
client.delete_session("my-spark-session")
print("Deleted my-spark-session")

print(f'\nclient.delete_session("{session_name}")')
client.delete_session(session_name)
print(f"Deleted {session_name}")

print("\n# Verify with kubectl:")
kubectl("get sparkconnect")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("Summary: SparkClient API")
print("=" * 70)
print("""
from kubeflow.spark import Driver, Executor, Name, SparkClient
from kubeflow.common.types import KubernetesBackendConfig

# Initialize client
client = SparkClient(backend_config=KubernetesBackendConfig(namespace="spark-test"))

# Create and connect (session is ready when connect() returns)
spark = client.connect(
    options=[Name("my-session")],
    driver=Driver(cores=2, memory="2g"),
    executor=Executor(cores=2, memory="4g", num_instances=3),
    timeout=120
)

# Use PySpark
df = spark.range(100)
df.show()

# List/Get/Delete
sessions = client.list_sessions()
info = client.get_session("my-session")
logs = client.get_session_logs("my-session")
spark.stop()
client.delete_session("my-session")
""")
print("=" * 70)
