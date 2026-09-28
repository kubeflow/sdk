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

"""E2E Test: Connect to Existing SparkConnect Session (two-client pattern).

This example demonstrates the "bring your own server" use case where:
1. A setup client creates a SparkConnect server
2. A test client connects to the existing server via base_url

This validates the connect(base_url="sc://...") codepath which bypasses
session creation and directly connects to an existing Spark Connect server.

Usage:
    # Run in-cluster only (via K8s Job):
    SPARK_E2E_RUN_IN_CLUSTER=1 python examples/spark/connect_existing_session.py
"""

import os
import sys
import uuid

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark import Name, SparkClient
from kubeflow.spark.backends.kubernetes.utils import build_service_url


def _backend_config():
    """Backend config; uses SPARK_TEST_NAMESPACE in CI."""
    return KubernetesBackendConfig(namespace=os.environ.get("SPARK_TEST_NAMESPACE", "spark-test"))


def _unique_session_name() -> str:
    """Generate unique session name to avoid conflicts in E2E runs."""
    return f"connect-existing-{uuid.uuid4().hex[:8]}"


def test_connect_to_existing_session():
    """Test connect(base_url=...) with two clients.

    Two-client pattern:
    - Setup client: creates SparkConnect server, stops SparkSession (server stays running)
    - Test client: connects via base_url to the existing server
    """
    print("=" * 70)
    print("E2E: Connect to Existing SparkConnect Session")
    print("=" * 70)

    session_name = _unique_session_name()
    setup_client = None
    test_spark = None

    try:
        # Phase 1: Setup client creates SparkConnect server
        print("\n[Phase 1] Creating SparkConnect server...")
        setup_client = SparkClient(backend_config=_backend_config())
        setup_spark = setup_client.connect(options=[Name(session_name)], timeout=180)

        info = setup_client.get_session(session_name)
        service_url = build_service_url(info)
        print(f"   Session: {session_name}")
        print(f"   URL: {service_url}")

        setup_spark.stop()
        print("   Setup SparkSession stopped (server still running)")

        # Phase 2: Test client connects via base_url
        print("\n[Phase 2] Connecting via base_url...")
        test_client = SparkClient(backend_config=_backend_config())
        test_spark = test_client.connect(base_url=service_url)
        print("   Connected successfully!")

        # Phase 3: Validate with Spark operations
        print("\n[Phase 3] Validating...")
        count = test_spark.range(100).count()
        print(f"   spark.range(100).count() = {count}")
        assert count == 100, f"Expected 100, got {count}"

        print("\n[SUCCESS] connect(base_url=...) works correctly!")

    finally:
        # Phase 4: Cleanup
        print("\n[Phase 4] Cleanup...")
        if test_spark:
            try:
                test_spark.stop()
            except Exception as e:
                print(f"   Warning: {e}")
        if setup_client:
            try:
                setup_client.delete_session(session_name)
                print(f"   Deleted {session_name}")
            except Exception as e:
                print(f"   Warning: {e}")


def main():
    """Entry point for E2E test."""
    if os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") != "1":
        print("SKIP: Requires in-cluster execution (SPARK_E2E_RUN_IN_CLUSTER=1)")
        sys.exit(0)

    try:
        test_connect_to_existing_session()
        sys.exit(0)
    except Exception as e:
        print(f"\nFailed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
