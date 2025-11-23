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

"""Integration tests for Spark Connect backend.

These tests require a running Spark Connect server.

Setup:
1. Install PySpark with Connect support:
   pip install 'pyspark[connect]>=3.4.0'

2. Start local Spark Connect server:
   $SPARK_HOME/sbin/start-connect-server.sh \
       --packages org.apache.spark:spark-connect_2.12:3.5.0

   Or using Docker:
   docker run -p 15002:15002 apache/spark:3.5.0 \
       /opt/spark/sbin/start-connect-server.sh

3. Run tests:
   pytest kubeflow/spark/test/test_connect_integration.py -v

Environment variables:
- SPARK_CONNECT_URL: Spark Connect URL (default: sc://localhost:15002)
- SKIP_INTEGRATION_TESTS: Set to skip these tests (useful in CI)
"""

import os
import sys

import pytest


def _is_pyspark_available() -> bool:
    """Check if PySpark Connect is available."""
    try:
        import pyspark  # noqa: F401

        return True
    except ImportError:
        return False


def _should_skip_integration_tests() -> bool:
    """Check if integration tests should be skipped."""
    return os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true"


def _get_connect_url() -> str:
    """Get Spark Connect URL from environment."""
    return os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")


pytestmark = pytest.mark.skipif(
    not _is_pyspark_available() or _should_skip_integration_tests(),
    reason="PySpark Connect not installed or integration tests disabled",
)


class TestConnectBackendIntegration:
    """Integration tests for ConnectBackend with real Spark Connect server."""

    def test_create_and_close_session(self):
        """Test creating and closing a session."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False, timeout=30)

        client = SparkSessionClient(backend_config=config)

        try:
            session = client.create_session(app_name="test-session")
            assert session is not None
            assert session.session_id is not None
            assert session.app_name == "test-session"
            assert not session.is_closed

            session.close()
            assert session.is_closed
        finally:
            client.close()

    def test_simple_sql_query(self):
        """Test executing a simple SQL query."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session(app_name="sql-test")

            try:
                df = session.sql("SELECT 1 AS id, 'test' AS name")
                result = df.collect()

                assert len(result) == 1
                assert result[0].id == 1
                assert result[0].name == "test"
            finally:
                session.close()

    def test_create_dataframe_and_show(self):
        """Test creating a DataFrame and showing data."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session(app_name="dataframe-test")

            try:
                data = [
                    (1, "Alice", 28),
                    (2, "Bob", 35),
                    (3, "Carol", 42),
                ]
                df = session.createDataFrame(data, ["id", "name", "age"])

                assert df.count() == 3

                result = df.collect()
                assert len(result) == 3
                assert result[0].name == "Alice"
                assert result[1].age == 35

                print("\nDataFrame content:")
                df.show()
            finally:
                session.close()

    def test_dataframe_transformations(self):
        """Test DataFrame transformations (filter, select, groupBy)."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session(app_name="transform-test")

            try:
                data = [
                    (1, "Engineering", 100000),
                    (2, "Engineering", 120000),
                    (3, "Sales", 80000),
                    (4, "Sales", 90000),
                    (5, "Marketing", 85000),
                ]
                df = session.createDataFrame(data, ["id", "department", "salary"])

                filtered = df.filter(df.salary > 85000)
                assert filtered.count() == 4

                selected = df.select("department", "salary")
                assert len(selected.columns) == 2

                grouped = df.groupBy("department").count()
                result = grouped.collect()
                assert len(result) == 3

                print("\nGrouped by department:")
                grouped.show()
            finally:
                session.close()

    def test_session_metrics(self):
        """Test session metrics collection."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session(app_name="metrics-test")

            try:
                initial_metrics = session.get_metrics()
                assert initial_metrics.queries_executed == 0

                session.sql("SELECT 1")
                session.sql("SELECT 2")

                updated_metrics = session.get_metrics()
                assert updated_metrics.queries_executed == 2
            finally:
                session.close()

    def test_multiple_sessions(self):
        """Test creating multiple concurrent sessions."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session1 = client.create_session(app_name="session-1")
            session2 = client.create_session(app_name="session-2")

            try:
                assert session1.session_id != session2.session_id

                sessions = client.list_sessions()
                assert len(sessions) == 2

                df1 = session1.sql("SELECT 'session1' AS source")
                df2 = session2.sql("SELECT 'session2' AS source")

                assert df1.collect()[0].source == "session1"
                assert df2.collect()[0].source == "session2"
            finally:
                session1.close()
                session2.close()

    def test_range_dataframe(self):
        """Test creating range DataFrame."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session(app_name="range-test")

            try:
                df = session.range(0, 10, 2)
                assert df.count() == 5

                result = df.collect()
                assert result[0].id == 0
                assert result[1].id == 2
                assert result[4].id == 8

                print("\nRange DataFrame:")
                df.show()
            finally:
                session.close()

    def test_context_manager(self):
        """Test session context manager."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)
        client = SparkSessionClient(backend_config=config)

        with client.create_session(app_name="context-test") as session:
            df = session.sql("SELECT 42 AS answer")
            result = df.collect()
            assert result[0].answer == 42

        assert session.is_closed

    def test_get_session_info(self):
        """Test getting session information."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session(app_name="info-test")

            try:
                info = session.get_info()
                assert info.session_id == session.session_id
                assert info.app_name == "info-test"
                assert info.state == "active"
                assert info.metrics is not None

                status = client.get_session_status(session.session_id)
                assert status.session_id == session.session_id
                assert status.app_name == "info-test"
            finally:
                session.close()


class TestConnectBackendErrorHandling:
    """Test error handling in ConnectBackend."""

    def test_connection_to_invalid_server(self):
        """Test connection to non-existent server."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(
            connect_url="sc://nonexistent-host:99999", use_ssl=False, timeout=5
        )

        with SparkSessionClient(backend_config=config) as client:
            with pytest.raises(Exception):
                client.create_session(app_name="fail-test")

    def test_query_on_closed_session(self):
        """Test querying after session is closed."""
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        with SparkSessionClient(backend_config=config) as client:
            session = client.create_session(app_name="closed-test")
            session.close()

            with pytest.raises(RuntimeError, match="closed"):
                session.sql("SELECT 1")


def main():
    """Run integration tests manually."""
    print("=" * 80)
    print("Spark Connect Integration Tests")
    print("=" * 80)
    print(f"\nConnect URL: {_get_connect_url()}")
    print(f"PySpark available: {_is_pyspark_available()}")
    print(f"Skip integration tests: {_should_skip_integration_tests()}")

    if not _is_pyspark_available():
        print("\nERROR: PySpark Connect not installed!")
        print("Install with: pip install 'pyspark[connect]>=3.4.0'")
        sys.exit(1)

    if _should_skip_integration_tests():
        print("\nINFO: Integration tests disabled (SKIP_INTEGRATION_TESTS=true)")
        sys.exit(0)

    print("\n" + "=" * 80)
    print("Running basic connectivity test...")
    print("=" * 80)

    try:
        from kubeflow.spark import ConnectBackendConfig, SparkSessionClient

        config = ConnectBackendConfig(connect_url=_get_connect_url(), use_ssl=False)

        print(f"\nConnecting to: {_get_connect_url()}")

        with SparkSessionClient(backend_config=config) as client:
            print("✓ Client created successfully")

            session = client.create_session(app_name="manual-test")
            print(f"✓ Session created: {session.session_id}")

            try:
                print("\n" + "-" * 80)
                print("Test 1: Simple SQL Query")
                print("-" * 80)
                df = session.sql("SELECT 1 AS id, 'Hello Spark Connect!' AS message")
                result = df.collect()
                print(f"✓ Query executed: {result[0].message}")
                df.show()

                print("\n" + "-" * 80)
                print("Test 2: Create DataFrame")
                print("-" * 80)
                data = [
                    (1, "Alice", 28),
                    (2, "Bob", 35),
                    (3, "Carol", 42),
                ]
                df = session.createDataFrame(data, ["id", "name", "age"])
                print(f"✓ DataFrame created with {df.count()} rows")
                df.show()

                print("\n" + "-" * 80)
                print("Test 3: DataFrame Transformations")
                print("-" * 80)
                filtered = df.filter(df.age > 30)
                print(f"✓ Filtered to {filtered.count()} rows (age > 30)")
                filtered.show()

                print("\n" + "-" * 80)
                print("Test 4: Session Metrics")
                print("-" * 80)
                metrics = session.get_metrics()
                print(f"✓ Queries executed: {metrics.queries_executed}")
                print(f"✓ Active queries: {metrics.active_queries}")

                print("\n" + "=" * 80)
                print("All tests passed! ✓")
                print("=" * 80)

            finally:
                session.close()
                print("\n✓ Session closed")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
