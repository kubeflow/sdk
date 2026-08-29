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

"""Unit tests for the E2E cluster watcher helpers."""

from unittest.mock import patch

from test.e2e.spark.cluster_watcher import (
    _driver_pod_from_sparkapplication,
    _driver_pods,
    _snapshot,
)


def test_snapshot_includes_sparkapplication_section() -> None:
    """Snapshot should include SparkApplication state alongside SparkConnect state."""
    with patch("test.e2e.spark.cluster_watcher._run_kubectl") as mock_run:
        mock_run.side_effect = [
            "",
            "",
            "",
            "",
            "",
        ]

        snapshot = _snapshot("spark-test", 12.0)

    assert any("SparkConnect" in line for line in snapshot)
    assert any("SparkApplication" in line for line in snapshot)


def test_driver_pod_from_sparkapplication_reads_status_field() -> None:
    """The watcher should read the driver pod from SparkApplication status."""
    with patch("test.e2e.spark.cluster_watcher._run_kubectl") as mock_run:
        mock_run.return_value = "spark-app-driver-abc123"

        pod = _driver_pod_from_sparkapplication("spark-test")

    assert pod == "spark-app-driver-abc123"


def test_driver_pods_collects_from_both_resources() -> None:
    """The watcher should collect driver pods from both SparkConnect and SparkApplication."""
    with patch(
        "test.e2e.spark.cluster_watcher._driver_pod_from_sparkconnect",
        return_value="spark-connect-driver",
    ), patch(
        "test.e2e.spark.cluster_watcher._driver_pod_from_sparkapplication",
        return_value="spark-application-driver",
    ):
        pods = _driver_pods("spark-test")

    assert pods == ["spark-connect-driver", "spark-application-driver"]
