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

"""Tests to validate Spark examples work correctly."""

import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.kubernetes import KubernetesBackend
from kubeflow.spark.types.options import Name
from kubeflow.spark.types.types import SparkConnectState

from .cluster_watcher import run_watcher_in_thread
from .run_in_cluster import run_example_in_cluster

# Path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "spark"

EXAMPLE_TIMEOUT_SEC = 600
WATCHER_INTERVAL_SEC = 5.0

# In-cluster: run example as K8s Job so client uses sc://...svc.cluster.local (no port-forward).
USE_IN_CLUSTER = os.environ.get("SPARK_E2E_RUN_IN_CLUSTER") == "1"
RUNNER_IMAGE = os.environ.get("SPARK_E2E_RUNNER_IMAGE", "")


def _run_example_with_watcher(
    example_path: Path,
    namespace: str,
    timeout_sec: int = EXAMPLE_TIMEOUT_SEC,
) -> tuple[int | None, str, str, list[str]]:
    """Run example script with cluster watcher; return (returncode, stdout, stderr, watcher_log)."""
    stop_event, watcher_log, watcher_thread = run_watcher_in_thread(
        namespace,
        interval_sec=WATCHER_INTERVAL_SEC,
        max_duration_sec=timeout_sec + 30,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout(pipe) -> None:
        try:
            for line in pipe:
                stdout_lines.append(line)
        except (ValueError, OSError):
            pass

    def read_stderr(pipe) -> None:
        try:
            for line in pipe:
                stderr_lines.append(line)
        except (ValueError, OSError):
            pass

    proc = subprocess.Popen(
        [sys.executable, str(example_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "SPARK_TEST_NAMESPACE": namespace},
    )
    t_out = threading.Thread(target=read_stdout, args=(proc.stdout,), daemon=True)
    t_err = threading.Thread(target=read_stderr, args=(proc.stderr,), daemon=True)
    t_out.start()
    t_err.start()

    returncode: int | None = None
    try:
        returncode = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=2)

    stop_event.set()
    watcher_thread.join(timeout=WATCHER_INTERVAL_SEC + 2)

    stdout_str = "".join(stdout_lines)
    stderr_str = "".join(stderr_lines)
    return returncode, stdout_str, stderr_str, watcher_log


@pytest.mark.integration
@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_spark_connect_crd_smoke():
    """Create SparkConnect via SDK and verify API accepts it (CRD-only; no operator)."""
    namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
    backend = KubernetesBackend(KubernetesBackendConfig(namespace=namespace))
    name = "smoke-session"
    info = backend._create_session(options=[Name(name)])
    assert info.name == name
    assert info.namespace == namespace
    assert info.state in (SparkConnectState.PROVISIONING, SparkConnectState.READY)
    assert backend.get_session(name).name == name
    backend.delete_session(name)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(EXAMPLE_TIMEOUT_SEC + 120)
class TestSparkExamples:
    """Validate Spark examples execute successfully."""

    def _dump_on_failure(
        self,
        returncode: int | None,
        stdout: str,
        stderr: str,
        watcher_log: list[str],
        msg: str,
    ) -> str:
        """Build failure message with cluster watcher log and example output."""
        parts = [msg]
        if watcher_log:
            parts.append("\n--- Cluster watcher (SparkConnect / pods / events / driver logs) ---")
            parts.append("\n".join(watcher_log))
        parts.append("\n--- Example stdout ---")
        parts.append(stdout or "(empty)")
        parts.append("\n--- Example stderr ---")
        parts.append(stderr or "(empty)")
        return "\n".join(parts)

    def _run_example(self, example_script_name: str, namespace: str):
        """Run example: in-cluster Job if SPARK_E2E_RUN_IN_CLUSTER=1 and image set, else subprocess."""
        if USE_IN_CLUSTER and RUNNER_IMAGE:
            success, logs, job_desc = run_example_in_cluster(
                example_script_name,
                namespace,
                image=RUNNER_IMAGE,
                timeout_sec=EXAMPLE_TIMEOUT_SEC,
            )
            if not success:
                fail_msg = (
                    f"In-cluster example {example_script_name} failed.\n"
                    f"--- Job describe ---\n{job_desc}\n--- Pod logs ---\n{logs or '(empty)'}"
                )
                assert success, fail_msg
            assert (
                "Session" in logs
                or "complete" in logs.lower()
                or "Level" in logs
                or "Driver" in logs
                or "Executor" in logs
                or "EXAMPLE" in logs
                or "E2E: Starting" in logs
            ), f"In-cluster example produced no expected output. Logs:\n{logs or '(empty)'}"
            return
        example_path = EXAMPLES_DIR / example_script_name
        assert example_path.exists(), f"Example not found: {example_path}"
        returncode, stdout, stderr, watcher_log = _run_example_with_watcher(
            example_path, namespace, timeout_sec=EXAMPLE_TIMEOUT_SEC
        )
        fail_msg = self._dump_on_failure(
            returncode,
            stdout,
            stderr,
            watcher_log,
            f"Example exited with code {returncode} (expected 0).",
        )
        assert returncode == 0, fail_msg
        return stdout

    def test_spark_connect_simple_example(self):
        """EX01: Validate spark_connect_simple.py runs without errors."""
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
        if USE_IN_CLUSTER and RUNNER_IMAGE:
            self._run_example("spark_connect_simple.py", namespace)
            return
        stdout = self._run_example("spark_connect_simple.py", namespace)
        assert "SparkConnect session created" in stdout or "Session" in stdout

    def test_spark_advanced_options_example(self):
        """EX02: Validate spark_advanced_options.py runs without errors."""
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
        if USE_IN_CLUSTER and RUNNER_IMAGE:
            self._run_example("spark_advanced_options.py", namespace)
            return
        stdout = self._run_example("spark_advanced_options.py", namespace)
        assert "Driver" in stdout or "Executor" in stdout

    def test_connect_existing_session_example(self):
        """EX03: Validate connect_existing_session.py - base_url connect via two-client pattern.

        Runs in-cluster only (K8s Job mode).
        """
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")

        if not (USE_IN_CLUSTER and RUNNER_IMAGE):
            pytest.skip("Requires in-cluster execution (SPARK_E2E_RUN_IN_CLUSTER=1)")

        self._run_example("connect_existing_session.py", namespace)
