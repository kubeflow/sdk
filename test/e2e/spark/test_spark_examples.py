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
from kubeflow.spark.options import Name
from kubeflow.spark.types.types import SparkConnectState

from .cluster_watcher import run_watcher_in_thread

# Path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "spark"

EXAMPLE_TIMEOUT_SEC = 600
WATCHER_INTERVAL_SEC = 5.0


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
            parts.append(
                "\n--- Cluster watcher (SparkConnect / SparkApplication / pods / events / driver logs) ---"
            )
            parts.append("\n".join(watcher_log))
        parts.append("\n--- Example stdout ---")
        parts.append(stdout or "(empty)")
        parts.append("\n--- Example stderr ---")
        parts.append(stderr or "(empty)")
        return "\n".join(parts)

    def _run_example(self, example_script_name: str, namespace: str):
        """Run example as a subprocess against the ambient kubeconfig."""
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
        stdout = self._run_example("spark_connect_simple.py", namespace)
        assert "SparkConnect session created" in stdout or "Session" in stdout

    def test_spark_advanced_options_example(self):
        """EX02: Validate spark_advanced_options.py runs without errors."""
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
        stdout = self._run_example("spark_advanced_options.py", namespace)
        assert "Driver" in stdout or "Executor" in stdout

    def test_batch_job_lifecycle_example(self):
        """EX04: Validate batch_job_lifecycle.py runs without errors."""
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
        stdout = self._run_example("batch_job_lifecycle.py", namespace)
        assert "BATCH JOB LIFECYCLE COMPLETE!" in stdout

    def test_batch_failed_job_example(self):
        """EX05: Validate batch_failed_job.py handles failed Spark jobs."""
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
        stdout = self._run_example("batch_failed_job.py", namespace)
        assert "Job failed as expected." in stdout

    def test_batch_func_job_lifecycle_example(self):
        """EX06: Validate batch_func_job_lifecycle.py runs without errors."""
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
        stdout = self._run_example("batch_func_job_lifecycle.py", namespace)
        assert "FUNCJOB LIFECYCLE COMPLETE!" in stdout

    def test_batch_job_options_example(self):
        """EX07: Validate batch_job_options.py runs without errors."""
        namespace = os.environ.get("SPARK_TEST_NAMESPACE", "spark-test")
        stdout = self._run_example("batch_job_options.py", namespace)
        assert "BATCH JOB OPTIONS COMPLETE!" in stdout
