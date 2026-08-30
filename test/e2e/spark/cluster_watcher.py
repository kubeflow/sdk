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

"""Cluster watcher for E2E tests: logs SparkConnect/SparkApplication, pods, events, driver logs."""

import subprocess
import threading
import time


def _run_kubectl(args: list[str], namespace: str, timeout: int = 10) -> str:
    cmd = ["kubectl", "-n", namespace] + args
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (
            (r.stdout or "").strip()
            if r.returncode == 0
            else f"(exit {r.returncode}) {r.stderr or r.stdout or ''}"
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"(error) {e!s}"


def _snapshot(namespace: str, elapsed_sec: float) -> list[str]:
    lines = [f"--- T+{elapsed_sec:.0f}s ---"]
    sc = _run_kubectl(["get", "sparkconnect", "-o", "wide"], namespace)
    lines.append(f"SparkConnect:\n{sc}" if sc else "SparkConnect: (none)")
    sa = _run_kubectl(["get", "sparkapplication", "-o", "wide"], namespace)
    lines.append(f"SparkApplication:\n{sa}" if sa else "SparkApplication: (none)")
    pods = _run_kubectl(["get", "pods", "-o", "wide"], namespace)
    lines.append(f"Pods:\n{pods}" if pods else "Pods: (none)")
    events = _run_kubectl(
        [
            "get",
            "events",
            "--sort-by=.lastTimestamp",
            "-o",
            "custom-columns=LAST_TS:.lastTimestamp,TYPE:.type,REASON:.reason,MESSAGE:.message,OBJECT:.involvedObject.name",
        ],
        namespace,
    )
    events_tail = "\n".join(events.split("\n")[-15:]) if events else "(none)"
    lines.append(f"Events (last 15):\n{events_tail}")
    return lines


def _driver_pod_from_cluster(namespace: str) -> str | None:
    """Resolve a driver pod name from either SparkConnect or SparkApplication status."""
    sc_out = _run_kubectl(
        ["get", "sparkconnect", "-o", "jsonpath={.items[*].status.server.podName}"],
        namespace,
    )
    if sc_out and not sc_out.startswith("(exit") and not sc_out.startswith("(error)"):
        pods = [p for p in sc_out.strip().split() if p]
        if pods:
            return pods[0]

    sa_out = _run_kubectl(
        ["get", "sparkapplication", "-o", "jsonpath={.items[*].status.driverInfo.podName}"],
        namespace,
    )
    if sa_out and not sa_out.startswith("(exit") and not sa_out.startswith("(error)"):
        pods = [p for p in sa_out.strip().split() if p]
        if pods:
            return pods[0]

    return None


def _driver_logs(namespace: str, pod_name: str, tail: int = 25) -> str:
    return _run_kubectl(["logs", pod_name, f"--tail={tail}"], namespace, timeout=15)


def run_watcher(
    namespace: str,
    stop_event: threading.Event,
    log_out: list[str],
    interval_sec: float = 5.0,
    max_duration_sec: float = 330.0,
    log_driver_when_ready: bool = True,
) -> None:
    """Run cluster watcher until stop_event or max_duration; append to log_out.

    Each interval: logs SparkConnect/SparkApplication list, pods, events; when
    a driver pod appears (from SparkConnect or SparkApplication status),
    appends its logs so we see driver/executor startup and where time is spent.
    """
    start = time.monotonic()
    last_driver: str | None = None

    while not stop_event.is_set():
        elapsed = time.monotonic() - start
        if elapsed >= max_duration_sec:
            break
        for line in _snapshot(namespace, elapsed):
            log_out.append(line)
        driver_pod = _driver_pod_from_cluster(namespace)
        if driver_pod and (log_driver_when_ready or driver_pod != last_driver):
            last_driver = driver_pod
            dr_logs = _driver_logs(namespace, driver_pod)
            log_out.append(f"Driver pod {driver_pod} logs (tail 25):\n{dr_logs}")
        stop_event.wait(timeout=interval_sec)


def run_watcher_in_thread(
    namespace: str,
    interval_sec: float = 5.0,
    max_duration_sec: float = 330.0,
) -> tuple[threading.Event, list[str], threading.Thread]:
    """Start watcher in a daemon thread; return (stop_event, log_buffer, thread)."""
    stop_event = threading.Event()
    log_buffer: list[str] = []

    def target() -> None:
        run_watcher(
            namespace,
            stop_event,
            log_buffer,
            interval_sec=interval_sec,
            max_duration_sec=max_duration_sec,
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return stop_event, log_buffer, thread
