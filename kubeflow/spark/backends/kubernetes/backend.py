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

"""Kubernetes backend for Spark operations."""

from collections import deque
from collections.abc import Iterator
import contextlib
import logging
import multiprocessing
import os
import random
import socket
import subprocess
import sys
import threading
import time

from kubeflow_spark_api import models
from kubernetes import client, config
from pyspark.sql import SparkSession

from kubeflow.common import constants as common_constants
from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.base import RuntimeBackend
from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.backends.kubernetes.utils import (
    build_service_url,
    build_spark_connect_cr,
    generate_session_name,
    get_spark_connect_info_from_cr,
)
from kubeflow.spark.types.options import Name
from kubeflow.spark.types.types import Driver, Executor, SparkConnectInfo, SparkConnectState

logger = logging.getLogger(__name__)

_spark_debug_logging_enabled = False


def _enable_spark_debug_logging() -> None:
    """Turn on INFO logging for kubeflow.spark to stderr (for E2E debug)."""
    global _spark_debug_logging_enabled
    if _spark_debug_logging_enabled:
        return
    _spark_debug_logging_enabled = True
    root = logging.getLogger("kubeflow.spark")
    root.setLevel(logging.INFO)
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        root.addHandler(h)


def _drain_stderr(proc: subprocess.Popen) -> deque:
    """Drain proc.stderr in a background thread to prevent pipe buffer deadlock.

    Returns a deque (maxlen=50) that accumulates the last 50 stderr lines for
    diagnostics. Must be called immediately after Popen so the buffer never fills.
    """
    buf: deque = deque(maxlen=50)

    def _reader() -> None:
        if proc.stderr is None:
            return
        for raw in proc.stderr:
            buf.append(raw.decode("utf-8", errors="replace").rstrip())

    threading.Thread(target=_reader, daemon=True).start()
    return buf


class KubernetesBackend(RuntimeBackend):
    """Kubernetes backend for managing SparkConnect sessions."""

    def __init__(self, backend_config: KubernetesBackendConfig):
        """Initialize Kubernetes Spark backend."""
        self.namespace = backend_config.namespace or "default"

        if backend_config.config_file:
            config.load_kube_config(config_file=backend_config.config_file)
        elif backend_config.context:
            config.load_kube_config(context=backend_config.context)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()

    def _extract_name_option(self, options: list | None) -> tuple[str, list]:
        """Extract Name option from options list, or generate name if absent.

        Args:
            options: List of option objects (Labels, Annotations, etc.).

        Returns:
            Tuple of (session_name, filtered_options):
            - session_name: Name from Name option, or auto-generated name
            - filtered_options: Options list with Name option removed
        """
        if not options:
            return generate_session_name(), []

        name_from_option = None
        filtered_options = []

        for option in options:
            if isinstance(option, Name):
                name_from_option = option.name
                # Don't add Name option to filtered list
            else:
                filtered_options.append(option)

        # Use Name option if provided, otherwise auto-generate
        session_name = name_from_option if name_from_option else generate_session_name()

        return session_name, filtered_options

    def _create_session(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
    ) -> SparkConnectInfo:
        """Create a new SparkConnect session (INTERNAL USE ONLY)."""
        # Extract Name option if present, or auto-generate
        name, filtered_options = self._extract_name_option(options)

        spark_connect = build_spark_connect_cr(
            name=name,
            namespace=self.namespace,
            num_executors=num_executors,
            resources_per_executor=resources_per_executor,
            spark_conf=spark_conf,
            driver=driver,
            executor=executor,
            options=filtered_options,  # Use filtered list
            backend=self,  # Pass backend for option validation
        )

        logger.info("Creating SparkConnect session '%s'", name)

        try:
            thread = self.custom_api.create_namespaced_custom_object(
                group=constants.SPARK_CONNECT_GROUP,
                version=constants.SPARK_CONNECT_VERSION,
                namespace=self.namespace,
                plural=constants.SPARK_CONNECT_PLURAL,
                body=spark_connect.to_dict(),
                async_req=True,
            )
            response = thread.get(common_constants.DEFAULT_TIMEOUT)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to create {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
        except client.ApiException as e:
            raise RuntimeError(
                f"Failed to create {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name} "
                f"(HTTP {e.status}: {e.body})"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e

        spark_connect_cr = models.SparkV1alpha1SparkConnect.from_dict(response)
        return get_spark_connect_info_from_cr(spark_connect_cr)

    def get_session(self, name: str) -> SparkConnectInfo:
        """Get information about a SparkConnect session."""
        try:
            thread = self.custom_api.get_namespaced_custom_object(
                group=constants.SPARK_CONNECT_GROUP,
                version=constants.SPARK_CONNECT_VERSION,
                namespace=self.namespace,
                plural=constants.SPARK_CONNECT_PLURAL,
                name=name,
                async_req=True,
            )
            response = thread.get(common_constants.DEFAULT_TIMEOUT)

            spark_connect_cr = models.SparkV1alpha1SparkConnect.from_dict(response)
            return get_spark_connect_info_from_cr(spark_connect_cr)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
        except client.ApiException as e:
            if e.status == 404:
                raise RuntimeError(
                    f"{constants.SPARK_CONNECT_KIND} not found: {self.namespace}/{name}"
                ) from e
            raise RuntimeError(
                f"Failed to get {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e

    def list_sessions(self) -> list[SparkConnectInfo]:
        """List all SparkConnect sessions."""
        try:
            thread = self.custom_api.list_namespaced_custom_object(
                group=constants.SPARK_CONNECT_GROUP,
                version=constants.SPARK_CONNECT_VERSION,
                namespace=self.namespace,
                plural=constants.SPARK_CONNECT_PLURAL,
                async_req=True,
            )
            response = thread.get(common_constants.DEFAULT_TIMEOUT)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to list {constants.SPARK_CONNECT_KIND}s in namespace: {self.namespace}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to list {constants.SPARK_CONNECT_KIND}s in namespace: {self.namespace}"
            ) from e

        spark_connect_list = models.SparkV1alpha1SparkConnectList.from_dict(response)
        return [get_spark_connect_info_from_cr(sc) for sc in spark_connect_list.items]

    def delete_session(self, name: str) -> None:
        """Delete a SparkConnect session."""
        try:
            thread = self.custom_api.delete_namespaced_custom_object(
                group=constants.SPARK_CONNECT_GROUP,
                version=constants.SPARK_CONNECT_VERSION,
                namespace=self.namespace,
                plural=constants.SPARK_CONNECT_PLURAL,
                name=name,
                async_req=True,
            )
            thread.get(common_constants.DEFAULT_TIMEOUT)
            logger.info("Deleted SparkConnect session '%s'", name)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to delete {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
        except client.ApiException as e:
            if e.status == 404:
                raise RuntimeError(
                    f"{constants.SPARK_CONNECT_KIND} not found: {self.namespace}/{name}"
                ) from e
            raise RuntimeError(
                f"Failed to delete {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e

    def _wait_for_session_ready(
        self,
        name: str,
        timeout: int = 300,
        polling_interval: int = 2,
    ) -> SparkConnectInfo:
        """Wait for a SparkConnect session to become ready (INTERNAL USE ONLY)."""
        start_time = time.time()
        last_log_time = 0.0

        while True:
            info = self.get_session(name)

            if info.state in (SparkConnectState.READY, SparkConnectState.RUNNING):
                logger.info(
                    "Session ready: %s/%s state=%s serviceName=%s (%.0fs)",
                    self.namespace,
                    name,
                    info.state,
                    info.service_name,
                    time.time() - start_time,
                )
                return info

            if info.state == SparkConnectState.FAILED:
                raise RuntimeError(
                    f"{constants.SPARK_CONNECT_KIND} failed: {self.namespace}/{name}"
                )

            now = time.time()
            if now - last_log_time >= 10.0:
                logger.info(
                    "Waiting for session: %s/%s state=%s serviceName=%s elapsed=%.0fs",
                    self.namespace,
                    name,
                    info.state,
                    info.service_name,
                    now - start_time,
                )
                last_log_time = now

            if now - start_time >= timeout:
                raise TimeoutError(
                    f"Timeout waiting for {constants.SPARK_CONNECT_KIND} to be ready: "
                    f"{self.namespace}/{name} (timeout: {timeout}s)"
                )

            time.sleep(polling_interval)

    def _wait_for_connect_port(
        self,
        host: str,
        port: int,
        timeout_sec: int = 60,
        interval_sec: float = 2.0,
        proc: subprocess.Popen | None = None,
    ) -> bool:
        """Wait until a TCP connection to host:port succeeds.

        If proc is provided, returns False immediately when the process exits
        rather than waiting out the full timeout.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                return False
            try:
                with socket.create_connection((host, port), timeout=2):
                    return True
            except OSError:
                time.sleep(interval_sec)
        return False

    def get_connect_url(
        self, info: SparkConnectInfo, local_port: int | None = None
    ) -> tuple[str, subprocess.Popen | None]:
        """Build connect URL; when running outside cluster, start port-forward and return localhost URL.

        When KUBERNETES_SERVICE_HOST is not set (e.g. local E2E), starts kubectl port-forward
        so the client can reach the Connect service via localhost.

        Args:
            info: Session info with service_name and namespace.
            local_port: Local port for port-forward (default: SPARK_CONNECT_PORT or env SPARK_CONNECT_LOCAL_PORT).

        Returns:
            (connect_url, port_forward_process or None). Caller may keep process reference;
            process exits when the Python process exits.
        """
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            url = build_service_url(info)
            logger.info("In-cluster connect URL: %s", url)
            return (url, None)

        logger.warning(
            "Running out-of-cluster. Using kubectl port-forward to reach the Spark Connect "
            "server. This is suitable for local development only. "
            "For production (EKS, GKE, AKS): expose the SparkConnect service as NodePort or "
            "LoadBalancer and use: client.connect(base_url='sc://<external-ip>:%d'). "
            "Set env var SPARK_CONNECT_LOCAL_PORT to pin the local port.",
            constants.SPARK_CONNECT_PORT,
        )

        port = local_port
        if port is None:
            port_str = os.environ.get("SPARK_CONNECT_LOCAL_PORT")
            port = int(port_str) if port_str else random.randint(15002, 16002)

        # Prefer pod when available (bypasses Service/EndpointSlice); then try svc names
        candidates: list[tuple[str, str]] = []
        if info.pod_name:
            candidates.append(("pod", info.pod_name))
        for svc in [f"{info.name}-svc", info.service_name, f"{info.name}-server"]:
            if svc and not any(c[0] == "svc" and c[1] == svc for c in candidates):
                candidates.append(("svc", svc))

        seen: set[str] = set()
        for kind, target in candidates:
            key = f"{kind}/{target}"
            if key in seen:
                continue
            seen.add(key)
            # Use 127.0.0.1 instead of localhost to force IPv4 (gRPC may prefer IPv6 which fails)
            url = f"sc://127.0.0.1:{port}"
            cmd = [
                "kubectl",
                "port-forward",
                key,
                f"{port}:{constants.SPARK_CONNECT_PORT}",
                "-n",
                info.namespace,
            ]
            logger.info(
                "Port-forward command: %s (connect_url=%s)",
                " ".join(cmd),
                url,
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Drain stderr immediately in background to prevent pipe buffer deadlock.
            # On WSL/Linux the 64 KB pipe buffer fills fast; if kubectl blocks on write
            # the tunnel appears dead even when it is healthy.
            stderr_buf = _drain_stderr(proc)

            # Active alive-check: poll up to 5 s in short intervals instead of a blind sleep.
            # A blind sleep(3) is a race — too short on slow systems, wasteful on fast ones.
            alive_deadline = time.monotonic() + 5.0
            while time.monotonic() < alive_deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)

            if proc.poll() is not None:
                err_msg = "\n".join(stderr_buf) or "(no stderr)"
                logger.warning(
                    "Port-forward to %s failed (exit %s): %s",
                    key,
                    proc.returncode,
                    err_msg,
                )
                continue

            # Pass proc so the wait loop returns False immediately if port-forward dies,
            # rather than hanging for the full timeout_sec probing a dead tunnel.
            if self._wait_for_connect_port("127.0.0.1", port, timeout_sec=90, proc=proc):
                if proc.poll() is not None:
                    err_msg = "\n".join(stderr_buf) or "(no stderr)"
                    logger.warning(
                        "Port-forward to %s died after port check (exit %s): %s",
                        key,
                        proc.returncode,
                        err_msg,
                    )
                    continue
                logger.info(
                    "Connect URL: %s | port-forward: %s -> localhost:%s | namespace=%s",
                    url,
                    key,
                    port,
                    info.namespace,
                )
                return (url, proc)

            proc.terminate()
            proc.wait(timeout=5)
            err_msg = "\n".join(stderr_buf) or "(no stderr)"
            logger.warning(
                "Port %s did not become reachable in time for %s. stderr: %s",
                port,
                key,
                err_msg,
            )

        raise RuntimeError(f"Port-forward failed for all candidates in {info.namespace}")

    def connect(
        self,
        info: SparkConnectInfo,
        connect_timeout: int = 120,
        grpc_ready_delay: int | None = None,
    ) -> SparkSession:
        """Connect to a Spark Connect session and return a SparkSession.

        This method handles all the connection logic including:
        - Getting the connect URL (with port-forwarding if needed)
        - Waiting for gRPC server readiness
        - Creating the SparkSession with timeout handling

        Args:
            info: SparkConnectInfo for the session to connect to.
            connect_timeout: Timeout in seconds for SparkSession.getOrCreate().
            grpc_ready_delay: Delay in seconds to wait for gRPC server readiness.
                If None, uses SPARK_CONNECT_READY_DELAY_SEC env var or default (3s).

        Returns:
            Connected SparkSession.

        Raises:
            TimeoutError: If connection times out.
            RuntimeError: If port-forward fails.
        """
        # Pin local_port before the first get_connect_url call so that if the first
        # attempt fails and a restart is needed, both calls use the same port.
        # The gRPC client URL is built once; a restart that picks a different port
        # would silently connect to a tunnel the client never knows about.
        pinned_local_port: int | None = None
        if not os.environ.get("KUBERNETES_SERVICE_HOST"):
            port_str = os.environ.get("SPARK_CONNECT_LOCAL_PORT")
            pinned_local_port = int(port_str) if port_str else random.randint(15002, 16002)

        # Get connect URL (handles port-forwarding for local development)
        connect_url, pf_proc = self.get_connect_url(info, local_port=pinned_local_port)
        local_port = pinned_local_port

        # Check port-forward process status
        if pf_proc is not None and pf_proc.poll() is not None:
            raise RuntimeError(
                f"Port-forward process exited with code {pf_proc.returncode} before connect."
            )

        # Log connection info
        try:
            import pyspark as _pyspark

            logger.info(
                "Connect URL: %s | PySpark client version: %s | connect_timeout=%ss",
                connect_url,
                getattr(_pyspark, "__version__", "unknown"),
                connect_timeout,
            )
        except Exception:
            logger.info("Connect URL: %s | connect_timeout=%ss", connect_url, connect_timeout)

        # Determine gRPC readiness delay
        delay_sec = grpc_ready_delay
        if delay_sec is None:
            delay_env = os.environ.get("SPARK_CONNECT_READY_DELAY_SEC")
            if delay_env is not None:
                with contextlib.suppress(ValueError):
                    delay_sec = int(delay_env)
            if delay_sec is None:
                delay_sec = 5 if os.environ.get("SPARK_E2E_DEBUG") else 3

        if delay_sec > 0:
            logger.info("Waiting %ss for Spark Connect server gRPC readiness", delay_sec)
            grpc_deadline = time.monotonic() + delay_sec
            while time.monotonic() < grpc_deadline:
                if pf_proc is not None and pf_proc.poll() is not None:
                    logger.warning("Port-forward died during gRPC ready wait, restarting...")
                    # get_connect_url already waits for the port to become reachable internally,
                    # so we break out of this loop immediately after restart rather than
                    # re-probing with a 1 s window (which always fails on a brand-new process).
                    connect_url, pf_proc = self.get_connect_url(info, local_port=local_port)
                    break
                time.sleep(0.5)

        # Final port-forward health check before connection attempt
        if pf_proc is not None and pf_proc.poll() is not None:
            logger.warning("Port-forward died before connect, restarting...")
            connect_url, pf_proc = self.get_connect_url(info, local_port=local_port)

        # Create SparkSession with timeout
        result: list = []
        exc_holder: list = []

        def _get_or_create() -> None:
            try:
                session = SparkSession.builder.remote(connect_url).getOrCreate()
                result.append(session)
            except Exception as e:
                exc_holder.append(e)

        thread = threading.Thread(target=_get_or_create, daemon=True)
        thread.start()
        thread.join(timeout=connect_timeout)

        if not thread.is_alive():
            if exc_holder:
                raise exc_holder[0]
            if result:
                return result[0]

        # Connection timed out
        base_msg = (
            f"Spark Connect connection to {connect_url} did not complete "
            f"within {connect_timeout}s. "
            "Verify: (1) port-forward target is the Spark Connect server pod, "
            "(2) PySpark and server Spark major.minor match, "
            "(3) driver pod logs for gRPC/auth errors; "
            "see Spark sql/connect for server config."
        )
        if pf_proc is not None and pf_proc.poll() is not None:
            stderr_b = pf_proc.stderr.read() if pf_proc.stderr else b""
            stderr_str = stderr_b.decode("utf-8", errors="replace").strip() if stderr_b else ""
            base_msg += (
                f" Port-forward process exited during connect "
                f"(code={pf_proc.returncode}). stderr: {stderr_str}"
            )
        raise TimeoutError(base_msg)

    def create_and_connect(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
        timeout: int = 300,
        connect_timeout: int = 120,
    ) -> SparkSession:
        """Create a new SparkConnect session and connect to it.

        This method handles the full session lifecycle:
        1. Creates a new session via _create_session
        2. Waits for session to become ready
        3. Connects to the session and returns SparkSession

        Args:
            num_executors: Number of executor instances.
            resources_per_executor: Resource requirements per executor.
            spark_conf: Spark configuration properties.
            driver: Driver configuration.
            executor: Executor configuration.
            options: List of configuration options (use Name option for custom name).
            timeout: Timeout in seconds to wait for session ready.
            connect_timeout: Timeout in seconds for SparkSession.getOrCreate().

        Returns:
            Connected SparkSession.

        Raises:
            TimeoutError: If session creation or connection times out.
            RuntimeError: If session creation or connection fails.
        """
        if os.environ.get("SPARK_E2E_DEBUG"):
            _enable_spark_debug_logging()

        info = self._create_session(
            num_executors=num_executors,
            resources_per_executor=resources_per_executor,
            spark_conf=spark_conf,
            driver=driver,
            executor=executor,
            options=options,
        )
        logger.info(
            "Created session %s/%s, waiting for ready (timeout=%ss)",
            info.namespace,
            info.name,
            timeout,
        )

        info = self._wait_for_session_ready(info.name, timeout=timeout)
        logger.info("Session ready, connecting (service_name=%s)", info.service_name)

        return self.connect(info, connect_timeout=connect_timeout)

    def get_session_logs(
        self,
        name: str,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a SparkConnect session."""
        info = self.get_session(name)

        if not info.pod_name:
            raise RuntimeError(
                f"No server pod for {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            )

        try:
            if follow:
                thread = self.core_api.read_namespaced_pod_log(
                    name=info.pod_name,
                    namespace=self.namespace,
                    follow=True,
                    _preload_content=False,
                    async_req=True,
                )
                resp = thread.get(common_constants.DEFAULT_TIMEOUT)
                for line in resp.stream():
                    yield line.decode("utf-8").rstrip("\n")
            else:
                thread = self.core_api.read_namespaced_pod_log(
                    name=info.pod_name,
                    namespace=self.namespace,
                    async_req=True,
                )
                logs = thread.get(common_constants.DEFAULT_TIMEOUT)
                for line in logs.split("\n"):
                    yield line
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to get logs for {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get logs for {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
