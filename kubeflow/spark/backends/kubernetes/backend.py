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
from typing import Optional

from kubernetes import client, config
from pyspark.sql import SparkSession

from kubeflow.common import constants as common_constants
from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.backends.base import RuntimeBackend
from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.backends.kubernetes.utils import (
    build_service_url,
    build_spark_connect_crd,
    generate_session_name,
    parse_spark_connect_status,
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

    def _extract_name_option(self, options: Optional[list]) -> tuple[str, list]:
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
        num_executors: Optional[int] = None,
        resources_per_executor: Optional[dict[str, str]] = None,
        spark_conf: Optional[dict[str, str]] = None,
        driver: Optional[Driver] = None,
        executor: Optional[Executor] = None,
        options: Optional[list] = None,
    ) -> SparkConnectInfo:
        """Create a new SparkConnect session (INTERNAL USE ONLY)."""
        # Validate input types
        if resources_per_executor is not None and not isinstance(resources_per_executor, dict):
            raise TypeError(
                f"resources_per_executor must be a dict, got {type(resources_per_executor)}"
            )
        if spark_conf is not None and not isinstance(spark_conf, dict):
            raise TypeError(f"spark_conf must be a dict, got {type(spark_conf)}")
        if num_executors is not None and not isinstance(num_executors, int):
            raise TypeError(f"num_executors must be an int, got {type(num_executors)}")
        if driver is not None and not isinstance(driver, Driver):
            raise TypeError(f"driver must be a Driver instance, got {type(driver)}")
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError(f"executor must be an Executor instance, got {type(executor)}")

        # Extract Name option if present, or auto-generate
        name, filtered_options = self._extract_name_option(options)

        crd = build_spark_connect_crd(
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
                body=crd,
                async_req=True,
            )
            response = thread.get(common_constants.DEFAULT_TIMEOUT)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                f"Timeout to create {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {constants.SPARK_CONNECT_KIND}: {self.namespace}/{name}"
            ) from e

        return parse_spark_connect_status(response)

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
            return parse_spark_connect_status(response)
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

        return [parse_spark_connect_status(item) for item in response.get("items", [])]

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
        self, host: str, port: int, timeout_sec: int = 60, interval_sec: float = 2.0
    ) -> bool:
        """Wait until a TCP connection to host:port succeeds (Spark Connect server reachable)."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return True
            except OSError:
                time.sleep(interval_sec)
        return False

    def get_connect_url(
        self, info: SparkConnectInfo, local_port: Optional[int] = None
    ) -> tuple[str, Optional[subprocess.Popen]]:
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
            # Use 127.0.0.1 instead of localhost to force IPv4 (gRPC may prefer IPv6 which can fail)
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
            time.sleep(3.0)  # Allow port-forward to fully establish
            if proc.poll() is not None:
                stderr = (proc.stderr and proc.stderr.read()) or b""
                err_msg = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                logger.warning(
                    "Port-forward to %s failed (exit %s): %s",
                    key,
                    proc.returncode,
                    err_msg,
                )
                continue
            if self._wait_for_connect_port("127.0.0.1", port, timeout_sec=90):
                # Final verification: ensure process is still alive after port check
                if proc.poll() is not None:
                    stderr = (proc.stderr and proc.stderr.read()) or b""
                    err_msg = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
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
            logger.warning("Port %s did not become reachable in time for %s", port, key)
        raise RuntimeError(f"Port-forward failed for all candidates in {info.namespace}")

    def connect(
        self,
        info: SparkConnectInfo,
        connect_timeout: int = 120,
        grpc_ready_delay: Optional[int] = None,
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
        # Get connect URL (handles port-forwarding for local development)
        connect_url, pf_proc = self.get_connect_url(info)
        # Track local port for reconnection attempts
        local_port = int(connect_url.split(":")[-1]) if pf_proc else None

        # Check port-forward process status
        if pf_proc is not None and pf_proc.poll() is not None:
            stderr_b = pf_proc.stderr.read() if pf_proc.stderr else b""
            stderr_str = stderr_b.decode("utf-8", errors="replace").strip() if stderr_b else ""
            raise RuntimeError(
                f"Port-forward process exited with code {pf_proc.returncode} "
                f"before connect. stderr: {stderr_str}"
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
            # Use active probing instead of blind sleep to detect port-forward death early
            probe_start = time.monotonic()
            while time.monotonic() - probe_start < delay_sec:
                # Check if port-forward process died
                if pf_proc is not None and pf_proc.poll() is not None:
                    logger.warning("Port-forward died during gRPC ready wait, restarting...")
                    connect_url, pf_proc = self.get_connect_url(info, local_port=local_port)
                    local_port = int(connect_url.split(":")[-1]) if pf_proc else None
                # Verify port is still reachable
                if local_port and not self._wait_for_connect_port(
                    "127.0.0.1", local_port, timeout_sec=1, interval_sec=0.5
                ):
                    logger.warning("Port %s not reachable during gRPC ready wait", local_port)
                time.sleep(1)

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
        num_executors: Optional[int] = None,
        resources_per_executor: Optional[dict[str, str]] = None,
        spark_conf: Optional[dict[str, str]] = None,
        driver: Optional[Driver] = None,
        executor: Optional[Executor] = None,
        options: Optional[list] = None,
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
