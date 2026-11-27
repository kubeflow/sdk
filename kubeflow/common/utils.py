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
import ast
from collections.abc import Callable
import inspect
import os
import textwrap

from kubernetes import config

from kubeflow.common import constants
from kubeflow.common.types import KubernetesBackendConfig


def is_running_in_k8s() -> bool:
    return os.path.isdir("/var/run/secrets/kubernetes.io/")


def load_kube_config(cfg: KubernetesBackendConfig) -> None:
    """Load the Kubernetes client configuration for a backend.

    Trainer, Optimizer and Spark share this so that a given KubernetesBackendConfig
    resolves to the same cluster in every client.

    Precedence:
        1. ``client_configuration`` is set: nothing is loaded, the caller builds its
           API client from it.
        2. ``config_file`` is set, or the SDK runs outside a pod: kube-config is loaded
           with ``config_file`` and ``context``.
        3. Otherwise in-cluster config is loaded. If that fails, kube-config is tried
           so a pod with a mounted kube-config still works.

    Args:
        cfg: Kubernetes backend configuration.

    Raises:
        kubernetes.config.ConfigException: If no configuration can be loaded.
    """
    if cfg.client_configuration is not None:
        return

    if cfg.config_file or not is_running_in_k8s():
        config.load_kube_config(config_file=cfg.config_file, context=cfg.context)
        return

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config(context=cfg.context)


def get_default_target_namespace(context: str | None = None) -> str:
    if not is_running_in_k8s():
        try:
            all_contexts, current_context = config.list_kube_config_contexts()
            # If context is set, we should get namespace from it.
            if context:
                for c in all_contexts:
                    if isinstance(c, dict) and c.get("name") == context:
                        return c["context"]["namespace"]
            # Otherwise, try to get namespace from the current context.
            return current_context["context"]["namespace"]
        except Exception:
            return constants.DEFAULT_NAMESPACE
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        return f.readline()


def validate_wait_for_job_status(polling_interval: int, timeout: int) -> None:
    """Validate polling_interval and timeout values used by wait_for_job_status methods.

    Args:
        polling_interval: The polling interval in seconds.
        timeout: The timeout in seconds.

    Raises:
        ValueError: If polling_interval or timeout are not positive, or if polling_interval
            is not strictly less than timeout.
    """
    if timeout <= 0:
        raise ValueError(f"Timeout must be a positive number, got timeout={timeout}")
    if polling_interval <= 0:
        raise ValueError(
            f"Polling interval must be a positive number, got polling_interval={polling_interval}"
        )
    if polling_interval >= timeout:
        raise ValueError(
            "Polling interval must be strictly less than timeout. "
            f"Received polling_interval={polling_interval}, timeout={timeout}"
        )


def validate_python_function(func: Callable) -> None:
    """Validate a Python function.

    Args:
        func: The Python function to validate.

    Raises:
        ValueError: If the function is not a Python function, is asynchronous,
            is a lambda function, is decorated, or its source cannot be
            inspected or parsed.
    """
    if not inspect.isfunction(func):
        raise ValueError("Function must be a Python function.")

    if inspect.iscoroutinefunction(func):
        raise ValueError("Async functions are not supported.")

    if func.__name__ == "<lambda>":
        raise ValueError("Lambda functions are not supported.")

    try:
        func_source = textwrap.dedent(inspect.getsource(func))
    except TypeError as e:
        raise ValueError(
            "Function must be a pure-Python function; built-in or "
            "C-implemented callables are not supported."
        ) from e
    except OSError as e:
        raise ValueError(
            "Function source could not be read. Functions defined interactively are not supported."
        ) from e

    try:
        func_tree = ast.parse(func_source)
    except SyntaxError as e:
        raise ValueError("Function source could not be parsed.") from e

    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list
        for node in func_tree.body
    ):
        raise ValueError("Decorated functions are not supported.")
