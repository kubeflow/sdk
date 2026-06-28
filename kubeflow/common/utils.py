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
import os

from kubernetes import config

from kubeflow.common import constants


def is_running_in_k8s() -> bool:
    return os.path.isdir("/var/run/secrets/kubernetes.io/")


def get_default_target_namespace(context: str | None = None) -> str:
    if not is_running_in_k8s():
        try:
            all_contexts, current_context = config.list_kube_config_contexts()
            # If a context is explicitly requested, honor only that context. Fall
            # back to the default namespace rather than silently using the current
            # context's namespace when the requested context is missing or has no
            # namespace set. Use "is not None" so an explicit empty string is still
            # treated as an explicit request instead of falling through to the
            # current context.
            if context is not None:
                for c in all_contexts:
                    if isinstance(c, dict) and c.get("name") == context:
                        namespace = c.get("context", {}).get("namespace")
                        return namespace if namespace else constants.DEFAULT_NAMESPACE
                return constants.DEFAULT_NAMESPACE
            # Otherwise, try to get namespace from the current context.
            return current_context["context"]["namespace"]
        except Exception:
            return constants.DEFAULT_NAMESPACE
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        return f.readline().strip()


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
