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
from collections.abc import Iterator
import multiprocessing
import os
import random
import string
import uuid

from kubernetes import client, config, watch

from kubeflow.common import constants, types
from kubeflow.common.types import KubernetesBackendConfig


def is_running_in_k8s() -> bool:
    return os.path.isdir("/var/run/secrets/kubernetes.io/")


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


def load_kube_config(cfg: KubernetesBackendConfig) -> None:
    """Load Kubernetes configuration based on the provided backend config."""
    if cfg.namespace is None:
        cfg.namespace = get_default_target_namespace(cfg.context)

    if cfg.client_configuration is None:
        if cfg.config_file or not is_running_in_k8s():
            config.load_kube_config(config_file=cfg.config_file, context=cfg.context)
        else:
            config.load_incluster_config()


def generate_random_name(prefix: str = "", length: int = 11) -> str:
    """Generate a random name with an optional prefix.

    Note: when a prefix is given the total length is ``len(prefix) + 1 + length``
    (prefix, hyphen, random hex); without a prefix the total length is ``length + 1``
    (one leading lowercase letter followed by ``length`` hex characters).
    """
    if prefix:
        return f"{prefix}-{uuid.uuid4().hex[:length]}"
    return random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:length]


def read_pod_logs(
    core_api: client.CoreV1Api,
    pod_name: str,
    namespace: str,
    container_name: str | None = None,
    follow: bool = False,
    timeout: int = constants.DEFAULT_TIMEOUT,
) -> Iterator[str]:
    """Read logs from a Kubernetes pod.

    Args:
        core_api: Kubernetes CoreV1Api client.
        pod_name: Name of the pod.
        namespace: Kubernetes namespace.
        container_name: Name of the container.
        follow: Whether to stream logs continuously.
        timeout: Timeout in seconds for the API call.

    Yields:
        Log lines from the pod.

    Raises:
        TimeoutError: If the API call times out.
        RuntimeError: If pod logs cannot be retrieved.
    """
    try:
        kwargs: dict = {
            "name": pod_name,
            "namespace": namespace,
        }
        if container_name:
            kwargs["container"] = container_name

        if follow:
            # watch.Watch().stream() returns a plain generator — it is a synchronous
            # blocking HTTP stream and has no .get() method. Do NOT pass async_req here.
            log_stream = watch.Watch().stream(
                core_api.read_namespaced_pod_log, follow=True, **kwargs
            )
            yield from log_stream
        else:
            # async_req=True makes the k8s client return an ApplyResult whose
            # .get(timeout) blocks until the response arrives or raises
            # multiprocessing.TimeoutError — the same pattern as get_job_events.
            logs = core_api.read_namespaced_pod_log(**kwargs, async_req=True).get(timeout)
            if logs:
                yield from logs.splitlines()

    except multiprocessing.TimeoutError as e:
        raise TimeoutError(f"Timeout while reading logs for the pod {namespace}/{pod_name}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read logs for the pod {namespace}/{pod_name}") from e


def get_job_events(
    core_api: client.CoreV1Api,
    namespace: str,
    job_resources: set[str],
    job_kinds: set[str],
    timeout: int = constants.DEFAULT_TIMEOUT,
) -> list[types.Event]:
    """Retrieve Kubernetes events related to a specific job and its resources.

    Args:
        core_api: Kubernetes CoreV1Api client.
        namespace: The namespace to get the events from.
        job_resources: A set of resource names (e.g., job name, pod names, trial names).
        job_kinds: A set of Kubernetes resource kinds (e.g., {"TrainJob", "Pod"}).
        timeout: Timeout in seconds for the API call.

    Returns:
        List of Kubernetes events related to the given resources, sorted by time.

    Raises:
        TimeoutError: If the API call times out.
    """
    events = []
    try:
        # Retrieve events from the namespace
        event_response: client.V1EventList = core_api.list_namespaced_event(
            namespace=namespace,
            async_req=True,
        ).get(timeout)

        # Filter events related to the given resources
        for event in event_response.items:
            if not (event.metadata and event.involved_object and event.first_timestamp):
                continue

            involved_object = event.involved_object

            # Check if event is related to the job resources
            if involved_object.kind in job_kinds and involved_object.name in job_resources:
                events.append(
                    types.Event(
                        involved_object_kind=involved_object.kind,
                        involved_object_name=involved_object.name,
                        message=event.message or "",
                        reason=event.reason or "",
                        event_time=event.first_timestamp,
                    )
                )

        # Sort events by first occurrence time
        events.sort(key=lambda e: e.event_time)

        return events
    except multiprocessing.TimeoutError as e:
        raise TimeoutError("Timeout while retrieving job events") from e
