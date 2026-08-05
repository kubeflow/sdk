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

"""Shared OpenTelemetry instrumentation utilities for the Kubeflow Python SDK.

Design principles:
  1. Zero overhead when opentelemetry-api is not installed — NoOp tracer returned.
  2. Zero overhead when KUBEFLOW_TRACING_DISABLED=1 is set.
  3. Single source of truth for span names and attribute keys across all SDK clients.
  4. Optional dependency — opentelemetry-api must NOT be in core requirements.
  5. Context propagation handled automatically via OTel context variables.

Usage inside SDK internals:
    from kubeflow.common.telemetry import get_tracer, SpanNames, SpanAttributes

    tracer = get_tracer("kubeflow.trainer")
    with tracer.start_as_current_span(SpanNames.TRAINER_TRAIN) as span:
        span.set_attribute(SpanAttributes.NAMESPACE, namespace)

User opt-in to enable tracing:
    import kubeflow.common.telemetry as telemetry
    telemetry.configure(exporter="otlp", endpoint="http://localhost:4317")
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRACING_DISABLED_ENV = "KUBEFLOW_TRACING_DISABLED"
_otel_available: bool | None = None


def _is_otel_available() -> bool:
    """Check once if opentelemetry-api is installed. Result is cached globally.

    Returns:
        True if opentelemetry-api is importable, False otherwise.
    """
    global _otel_available
    if _otel_available is None:
        try:
            import opentelemetry  # noqa: F401

            _otel_available = True
        except ImportError:
            _otel_available = False
            logger.debug(
                "opentelemetry-api not installed. SDK tracing disabled. "
                "Enable with: pip install 'kubeflow[telemetry]'"
            )
    return _otel_available


def _tracing_disabled() -> bool:
    """Return True if KUBEFLOW_TRACING_DISABLED env var is set to a truthy value.

    Returns:
        True if tracing is explicitly disabled via environment variable.
    """
    return os.environ.get(_TRACING_DISABLED_ENV, "").lower() in ("1", "true", "yes")


def get_tracer(instrumentation_name: str = "kubeflow.sdk") -> Any:
    """Return a real OTel Tracer or a zero-overhead NoOp tracer.

    Safe to call unconditionally in SDK internals. Returns _NoOpTracer
    when opentelemetry-api is not installed, with zero allocation cost.

    Args:
        instrumentation_name: Dot-separated component name.
            Convention: kubeflow.<client> e.g. kubeflow.trainer.

    Returns:
        opentelemetry.trace.Tracer if OTel is available and tracing is
        enabled, otherwise _NoOpTracer.
    """
    if _tracing_disabled() or not _is_otel_available():
        return _NOOP_TRACER
    from opentelemetry import trace

    return trace.get_tracer(instrumentation_name)


def configure(
    exporter: str = "otlp",
    endpoint: str = "http://localhost:4317",
    service_name: str = "kubeflow-sdk",
) -> None:
    """Configure the global OpenTelemetry TracerProvider.

    Optional convenience for users who want traces without OTel boilerplate.
    SDK users who manage their own TracerProvider can skip this call entirely.

    Args:
        exporter: One of otlp, console.
            Note: Jaeger accepts OTLP natively on port 4317 — use exporter='otlp'.
        endpoint: Exporter endpoint URL.
        service_name: Service name reported in traces.

    Raises:
        ImportError: If opentelemetry-sdk is not installed.
        ValueError: If an unknown exporter name is given.
    """
    if not _is_otel_available():
        raise ImportError("opentelemetry-api required. Install: pip install 'kubeflow[telemetry]'")
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        raise ImportError(
            "opentelemetry-sdk required. Install: pip install 'kubeflow[telemetry]'"
        ) from e

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(_build_exporter(exporter, endpoint)))
    trace.set_tracer_provider(provider)
    logger.info("Kubeflow SDK tracing configured: exporter=%s endpoint=%s", exporter, endpoint)


def _build_exporter(exporter: str, endpoint: str) -> Any:
    """Build the OTel span exporter for the given backend.

    Args:
        exporter: Exporter type string.
        endpoint: Backend endpoint URL.

    Returns:
        Configured OTel SpanExporter instance.

    Raises:
        ImportError: If the required exporter package is missing.
        ValueError: If exporter name is unknown.
    """
    if exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            return OTLPSpanExporter(endpoint=endpoint)
        except ImportError as e:
            raise ImportError("Install opentelemetry-exporter-otlp for OTLP export.") from e
    if exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()
    raise ValueError(
        f"Unknown exporter '{exporter}'. Choose from: otlp, console. "
        "Note: Jaeger accepts OTLP natively on port 4317 — use exporter='otlp'."
    )


class SpanNames:
    """Canonical OTel span names for all Kubeflow SDK operations.

    Convention: kubeflow.sdk.<client>.<operation>
    All SDK instrumentation MUST use these constants, never raw strings.
    """

    TRAINER_TRAIN = "kubeflow.sdk.trainer.train"
    TRAINER_CREATE_JOB = "kubeflow.sdk.trainer.create_trainjob"
    TRAINER_GET_RUNTIME = "kubeflow.sdk.trainer.get_runtime"
    TRAINER_POLL_STATUS = "kubeflow.sdk.trainer.poll_status"
    TRAINER_LIST_JOBS = "kubeflow.sdk.trainer.list_jobs"
    TRAINER_DELETE_JOB = "kubeflow.sdk.trainer.delete_trainjob"
    PIPELINE_RUN = "kubeflow.sdk.pipeline.run"
    PIPELINE_COMPILE = "kubeflow.sdk.pipeline.compile"
    OPTIMIZER_CREATE = "kubeflow.sdk.optimizer.create_experiment"
    OPTIMIZER_MONITOR = "kubeflow.sdk.optimizer.monitor_experiment"
    HUB_REGISTER_MODEL = "kubeflow.sdk.hub.register_model"
    HUB_GET_MODEL = "kubeflow.sdk.hub.get_model"


class SpanAttributes:
    """Canonical OTel attribute keys for Kubeflow SDK spans.

    Follows OTel semantic conventions where they exist,
    extends with kubeflow.* namespace for domain-specific attributes.
    All instrumentation MUST use these constants, never raw strings.
    """

    NAMESPACE = "kubeflow.namespace"
    BACKEND = "kubeflow.backend"
    TRAINJOB_NAME = "kubeflow.trainjob.name"
    TRAINJOB_RUNTIME = "kubeflow.trainjob.runtime"
    TRAINJOB_STATUS = "kubeflow.trainjob.status"
    TRAINJOB_NUM_NODES = "kubeflow.trainjob.num_nodes"
    TRAINJOB_FRAMEWORK = "kubeflow.trainjob.framework"
    POLL_ITERATION = "kubeflow.poll.iteration"
    POLL_TIMEOUT = "kubeflow.poll.timeout_seconds"
    POLL_INTERVAL = "kubeflow.poll.interval_seconds"
    RUNTIME_NAME = "kubeflow.runtime.name"
    RUNTIME_SCOPE = "kubeflow.runtime.scope"
    ERROR_TYPE = "error.type"


class _NoOpSpan:
    """Span that does nothing. Zero allocation fallback when OTel is unavailable."""

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> _NoOpSpan:
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> _NoOpSpan:
        return self

    def record_exception(self, exception: Exception) -> _NoOpSpan:
        return self

    def set_status(self, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return self

    def is_recording(self) -> bool:
        return False

    def get_span_context(self):
        return None

    def end(self) -> None:
        pass


class _NoOpTracer:
    """Tracer that produces _NoOpSpans. Zero overhead fallback."""

    @contextlib.contextmanager  # type: ignore[misc]
    def start_as_current_span(self, name: str, **kwargs: Any):  # type: ignore[return]
        yield _NOOP_SPAN

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NOOP_SPAN


# Module-level singletons — avoids per-call allocation on the hot NoOp path.
_NOOP_SPAN = _NoOpSpan()
_NOOP_TRACER = _NoOpTracer()
