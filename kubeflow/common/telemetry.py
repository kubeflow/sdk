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

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Span, Tracer

logger = logging.getLogger(__name__)

# Instrumentation identity. The version tracks the SDK version that introduced
# telemetry support and is bumped when the instrumentation surface changes
# (new span names, attribute renames, metric additions).
_INSTRUMENTATION_NAME = "kubeflow.sdk"
_INSTRUMENTATION_VERSION = "0.5.0"

# Module-level singletons. Populated exactly once by _setup().
_tracer: Tracer | None = None
_meter: Meter | None = None
_initialized: bool = False


def _setup() -> None:
    """Lazily initialize OTel tracer and meter if the API is installed.

    This function is called on first use of :func:`get_tracer` or
    :func:`get_meter`. It attempts to import ``opentelemetry`` exactly once.
    If the import fails, ``_tracer`` and ``_meter`` remain ``None`` and all
    subsequent calls to :func:`sdk_span` become zero-cost no-ops.

    The tracer and meter are obtained via the global ``TracerProvider`` and
    ``MeterProvider``. If the user has not configured a provider (i.e. they
    installed ``opentelemetry-api`` but not ``opentelemetry-sdk``), the API
    returns built-in no-op implementations — so this module never needs its
    own stub classes.
    """
    global _tracer, _meter, _initialized  # noqa: PLW0603
    if _initialized:
        return
    _initialized = True

    try:
        from opentelemetry import metrics, trace

        _tracer = trace.get_tracer(_INSTRUMENTATION_NAME, _INSTRUMENTATION_VERSION)
        _meter = metrics.get_meter(_INSTRUMENTATION_NAME, _INSTRUMENTATION_VERSION)
    except ImportError:
        logger.debug(
            "opentelemetry-api is not installed. Install it with: pip install 'kubeflow[telemetry]'"
        )
        _tracer = None
        _meter = None


def get_tracer() -> Tracer | None:
    """Return the SDK's OTel tracer, or ``None`` if OTel is unavailable."""
    _setup()
    return _tracer


def get_meter() -> Meter | None:
    """Return the SDK's OTel meter, or ``None`` if OTel is unavailable."""
    _setup()
    return _meter


@contextmanager
def sdk_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Span | None]:
    """Context manager that creates an OTel span if available, or no-ops.

    When ``opentelemetry-api`` is installed, this creates a span via
    ``tracer.start_as_current_span`` — meaning it is automatically set as
    the current span in the OTel context, and any child spans created
    inside the ``with`` block will be parented to it.

    When OTel is not installed, this yields ``None`` with no overhead.
    Callers must guard attribute-setting calls with ``if span:``::

        with sdk_span("kubeflow.trainer.train") as span:
            result = do_work()
            if span:
                span.set_attribute("kubeflow.trainjob.name", result.name)

    Args:
        name: The span name. Should follow the ``kubeflow.<client>.<operation>``
            convention (e.g. ``kubeflow.trainer.train``).
        attributes: Initial span attributes. Keys should use the ``kubeflow.*``
            namespace prefix.

    Yields:
        The active ``Span`` instance, or ``None`` if OTel is not available.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span
