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

"""Unit tests for kubeflow/common/telemetry.py.

Tests cover the NoOp fallback path, environment variable disabling,
span name/attribute conventions, and context manager behavior.
No network calls, no OTel backend required.
"""

import os
from unittest.mock import Mock, patch

import pytest

from kubeflow.common.telemetry import (
    SpanAttributes,
    SpanNames,
    _NoOpSpan,
    _NoOpTracer,
    get_tracer,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="returns _NoOpTracer when KUBEFLOW_TRACING_DISABLED=1",
            expected_status=SUCCESS,
            config={"env_value": "1"},
        ),
        TestCase(
            name="returns _NoOpTracer when KUBEFLOW_TRACING_DISABLED=true",
            expected_status=SUCCESS,
            config={"env_value": "true"},
        ),
        TestCase(
            name="returns _NoOpTracer when KUBEFLOW_TRACING_DISABLED=yes",
            expected_status=SUCCESS,
            config={"env_value": "yes"},
        ),
    ],
)
def test_get_tracer_returns_noop_when_disabled(test_case: TestCase) -> None:
    """get_tracer() must return _NoOpTracer when tracing is disabled via env var."""
    print(f"Executing test: {test_case.name}")
    env_value = test_case.config["env_value"]
    with patch.dict(os.environ, {"KUBEFLOW_TRACING_DISABLED": env_value}):
        tracer = get_tracer("kubeflow.test")
        assert isinstance(tracer, _NoOpTracer)
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="returns _NoOpTracer when opentelemetry-api not installed",
            expected_status=SUCCESS,
        ),
    ],
)
def test_get_tracer_returns_noop_when_otel_missing(test_case: TestCase) -> None:
    """get_tracer() must return _NoOpTracer when opentelemetry-api is unavailable."""
    print(f"Executing test: {test_case.name}")
    import kubeflow.common.telemetry as telemetry_module

    original = telemetry_module._otel_available
    try:
        telemetry_module._otel_available = False
        tracer = get_tracer("kubeflow.test")
        assert isinstance(tracer, _NoOpTracer)
    finally:
        telemetry_module._otel_available = original
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="set_attribute returns self", expected_status=SUCCESS),
        TestCase(name="add_event returns self", expected_status=SUCCESS),
        TestCase(name="record_exception returns self", expected_status=SUCCESS),
        TestCase(name="set_status returns self", expected_status=SUCCESS),
        TestCase(name="end returns None", expected_status=SUCCESS),
    ],
)
def test_noop_span_methods_are_safe(test_case: TestCase) -> None:
    """_NoOpSpan methods must be callable without raising exceptions."""
    print(f"Executing test: {test_case.name}")
    span = _NoOpSpan()
    if test_case.name == "set_attribute returns self":
        result = span.set_attribute("key", "value")
        assert result is span
    elif test_case.name == "add_event returns self":
        result = span.add_event("event", {"k": "v"})
        assert result is span
    elif test_case.name == "record_exception returns self":
        result = span.record_exception(ValueError("test"))
        assert result is span
    elif test_case.name == "set_status returns self":
        result = span.set_status("ok")
        assert result is span
    elif test_case.name == "end returns None":
        result = span.end()
        assert result is None
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="_NoOpSpan works as context manager",
            expected_status=SUCCESS,
        ),
    ],
)
def test_noop_span_context_manager(test_case: TestCase) -> None:
    """_NoOpSpan must work as a context manager without raising."""
    print(f"Executing test: {test_case.name}")
    span = _NoOpSpan()
    with span as s:
        assert s is span
        s.set_attribute("key", "value")
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="_NoOpTracer context manager yields _NoOpSpan",
            expected_status=SUCCESS,
        ),
    ],
)
def test_noop_tracer_context_manager(test_case: TestCase) -> None:
    """_NoOpTracer.start_as_current_span must yield a _NoOpSpan instance."""
    print(f"Executing test: {test_case.name}")
    tracer = _NoOpTracer()
    with tracer.start_as_current_span("test.span") as span:
        assert isinstance(span, _NoOpSpan)
        span.set_attribute("key", "value")
        span.add_event("test_event")
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="all SpanNames values start with kubeflow.sdk.",
            expected_status=SUCCESS,
        ),
        TestCase(
            name="no duplicate values in SpanNames",
            expected_status=SUCCESS,
        ),
    ],
)
def test_span_names_convention(test_case: TestCase) -> None:
    """SpanNames constants must follow kubeflow.sdk.<client>.<op> convention."""
    print(f"Executing test: {test_case.name}")
    values = [v for k, v in vars(SpanNames).items() if not k.startswith("_") and isinstance(v, str)]
    if test_case.name == "all SpanNames values start with kubeflow.sdk.":
        for val in values:
            assert val.startswith("kubeflow.sdk."), f"SpanNames.{val} must start with kubeflow.sdk."
    elif test_case.name == "no duplicate values in SpanNames":
        assert len(values) == len(set(values)), "SpanNames has duplicate values"
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="all SpanAttributes values start with kubeflow. or error.",
            expected_status=SUCCESS,
        ),
        TestCase(
            name="no duplicate values in SpanAttributes",
            expected_status=SUCCESS,
        ),
    ],
)
def test_span_attributes_convention(test_case: TestCase) -> None:
    """SpanAttributes constants must follow kubeflow.* or OTel standard naming."""
    print(f"Executing test: {test_case.name}")
    values = [
        v for k, v in vars(SpanAttributes).items() if not k.startswith("_") and isinstance(v, str)
    ]
    if test_case.name == "all SpanAttributes values start with kubeflow. or error.":
        for val in values:
            assert val.startswith("kubeflow.") or val.startswith("error."), (
                f"SpanAttributes value '{val}' must start with kubeflow. or error."
            )
    elif test_case.name == "no duplicate values in SpanAttributes":
        assert len(values) == len(set(values)), "SpanAttributes has duplicate values"
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train() emits TRAINER_TRAIN span with correct name and attributes",
            expected_status=SUCCESS,
            config={"runtime": "torch-distributed"},
        ),
    ],
)
def test_train_emits_telemetry_span(test_case) -> None:
    """Integration test: train() must emit a real OTel span captured by InMemorySpanExporter.

    Sets up a real TracerProvider backed by InMemorySpanExporter, injects it into
    KubernetesBackend following the same mock pattern as backend_test.py, calls
    train(), and asserts that the correct span name and attributes were recorded.

    Skipped automatically when opentelemetry-sdk is not installed.
    """
    print(f"Executing test: {test_case.name}")

    pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed; skipping")

    from kubeflow_trainer_api import models as api_models
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

    # Wire up an in-memory exporter so we can inspect emitted spans.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("kubeflow.trainer")

    # Minimal real TrainJobSpec so TrainerV1alpha1TrainJob.to_dict() succeeds.
    mock_spec = api_models.TrainerV1alpha1TrainJobSpec(
        runtimeRef=api_models.TrainerV1alpha1RuntimeRef(name=test_case.config["runtime"])
    )

    # Mirror the kubernetes_backend fixture from backend_test.py.
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(return_value=None),
            ),
        ),
        patch("kubernetes.client.CoreV1Api", return_value=Mock()),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.verify_backend",
            return_value=None,
        ),
    ):
        backend = KubernetesBackend(KubernetesBackendConfig())
        # Inject the real tracer so train() records real spans.
        backend._tracer = real_tracer
        # Short-circuit spec building — we are testing spans, not spec construction.
        backend._get_trainjob_spec = Mock(return_value=mock_spec)
        backend.train(runtime=test_case.config["runtime"])

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1, f"Expected at least one span, got {len(spans)}"

    train_span = next((s for s in spans if s.name == SpanNames.TRAINER_TRAIN), None)
    assert train_span is not None, (
        f"Expected span named {SpanNames.TRAINER_TRAIN!r}, got names: {[s.name for s in spans]}"
    )
    assert train_span.attributes.get(SpanAttributes.BACKEND) == "kubernetes"
    assert train_span.attributes.get(SpanAttributes.TRAINJOB_RUNTIME) == test_case.config["runtime"]
    assert SpanAttributes.TRAINJOB_NAME in train_span.attributes

    print("test execution complete")
