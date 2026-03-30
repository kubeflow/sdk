"""Tests for kubeflow.common.telemetry module."""

import os
import unittest
from unittest.mock import patch


class TestTelemetryDisabledMode(unittest.TestCase):

    def setUp(self):
        import kubeflow.common.telemetry as t
        t._tracer = None
        t._meter = None
        t._ENABLED = False

    def test_get_tracer_without_configure_returns_noop(self):
        from kubeflow.common.telemetry import get_tracer, _NoOpTracer
        tracer = get_tracer()
        self.assertIsInstance(tracer, _NoOpTracer)

    def test_noop_span_all_methods_safe(self):
        from kubeflow.common.telemetry import _NoOpSpan
        span = _NoOpSpan()
        span.set_attribute("key", "value")
        span.add_event("some_event")
        span.record_exception(RuntimeError("test"))
        span.set_status(None)

    def test_noop_tracer_context_manager(self):
        from kubeflow.common.telemetry import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span("test.span") as span:
            span.set_attribute("kubeflow.trainer.namespace", "default")

    def test_env_var_disables_telemetry(self):
        with patch.dict(os.environ, {"KUBEFLOW_TRACING_DISABLED": "1"}):
            from kubeflow.common import telemetry
            telemetry.configure(service_name="test", exporter="console")
            self.assertFalse(telemetry.is_enabled())

    def test_exporter_none_stays_disabled(self):
        from kubeflow.common import telemetry
        telemetry.configure(service_name="test", exporter="none")
        self.assertFalse(telemetry.is_enabled())

    def test_is_enabled_false_before_configure(self):
        from kubeflow.common.telemetry import is_enabled
        self.assertFalse(is_enabled())

    def test_get_meter_without_configure_returns_noop(self):
        from kubeflow.common.telemetry import get_meter, _NoOpMeter
        meter = get_meter()
        self.assertIsInstance(meter, _NoOpMeter)

    def test_noop_meter_counter(self):
        from kubeflow.common.telemetry import get_meter
        meter = get_meter()
        counter = meter.create_counter("test.counter")
        counter.add(1, {"attr": "val"})

    def test_noop_meter_histogram(self):
        from kubeflow.common.telemetry import get_meter
        meter = get_meter()
        hist = meter.create_histogram("test.histogram")
        hist.record(0.5, {"attr": "val"})
class TestSpanNamesAndAttributes(unittest.TestCase):

    def test_trainer_span_names_are_strings(self):
        from kubeflow.common.telemetry import SpanNames
        self.assertEqual(SpanNames.TRAINER_TRAIN, "kubeflow.sdk.trainer.train")
        self.assertEqual(SpanNames.TRAINER_GET_JOB, "kubeflow.sdk.trainer.get_job")
        self.assertEqual(SpanNames.TRAINER_WAIT, "kubeflow.sdk.trainer.wait_for_job_status")
        self.assertEqual(SpanNames.TRAINER_POLL_STATUS, "kubeflow.sdk.trainer.poll_status")

    def test_pipelines_span_names_are_strings(self):
        from kubeflow.common.telemetry import SpanNames
        self.assertEqual(SpanNames.PIPELINES_COMPILE, "kubeflow.sdk.pipelines.compile")
        self.assertEqual(SpanNames.PIPELINES_SUBMIT, "kubeflow.sdk.pipelines.submit")

    def test_span_attributes_follow_convention(self):
        from kubeflow.common.telemetry import SpanAttributes
        kubeflow_attrs = [
            SpanAttributes.JOB_NAME,
            SpanAttributes.NAMESPACE,
            SpanAttributes.STATUS,
        ]
        for attr in kubeflow_attrs:
            self.assertTrue(
                attr.startswith("kubeflow."),
                f"Attribute {attr} must start with 'kubeflow.'"
            )

    def test_genai_attributes_follow_otel_spec(self):
        from kubeflow.common.telemetry import SpanAttributes
        self.assertEqual(SpanAttributes.GEN_AI_OPERATION, "gen_ai.operation.name")
        self.assertEqual(SpanAttributes.GEN_AI_SYSTEM, "gen_ai.system")

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('opentelemetry') is not None,
        "opentelemetry-api not installed"
    )
    def test_configure_console_enables_telemetry(self):

        from kubeflow.common import telemetry
        telemetry.configure(service_name="test-service", exporter="console")
        self.assertTrue(telemetry.is_enabled())
        self.assertIsNotNone(telemetry._tracer)


if __name__ == "__main__":
    unittest.main()
