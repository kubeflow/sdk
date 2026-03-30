import os
import functools
from contextlib import contextmanager
from typing import Optional

_tracer = None
_meter = None
_ENABLED = False


def configure(
    service_name: str = "kubeflow-sdk",
    exporter: str = "console",
    endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
) -> None:
    global _tracer, _meter, _ENABLED

    if os.environ.get("KUBEFLOW_TRACING_DISABLED", "0") == "1":
        return

    if exporter == "none":
        return

    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        sampler = TraceIdRatioBased(sample_rate)

        tracer_provider = TracerProvider(resource=resource, sampler=sampler)

        if exporter == "console":
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
        elif exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_endpoint = endpoint or os.environ.get(
                "KUBEFLOW_OTLP_ENDPOINT", "http://localhost:4317"
            )
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )

        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer("kubeflow.sdk", "0.1.0")

        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=60000
        )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter("kubeflow.sdk", "0.1.0")

        _ENABLED = True

    except ImportError:
        pass

def get_tracer():
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        return trace.get_tracer("kubeflow.sdk")
    except ImportError:
        return _NoOpTracer()


def get_meter():
    if _meter is not None:
        return _meter
    try:
        from opentelemetry import metrics
        return metrics.get_meter("kubeflow.sdk")
    except ImportError:
        return _NoOpMeter()


def is_enabled() -> bool:
    return _ENABLED

class SpanNames:
    TRAINER_TRAIN         = "kubeflow.sdk.trainer.train"
    TRAINER_GET_JOB       = "kubeflow.sdk.trainer.get_job"
    TRAINER_GET_LOGS      = "kubeflow.sdk.trainer.get_job_logs"
    TRAINER_WAIT          = "kubeflow.sdk.trainer.wait_for_job_status"
    TRAINER_CREATE_JOB    = "kubeflow.sdk.trainer.create_trainjob"
    TRAINER_POLL_STATUS   = "kubeflow.sdk.trainer.poll_status"
    PIPELINES_COMPILE     = "kubeflow.sdk.pipelines.compile"
    PIPELINES_SUBMIT      = "kubeflow.sdk.pipelines.submit"
    OPTIMIZER_OPTIMIZE    = "kubeflow.sdk.optimizer.optimize"
    SPARK_SUBMIT          = "kubeflow.sdk.spark.submit"

class SpanAttributes:
    JOB_NAME              = "kubeflow.trainer.job_name"
    JOB_ID                = "kubeflow.trainer.job_id"
    NAMESPACE             = "kubeflow.trainer.namespace"
    NUM_NODES             = "kubeflow.trainer.num_nodes"
    RUNTIME               = "kubeflow.trainer.runtime"
    STATUS                = "kubeflow.trainer.status"
    POLL_ITERATION        = "kubeflow.trainer.poll.iteration"
    PIPELINE_NAME         = "kubeflow.pipelines.pipeline_name"
    PIPELINE_PATH         = "kubeflow.pipelines.pipeline_path"
    RUN_ID                = "kubeflow.pipelines.run_id"
    EXPERIMENT_NAME       = "kubeflow.pipelines.experiment_name"
    GEN_AI_OPERATION      = "gen_ai.operation.name"
    GEN_AI_SYSTEM         = "gen_ai.system"
    GEN_AI_MODEL          = "gen_ai.request.model"


class _NoOpSpan:

    def set_attribute(self, key, value): pass
    def add_event(self, name, attributes=None): pass
    def record_exception(self, exception, attributes=None): pass
    def set_status(self, status, description=None): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class _NoOpTracer:

    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        yield _NoOpSpan()


class _NoOpMeter:

    def create_counter(self, *a, **kw): return _NoOpInstrument()
    def create_histogram(self, *a, **kw): return _NoOpInstrument()


class _NoOpInstrument:

    def add(self, *a, **kw): pass
    def record(self, *a, **kw): pass


