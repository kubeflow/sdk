# KEP-164: Integrate Kubeflow SDK with OpenTelemetry

## Authors

- Yash Agarwal - [@XploY04](https://github.com/XploY04)

Ref: https://github.com/kubeflow/sdk/issues/164

## Summary

This KEP adds native OpenTelemetry instrumentation to the Kubeflow SDK so users can get
distributed traces, metrics, and correlated logs from SDK operations like training job
submission, status polling, and lifecycle management. The SDK depends on `opentelemetry-api`
for no-op-by-default instrumentation and `opentelemetry-semantic-conventions` for standard
attribute constants. Users who want telemetry export install `opentelemetry-sdk` and an
exporter separately.

## Motivation

When a training job fails or takes too long, users have no way to see what the SDK
actually did. Which Kubernetes API calls were made? How long did each take? Where did it
break? Right now, the only observability is `logging.getLogger()` with no correlation to
job lifecycle.

This was raised as user feedback at KubeCon + CloudNativeCon 2025 NA.

OpenTelemetry is the CNCF standard for observability. Adding it to the SDK enables:

- Tracing SDK operations and correlating them with training pod activity
- Collecting operational metrics (job counts, durations, error rates) for monitoring dashboards
- Exporting telemetry to any OTel-compatible backend (Jaeger, Grafana Tempo, Datadog, etc.)
- Running with zero performance cost when telemetry is not configured (no-op by default)

### Goals

- Instrument `TrainerClient` and all three backends (Kubernetes, Container, LocalProcess)
  with OTel traces.
- Build a shared telemetry module (`kubeflow/common/telemetry/`) that all SDK clients can
  reuse.
- Propagate trace context across pod boundaries via W3C `TRACEPARENT` env var injection.
- Collect operational metrics: job counts, operation durations, active jobs, errors.
- Emit spans and metrics through the standard OTel API so user applications can configure
  their own providers, exporters, resources, and sampling.
- Provide an optional `configure_telemetry()` quickstart helper for examples and simple
  scripts that do not need full OTel setup control.
- Integrate OTel's `LoggingHandler` so existing Python `logging` calls are automatically
  correlated with active trace context (trace ID and span ID appear in log records).
- Apply OTel GenAI semantic conventions where they map to training workflows
  (model references in initializers, training job lifecycle spans).

### Stretch Goals

- Instrument `OptimizerClient` (Katib) with the same two-level tracing pattern.
- Instrument `ModelRegistryClient` with single-level tracing (REST API wrapper, no
  backend layer).
- Instrument `SparkClient` with two-level tracing.
- Align with `PipelinesClient` instrumentation in the Pipelines repository once
  `PipelinesClient` lives there.

### Non-Goals

- Building a custom observability backend or UI.
- Collecting training-specific metrics (loss, accuracy, learning rate). That belongs in
  experiment tracking tools like MLflow or Weights & Biases.
- Auto-instrumenting user training functions inside pods. Users opt into pod-side tracing
  by reading `TRACEPARENT` from their environment.
- Adding `opentelemetry-sdk` or any exporter as a hard dependency.
- Instrumenting third-party libraries. That is the responsibility of those projects or
  separate instrumentation packages.

## Proposal

The SDK calls the OTel API directly from its own code, following the OTel recommendation
for [native library instrumentation](https://opentelemetry.io/docs/concepts/instrumentation/libraries/).

The core implementation covers `TrainerClient` (all three backends: Kubernetes, Container,
LocalProcess). A shared telemetry module in
`kubeflow/common/telemetry/` provides helpers for tracer/meter access, attribute constants,
context propagation, and an optional `configure_telemetry()` quickstart function.

User-owned OTel configuration is the primary path. If a user application configures its
own `TracerProvider`, `MeterProvider`, exporters, resources, or sampling, the SDK emits
signals through the OTel API and respects that setup. The `configure_telemetry()` helper
is optional convenience for quickstarts and simple scripts, not a required Kubeflow-specific
configuration layer.

Every traced operation produces two spans: one at the client level (`INTERNAL`) and one at
the backend level (`CLIENT`). This keeps traces readable while preserving debugging detail.
When telemetry is configured, the overhead per operation is one span allocation and a few
attribute writes, negligible for training workflows that run for seconds to hours.
`OptimizerClient`, `ModelRegistryClient`, and `SparkClient` are stretch deliverables that
reuse the same module.

### User Experience

**Installation:**

```bash
# SDK works as before; tracing is no-op
pip install kubeflow

# User adds OTel SDK + exporter when they want observability
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

**Usage with `configure_telemetry()` helper (quickstart):**

This helper is optional. It exists for examples and small scripts where users want to see
their first trace without writing OTel provider/exporter boilerplate.

```python
from kubeflow.common.telemetry import configure_telemetry
from kubeflow.trainer import TrainerClient, CustomTrainer

# One-line setup: configures TracerProvider, MeterProvider, and OTLP exporter
configure_telemetry(exporter="otlp", endpoint="http://localhost:4317")

client = TrainerClient()
job_name = client.train(
    trainer=CustomTrainer(func=my_train_fn, num_nodes=2),
    runtime="torch-distributed",
)
job = client.wait_for_job_status(job_name, {"Complete", "Failed"})
```

**Usage with manual OTel configuration (full control):**

Users who prefer full control can configure OTel directly using `TracerProvider` /
`MeterProvider` in code, or via standard OTel env vars (`OTEL_TRACES_EXPORTER`,
`OTEL_METRICS_EXPORTER`, `OTEL_SERVICE_NAME`). Resource attributes (`service.name`,
`service.version`) are the user's responsibility when using manual configuration.

```python
# --- User configures OTel directly ---
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry import trace

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# --- Use Kubeflow SDK exactly as before, traces are emitted automatically ---
from kubeflow.trainer import TrainerClient, CustomTrainer

client = TrainerClient()
job_name = client.train(
    trainer=CustomTrainer(func=my_train_fn, num_nodes=2),
    runtime="torch-distributed",
)
job = client.wait_for_job_status(job_name, {"Complete", "Failed"})
```

**What appears in Jaeger/Grafana Tempo:**

```
TrainerClient.train                                    [1.5s]  INTERNAL
| kubeflow.trainer.type:  CustomTrainer
| kubeflow.runtime.name:  torch-distributed
| kubeflow.job.name:      a3f8b2c1d9e0f
|
+-- KubernetesBackend.train                            [1.2s]  CLIENT
    | kubeflow.backend.type:   kubernetes
    | k8s.namespace.name:      default
    | kubeflow.job.name:       a3f8b2c1d9e0f
    +-- status: OK

TrainerClient.wait_for_job_status                      [45.0s] INTERNAL
| kubeflow.job.name:     a3f8b2c1d9e0f
| kubeflow.job.status:   Complete
| Events:
|   [0.0s]   status_change: Created
|   [2.1s]   status_change: Running
|   [45.0s]  status_change: Complete
|
+-- KubernetesBackend.wait_for_job_status              [44.8s] CLIENT
    +-- status: OK
```

**What an error trace looks like:**

```
TrainerClient.train                                    [0.3s]  INTERNAL  ERROR
| kubeflow.trainer.type:  CustomTrainer
| error.type:             kubernetes.client.ApiException
|
+-- KubernetesBackend.train                            [0.2s]  CLIENT   ERROR
    | kubeflow.backend.type:   kubernetes
    | Exception: ApiException(403): Forbidden
    +-- status: ERROR
```

### User Stories

#### Story 1: Debugging a Slow Training Job Submission

As an ML engineer, I call `client.train()` and it takes 10 seconds to return. With OTel
traces, I can see that `KubernetesBackend.train` spent 9.5 seconds on
`create_namespaced_custom_object`. The Kubernetes API server was slow, not the SDK.

#### Story 2: Monitoring Job Error Rates

As a platform operator, I configure an OTel Collector to export metrics to Prometheus. I
build a Grafana dashboard showing `kubeflow.trainer.errors` by `error.type` and
`kubeflow.operation.name`. When `TimeoutError` spikes for `get_job`, I know the API server
is under pressure.

#### Story 3: End-to-End Trace Across Pod Boundary

As an ML engineer, I configure OTel in both my SDK script and my training function. The
SDK injects `TRACEPARENT` into the training pod's environment. My training code reads it
and starts a child span. In Jaeger, I see one connected trace from `client.train()` all
the way through to `training.epoch` inside the pod.

#### Story 4: Zero Impact Without Configuration

As an ML engineer, I install `kubeflow` without any OTel packages. My existing workflows
are unaffected: no errors, no performance overhead, no configuration needed. The SDK
uses no-op implementations by default.

## Design Details

### Architecture

```
kubeflow/
+-- common/
|   +-- telemetry/                    # NEW shared OTel infrastructure
|   |   +-- __init__.py               # configure_telemetry() and module-level tracer/meter setup
|   |   +-- attributes.py             # Kubeflow-specific attribute constants
|   |   +-- configure.py              # Exporter/sampler setup helper
|   |   +-- propagation.py            # TRACEPARENT injection for pods/containers
|   +-- constants.py
|   +-- types.py
|   +-- utils.py
|
+-- trainer/
|   +-- api/
|   |   +-- trainer_client.py         # MODIFIED add spans to public methods
|   +-- backends/
|   |   +-- kubernetes/backend.py     # MODIFIED add spans to backend methods
|   |   +-- container/backend.py      # MODIFIED add spans to backend methods
|   |   +-- localprocess/backend.py   # MODIFIED add spans to backend methods
|   +-- ...
|
+-- optimizer/                         # Stretch: uses same telemetry module
+-- hub/                               # Stretch: uses same telemetry module
+-- spark/                             # Stretch: uses same telemetry module
```

### Dependency Changes

```toml
# pyproject.toml
dependencies = [
    "kubernetes>=27.2.0",
    "pydantic>=2.10.0",
    "kubeflow-trainer-api>=2.0.0",
    "kubeflow-katib-api>=0.19.0",
    "opentelemetry-api>=1.4,<2.0",                      # NEW
    "opentelemetry-semantic-conventions>=0.50b0",        # NEW
]

[dependency-groups]
dev = [
    # ... existing deps ...
    "opentelemetry-sdk>=1.4",                            # NEW for InMemorySpanExporter in tests
]
```

**`opentelemetry-api` as a core dependency.** The
[OTel native instrumentation guidelines](https://opentelemetry.io/docs/concepts/instrumentation/libraries/)
recommend libraries use `opentelemetry-api` for native instrumentation without requiring
the OTel SDK or exporters. The package is under 100KB compressed, with `typing-extensions`
and `importlib-metadata` as transitive dependencies, and is no-op by default. There is no
runtime cost when users don't configure telemetry.

**`opentelemetry-semantic-conventions`.** Provides Python constants for standard attribute
names (e.g., `K8S_NAMESPACE_NAME` instead of the raw string `"k8s.namespace.name"`).
Prevents silent typos and provides IDE autocomplete.

### Instrumentation Style

All instrumentation uses the **context manager** pattern. Exceptions are recorded on spans
using `record_exception()` with `set_status(ERROR)`, following OTel conventions:

```python
from opentelemetry import trace

tracer = trace.get_tracer("kubeflow.trainer")

class TrainerClient:
    def train(self, trainer, runtime, ...) -> str:
        with tracer.start_as_current_span("TrainerClient.train") as span:
            span.set_attribute(KUBEFLOW_TRAINER_TYPE, type(trainer).__name__)
            span.set_attribute(KUBEFLOW_RUNTIME_NAME, runtime_name)

            try:
                job_name = self.backend.train(...)
                span.set_attribute(KUBEFLOW_JOB_NAME, job_name)
                return job_name
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise
```

### Span Granularity: Two Levels (Client + Backend)

Every traced operation produces two spans:

```
ClientMethod       [INTERNAL]    <-- what the user called
  +-- BackendMethod  [CLIENT]    <-- what actually executed
```

Sub-operations within a backend (resolving runtime, building spec, making the K8s API
call) are recorded as span events, not child spans.

Exception: `LocalProcessBackend` spans use `INTERNAL` kind (not `CLIENT`) because
subprocess spawning is a local operation, not a remote call.

### Span Naming Convention

Spans are named `Class.method`, following the pattern used by the
[AWS SDK OTel conventions](https://opentelemetry.io/docs/specs/semconv/cloud-providers/aws-sdk/)
(`Service.Operation`):

```
TrainerClient.train
TrainerClient.get_job
TrainerClient.wait_for_job_status
KubernetesBackend.train
ContainerBackend.train
LocalProcessBackend.train
```

> **Open question:** The OTel community has multiple span naming patterns (`Class.method`,
> `{verb} {object}`, `{package.service}/{method}`). We chose `Class.method` for clarity
> and debuggability. We welcome feedback from maintainers on whether a different convention
> is preferred.

### Attribute Naming

#### Standard OTel Conventions (reused as-is)

| Attribute | Source | Used When |
|---|---|---|
| `k8s.namespace.name` | [K8s semconv](https://opentelemetry.io/docs/specs/semconv/resource/k8s/) | Kubernetes backend operations |
| `error.type` | General semconv | On error spans |

We do not use `k8s.job.name` for TrainJob names. That OTel attribute is defined for
Kubernetes batch/v1 Jobs, and a TrainJob is a CRD not a batch Job. TrainJob names go in
the Kubeflow-specific `kubeflow.job.name` attribute instead.

#### GenAI Semantic Conventions

The [OTel GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) defines
attributes and span conventions for AI/ML workloads. Most of these target inference
(token counts, prompts, completions) and don't apply to training orchestration. The
conventions that fit:

| GenAI Convention | Where it applies in Kubeflow |
|---|---|
| `gen_ai.request.model` | When a model initializer references a model by name (e.g., `hf://meta-llama/Llama-3` in `HuggingFaceModelInitializer`) |
| `gen_ai.system` | Identifies the framework: `"huggingface"`, `"pytorch"`, etc. based on the trainer/runtime |
| `gen_ai.operation.name` | Maps to training lifecycle operations: `"train"`, `"fine-tune"` |

Inference-specific attributes (token counts, prompt/completion content, sampling parameters)
are left unused. As the GenAI semconv matures and adds training-specific attributes, we can
adopt them incrementally.

#### Kubeflow-Specific Attributes

All Kubeflow-specific attributes used by TrainerClient and its backends. Potential
PipelinesClient attributes are listed later only as cross-repo compatibility guidance.

| Attribute | Description | Example Values |
|---|---|---|
| `kubeflow.job.name` | Training job identifier | `a3f8b2c1d9e0f` |
| `kubeflow.job.status` | Current job status | `Running`, `Complete`, `Failed` |
| `kubeflow.backend.type` | Execution backend | `kubernetes`, `container`, `local-process` |
| `kubeflow.container.runtime` | Container runtime | `docker`, `podman` |
| `kubeflow.trainer.type` | Trainer class | `CustomTrainer`, `BuiltinTrainer` |
| `kubeflow.runtime.name` | Training runtime name | `torch-distributed` |
| `kubeflow.num_nodes` | Number of training nodes | `2`, `4` |
| `kubeflow.initializer.type` | Initializer class | `HuggingFaceModelInitializer` |
| `kubeflow.initializer.uri` | Model/dataset URI | `hf://meta-llama/Llama-3` |
| `kubeflow.operation.name` | SDK operation (for metrics) | `train`, `get_job` |

### Context Propagation

The SDK injects W3C Trace Context into training pods, containers, and subprocesses using
the `TRACEPARENT` environment variable. This follows the
[OTel Environment Carriers spec](https://opentelemetry.io/docs/specs/otel/context/env-carriers/).
The propagator injects both `TRACEPARENT` and `TRACESTATE` (if present). `TRACESTATE`
carries vendor-specific trace data and is part of the W3C Trace Context standard.

```python
# kubeflow/common/telemetry/propagation.py

from opentelemetry import context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

def inject_trace_context() -> dict[str, str]:
    """Extract current trace context as environment variables."""
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier, context.get_current())
    return {k.upper(): v for k, v in carrier.items()}
```

Injection points:

| Backend | Where TRACEPARENT is injected |
|---|---|
| Kubernetes | Pod template env vars in TrainJob CR spec |
| Container | Docker/Podman container environment |
| LocalProcess | Subprocess `env` dict |

User-side extraction (optional, written by the user in their training code):

```python
# Inside the training pod; user writes this if they want end-to-end tracing
import os
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry import trace

carrier = {"traceparent": os.environ.get("TRACEPARENT", "")}
parent_ctx = TraceContextTextMapPropagator().extract(carrier)

tracer = trace.get_tracer("my.training")
with tracer.start_as_current_span("training.run", context=parent_ctx):
    # Training code now part of the same trace
    ...
```

If the training code does not use OTel, the `TRACEPARENT` env var is harmlessly ignored.

### Metrics

| Metric Name | Type | Unit | Description |
|---|---|---|---|
| `kubeflow.trainer.job.created` | Counter | `{job}` | Total training jobs submitted |
| `kubeflow.trainer.operation.duration` | Histogram | `s` | Duration of each SDK API call |
| `kubeflow.trainer.job.active` | UpDownCounter | `{job}` | Currently running jobs |
| `kubeflow.trainer.errors` | Counter | `{error}` | Total errors by type and operation |

Attribute dimensions (low cardinality only):

| Metric | Dimensions |
|---|---|
| `job.created` | `kubeflow.backend.type`, `kubeflow.trainer.type` |
| `operation.duration` | `kubeflow.operation.name`, `kubeflow.backend.type`, `error.type` |
| `job.active` | `kubeflow.backend.type` |
| `errors` | `kubeflow.operation.name`, `error.type`, `kubeflow.backend.type` |

To prevent cardinality explosion, per-job identifiers (`kubeflow.job.name`,
`kubeflow.runtime.name`) are recorded on spans only, not as metric dimensions.

### Log Correlation

The SDK does not replace or wrap existing `logging.getLogger()` calls. Instead, OTel's
`LoggingHandler` automatically injects trace context (trace ID, span ID) into log records
when a `TracerProvider` is configured. Existing SDK logs like:

```
INFO:kubeflow.trainer:Creating TrainJob a3f8b2c1d9e0f in namespace default
```

become correlated with the active span:

```
INFO:kubeflow.trainer:Creating TrainJob a3f8b2c1d9e0f in namespace default
  [trace_id=4bf92f3577b01e8d span_id=00f067aa0ba902b7]
```

This lets users query logs by trace ID in tools like Grafana Loki and see every log line
that happened during a specific `train()` call. Users configure OTel's `LoggingHandler`
on their side. The `configure_telemetry()` helper can optionally set up the logging bridge
as well.

### Telemetry Module API

```python
# kubeflow/common/telemetry/__init__.py

from opentelemetry import trace, metrics

def configure_telemetry(
    exporter: str = "otlp",
    endpoint: str | None = None,
    sampling_rate: float = 1.0,
    service_name: str = "kubeflow-sdk",
) -> None:
    """Set up OTel TracerProvider and MeterProvider with sensible defaults.

    Requires `opentelemetry-sdk` and the exporter package to be installed.
    Raises ImportError with install instructions if they are missing.

    Args:
        exporter: Exporter type. "otlp", "console", or "none".
        endpoint: Collector endpoint. Defaults to OTEL_EXPORTER_OTLP_ENDPOINT or localhost:4317.
        sampling_rate: Fraction of traces to sample, 0.0 to 1.0.
        service_name: Value for the service.name resource attribute.
    """
    ...
```

### PipelinesClient Alignment

`PipelinesClient` is expected to live in the Pipelines repository
([kubeflow/pipelines#13405](https://github.com/kubeflow/pipelines/pull/13405)). This KEP
therefore does not include PipelinesClient file changes in the SDK repository. Pipelines
instrumentation should be implemented in the Pipelines repo, using compatible span names
and attributes where it makes sense.

Unlike `TrainerClient`, PipelinesClient does not have the same SDK backend layer. A future
Pipelines KEP or PR can use single-level tracing (`INTERNAL` spans only).

A compatible trace shape could look like:

```
PipelinesClient.create_pipeline              [0.8s]  INTERNAL
| kubeflow.pipeline.name: my-training-pipeline
+-- status: OK

PipelinesClient.create_run                   [1.2s]  INTERNAL
| kubeflow.pipeline.name: my-training-pipeline
| kubeflow.experiment.name: llama-finetune
| kubeflow.run.id: run-abc123
+-- status: OK

PipelinesClient.wait_for_run                 [120s]  INTERNAL
| kubeflow.run.id: run-abc123
| kubeflow.run.status: Succeeded
| Events:
|   [0.0s]   status_change: Pending
|   [5.2s]   status_change: Running
|   [120s]   status_change: Succeeded
+-- status: OK
```

Potential PipelinesClient-specific attributes:

| Attribute | Description |
|---|---|
| `kubeflow.pipeline.name` | Pipeline name |
| `kubeflow.pipeline.version_id` | Pipeline version identifier |
| `kubeflow.experiment.name` | Experiment name |
| `kubeflow.run.id` | Run identifier |
| `kubeflow.run.status` | Run status |

### Stretch Deliverables: OptimizerClient, ModelRegistryClient, SparkClient

These clients follow the same pattern established by `TrainerClient` and reuse the
same `kubeflow/common/telemetry/` module:

| Client | Architecture | Tracing Pattern | Complexity |
|---|---|---|---|
| `OptimizerClient` | Kubernetes backend only | Two-level: INTERNAL / CLIENT | Medium (same pattern as TrainerClient with fewer methods) |
| `ModelRegistryClient` | REST API wrapper (delegates to `model_registry.ModelRegistry`) | Single-level: INTERNAL only | Low (no backend layer) |
| `SparkClient` | Kubernetes backend only | Two-level: INTERNAL / CLIENT | Medium (same pattern as TrainerClient) |

The shared telemetry module means instrumenting each additional client is mostly
mechanical: wrap public methods with spans, set the right attributes, add unit tests.

### Notes/Constraints/Caveats

- The OTel Environment Carriers spec (used for `TRACEPARENT` injection) is currently at
  Alpha status in the OTel specification. The injection mechanism may need to be updated
  as the spec matures.
- The GenAI semantic conventions are still evolving. Training-specific attributes may be
  added upstream, which we can adopt incrementally.
- `configure_telemetry()` requires `opentelemetry-sdk` (not a core dependency). Users who
  call it without the SDK installed get a clear `ImportError`.
- OTel context propagation uses Python's `contextvars`, which is thread-safe and compatible
  with both threaded and asyncio-based usage. Concurrent SDK operations (e.g., submitting
  multiple jobs from separate threads) each maintain independent trace context.

### Risks and Mitigations

- **Risk**: OTel API breaking changes between major versions.
  - **Mitigation**: Pin `opentelemetry-api>=1.4,<2.0` to avoid unexpected breakage. Monitor
    the OTel Python SIG for v2 migration guidance.
- **Risk**: Span noise if users don't configure sampling for high-throughput workloads.
  - **Mitigation**: Document sampling configuration in getting-started guide.
    `configure_telemetry()` accepts a `sampling_rate` parameter.
- **Risk**: `TRACEPARENT` env var injection relies on an Alpha-status OTel spec.
  - **Mitigation**: The injection is a simple env var. Even if the spec changes, the
    mechanism is trivial to update. The env var is harmlessly ignored if not consumed.

### Test Plan

`opentelemetry-sdk` is added to the `dev` dependency group for `InMemorySpanExporter`.

A shared test helper goes in `kubeflow/common/telemetry/test_utils.py`:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, InMemorySpanExporter


def setup_test_telemetry() -> InMemorySpanExporter:
    """Set up an in-memory span exporter for test assertions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter
```

Test categories:

| Category | What is tested | Pattern |
|---|---|---|
| Span creation | Correct span names, parent-child relationships | `InMemorySpanExporter` assertions |
| Span attributes | Correct attribute keys and values per method | `OtelTestCase` dataclass with `expected_attributes` |
| Error recording | `record_exception` called, span status is ERROR | Sentinel-based error triggers (existing pattern) |
| No-op behavior | SDK works normally with `NoOpTracerProvider` | Reset provider, verify no errors |
| Context propagation | `TRACEPARENT` injected into pod/container env | Mock backend, assert env var present |
| Metrics | Correct metric names, values, and dimensions | `InMemoryMetricReader` assertions |

Tests follow the existing SDK parametrized pattern using an `OtelTestCase` dataclass that
extends `TestCase` with `expected_spans`, `expected_attributes`, and
`expected_span_status` fields.

### Graduation Criteria

- **Alpha:** TrainerClient + all three backends instrumented with traces. Shared telemetry
  module complete. Optional `configure_telemetry()` helper working. Unit tests passing.
  Documentation and examples.
- **Beta:** Metrics implemented. Log correlation documented and tested. GenAI conventions
  applied where applicable. Pipelines repo alignment documented for compatible future
  PipelinesClient instrumentation.
- **Stable:** Stretch clients (OptimizerClient, ModelRegistryClient, SparkClient)
  instrumented. E2E tests with OTel Collector. At least one release cycle with no
  breaking changes to span names or attribute keys.

## Implementation Plan

| Phase | Scope | Deliverables |
|---|---|---|
| **Phase 1** | Telemetry module | `kubeflow/common/telemetry/` with `configure_telemetry()`, attribute constants, propagation helpers. `pyproject.toml` dependency changes. Unit tests. |
| **Phase 2** | TrainerClient tracing | Spans on all 10 public methods. Error recording. Unit tests with `InMemorySpanExporter`. |
| **Phase 3** | Backend tracing | Spans on all 3 backends. `TRACEPARENT` injection. Span events for sub-operations. GenAI convention attributes where applicable. Unit tests. |
| **Phase 4** | Metrics + log correlation | 4 metrics with attribute dimensions. Log correlation documentation and optional `configure_telemetry()` logging bridge option. Unit tests with `InMemoryMetricReader`. |
| **Phase 5** | Documentation | Getting started guide, configuration reference, examples with Jaeger + Prometheus. Pipelines repo alignment notes for compatible span names and attributes. |
| **Phase 6** | Stretch: OptimizerClient | Two-level tracing for OptimizerClient (Katib). Unit tests. |
| **Phase 7** | Stretch: ModelRegistryClient + SparkClient | Single-level tracing for ModelRegistryClient, two-level tracing for SparkClient. Unit tests. |

## Implementation History

- 2025-11-18: Issue #164 created
- 2026-03-12: KEP created
- 2026-03-23: KEP submitted for review

## Drawbacks

- Two new core dependencies (`opentelemetry-api`, `opentelemetry-semantic-conventions`).
  Both are lightweight, but they still increase the dependency surface.
- Instrumentation code adds visual noise to method bodies. The context manager pattern
  wraps business logic in `with` blocks, adding one level of indentation.
- Attribute names and span conventions need to stay consistent across multiple clients
  as the SDK evolves. That takes discipline during code review.

## Alternatives

### Alternative 1: Separate instrumentation package

Create a standalone `opentelemetry-instrumentation-kubeflow` package that monkey-patches
SDK methods from outside.

Rejected because the
[OTel native instrumentation guidelines](https://opentelemetry.io/docs/concepts/instrumentation/libraries/)
recommend built-in instrumentation for libraries you own. Monkey-patching breaks silently
when SDK method signatures change, and it can't easily inject `TRACEPARENT` into pod specs.
Two packages to maintain and keep in sync is unnecessary overhead.

### Alternative 2: `opentelemetry-api` as optional dependency

Make OTel an optional extra (`pip install kubeflow[telemetry]`) with `try/except` import
guards.

Rejected because every instrumentation call would need an `if _tracer:` guard, duplicating
control flow throughout the codebase. The OTel API is under 100KB compressed, with only `typing-extensions` and
`importlib-metadata` as transitive dependencies, and is no-op by default. The SDK already
depends on `kubernetes` which is much heavier. The OTel spec explicitly says libraries
should depend on the API.

### Alternative 3: Decorator-based instrumentation

Use `@tracer.start_as_current_span("method_name")` decorators instead of context managers.

Rejected because decorators don't give a handle to the span inside the method body. You
can't set attributes from return values (e.g., `job_name` from `train()`) or record
events at different points during execution. The workaround (`trace.get_current_span()`)
mixes both styles for no benefit. Context managers provide one consistent pattern with
full control.

### Alternative 4: Level 3 granularity (Client + Backend + Internal Operations)

Trace every internal method within backends (resolve runtime, build spec, create CR).

Deferred. Two levels (Client + Backend) provide enough debugging value for an initial
release. Level 3 creates 5-7 spans per operation vs. 2, which adds noise and storage cost.
Internal method names are implementation details that may change between releases. If users
need finer-grained visibility, it can be added later.

### Alternative 5: Full Kubeflow-specific configuration object

Add a `TelemetryConfig` parameter to every client constructor or introduce
`KUBEFLOW_OTEL_*` env vars that override standard OTel configuration.

Rejected in favor of user-owned standard OTel configuration as the primary path, with an
optional `configure_telemetry()` convenience function for quickstarts and simple scripts.
This avoids per-client config objects and custom env vars while still reducing boilerplate
for new users. Users who need full control configure OTel directly, and there is no
ambiguity about precedence because the helper only sets up standard OTel providers.
