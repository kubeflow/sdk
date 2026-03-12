# KEP-164: Integrate Kubeflow SDK with OpenTelemetry

## Authors

- Yash Agarwal - [@XploY04](https://github.com/XploY04)


## Summary

This KEP adds native OpenTelemetry instrumentation to the Kubeflow SDK so users can get
distributed traces, metrics, and correlated logs from SDK operations like training job
submission, status polling, and lifecycle management.

The SDK depends on `opentelemetry-api` only, which is no-op by default. Users who want
actual telemetry install `opentelemetry-sdk` and an exporter separately. No SDK code
changes required on their end.

The first implementation covers `TrainerClient` and all three backends (Kubernetes,
Container, LocalProcess). The shared telemetry module in `kubeflow/common/telemetry/` is
built to be reused by `PipelinesClient`, `OptimizerClient`, `ModelRegistryClient`, and
`SparkClient` later.

## Motivation

When a training job fails or takes too long, users have no way to see what the SDK
actually did. Which Kubernetes API calls were made? How long did each take? Where did it
break? Right now, the only observability is `logging.getLogger()` with no correlation to
job lifecycle.

This was raised as user feedback at KubeCon + CloudNativeCon 2025 NA and is tracked in
[Issue #164](https://github.com/kubeflow/sdk/issues/164).

OpenTelemetry is the CNCF standard for observability. Adding it to the SDK gives us:

- Trace visibility across SDK operations and into training pods
- Operational metrics (job counts, durations, error rates)
- Vendor-neutral export to Jaeger, Grafana Tempo, Datadog, or any OTel-compatible backend
- No-op behavior when telemetry is not configured (the OTel API is a no-op by default)

### Goals

- Instrument `TrainerClient` and all three backends (Kubernetes, Container, LocalProcess)
  with OTel traces.
- Build a shared telemetry module (`kubeflow/common/telemetry/`) that all SDK clients can
  reuse.
- Propagate trace context across pod boundaries via W3C `TRACEPARENT` env var injection.
- Collect operational metrics: job counts, operation durations, active jobs, errors.
- Correlate existing Python `logging` calls with trace context (handled by OTel's log
  bridge, no SDK code changes needed).
- Write documentation and examples showing how to set up observability.

### Non-Goals

- Building a custom observability backend or UI.
- Collecting training-specific metrics (loss, accuracy, learning rate). That belongs in
  experiment tracking tools like MLflow or Weights & Biases.
- Auto-instrumenting user training functions inside pods. Users opt into pod-side tracing
  by reading `TRACEPARENT` from their environment.
- Adding `opentelemetry-sdk` or any exporter as a dependency. The SDK depends on the API
  package only. Users pick their own SDK and exporter.
- Instrumenting third-party libraries (`kfp.Client`, `kubernetes.client`). That is the
  responsibility of those projects or separate instrumentation packages.

## Proposal

The SDK calls the OTel API directly from its own code, following the OTel recommendation
for [native library instrumentation](https://opentelemetry.io/docs/concepts/instrumentation/libraries/).

### User Experience

**Installation:**

```bash
# SDK works as before tracing is no-op
pip install kubeflow

# User adds OTel SDK + exporter when they want observability
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

**Usage (no changes to existing SDK code):**

```python
# --- User configures OTel once (their responsibility, not the SDK's) ---
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry import trace

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# --- Use Kubeflow SDK exactly as before traces are emitted automatically ---
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
│ kubeflow.trainer.type:  CustomTrainer
│ kubeflow.runtime.name:  torch-distributed
│ kubeflow.job.name:      a3f8b2c1d9e0f
│
└── KubernetesBackend.train                            [1.2s]  CLIENT
    │ kubeflow.backend.type:   kubernetes
    │ k8s.namespace.name:      default
    │ kubeflow.job.name:       a3f8b2c1d9e0f
    └── status: OK

TrainerClient.wait_for_job_status                      [45.0s] INTERNAL
│ kubeflow.job.name:     a3f8b2c1d9e0f
│ kubeflow.job.status:   Complete
│ Events:
│   [0.0s]   status_change: Created
│   [2.1s]   status_change: Running
│   [45.0s]  status_change: Complete
│
└── KubernetesBackend.wait_for_job_status              [44.8s] CLIENT
    └── status: OK
```

### User Stories

#### Story 1: Debugging a Slow Training Job Submission

As an ML engineer, I call `client.train()` and it takes 10 seconds to return. With OTel
traces, I can see that `KubernetesBackend.train` spent 9.5 seconds on
`create_namespaced_custom_object` the Kubernetes API server was slow, not the SDK.

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

#### Story 4: No impact when OTel is not configured

As a user who doesn't care about observability, I `pip install kubeflow` and use the SDK
as before. The `opentelemetry-api` dependency provides no-op implementations. Every
`tracer.start_as_current_span()` call returns a no-op span. No performance cost, no
behavioral change.

## Design Details

### Architecture

```
kubeflow/
├── common/
│   ├── telemetry/                    # NEW shared OTel infrastructure
│   │   ├── __init__.py               # get_tracer(), get_meter()
│   │   ├── attributes.py             # Kubeflow-specific attribute constants
│   │   └── propagation.py            # TRACEPARENT injection for pods/containers
│   ├── constants.py
│   ├── types.py
│   └── utils.py
│
├── trainer/
│   ├── api/
│   │   └── trainer_client.py         # MODIFIED add spans to public methods
│   ├── backends/
│   │   ├── kubernetes/backend.py     # MODIFIED add spans to backend methods
│   │   ├── container/backend.py      # MODIFIED add spans to backend methods
│   │   └── localprocess/backend.py   # MODIFIED add spans to backend methods
│   └── ...
│
├── pipelines/                         # Future uses same telemetry module
├── optimizer/                         # Future uses same telemetry module
├── hub/                               # Future uses same telemetry module
└── spark/                             # Future uses same telemetry module
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

Why `opentelemetry-api` as a core dependency (not optional):

The [OTel native instrumentation guidelines](https://opentelemetry.io/docs/concepts/instrumentation/libraries/)
say libraries should depend on `opentelemetry-api` only and leave the SDK choice to the
application developer. The API package is lightweight (under 100KB compressed, only
`typing-extensions` as a transitive dependency) and is a no-op by default. Making it a core dependency means we don't need `try/except` import
guards or `if tracer:` checks scattered through the codebase.

Why `opentelemetry-semantic-conventions`:

Gives us Python constants for standard attribute names (e.g., `K8S_NAMESPACE_NAME` instead
of the raw string `"k8s.namespace.name"`). Prevents silent typos and gives IDE
autocomplete.

### Instrumentation Style

All instrumentation uses the **context manager** pattern:

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

Why context managers over decorators: decorators don't give you a handle to the span
object inside the method body. You can't set attributes from return values (e.g.,
`job_name`) or at different points during execution. Context managers give full control
with one consistent pattern.

### Span Granularity: Two Levels (Client + Backend)

Every traced operation produces two spans:

```
ClientMethod       [INTERNAL]    ← what the user called
  └── BackendMethod  [CLIENT]    ← what actually executed
```

Sub-operations within a backend (resolving runtime, building spec, making the K8s API
call) are recorded as span events, not child spans. This keeps traces readable while
still giving you debugging detail.

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
| `gen_ai.request.model` | [GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) | When model initializer references a model (e.g., `hf://meta-llama/Llama-3`) |
| `error.type` | General semconv | On error spans |

Note: We do NOT use `k8s.job.name` for TrainJob names. That OTel attribute is defined for
Kubernetes batch/v1 Jobs, and a TrainJob is a CRD not a batch Job. TrainJob names go in
the Kubeflow-specific `kubeflow.job.name` attribute instead.

Note on GenAI conventions: these are designed for inference workloads (token counts,
prompts, completions). Most attributes (`temperature`, `max_tokens`, `top_p`,
`usage.input_tokens`) don't apply to training orchestration. We only use
`gen_ai.request.model` where it genuinely fits, i.e., when the SDK references a model by
name via an initializer. Everything else is Kubeflow-specific.

#### Kubeflow-Specific Attributes

```python
# kubeflow/common/telemetry/attributes.py

# Job attributes
KUBEFLOW_JOB_NAME = "kubeflow.job.name"
KUBEFLOW_JOB_STATUS = "kubeflow.job.status"

# Backend attributes
KUBEFLOW_BACKEND_TYPE = "kubeflow.backend.type"
# Values: "kubernetes", "container", "local-process"

KUBEFLOW_CONTAINER_RUNTIME = "kubeflow.container.runtime"
# Values: "docker", "podman"

# Trainer attributes
KUBEFLOW_TRAINER_TYPE = "kubeflow.trainer.type"
# Values: "CustomTrainer", "CustomTrainerContainer", "BuiltinTrainer"

KUBEFLOW_RUNTIME_NAME = "kubeflow.runtime.name"
KUBEFLOW_NUM_NODES = "kubeflow.num_nodes"

# Initializer attributes
KUBEFLOW_INITIALIZER_TYPE = "kubeflow.initializer.type"
KUBEFLOW_INITIALIZER_URI = "kubeflow.initializer.uri"

# Operation attribute (for metrics)
KUBEFLOW_OPERATION_NAME = "kubeflow.operation.name"
```

### Error Handling

Exceptions are recorded on spans with full tracebacks:

```python
except Exception as e:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    raise
```

`record_exception` adds the exception type, message, and traceback as a span event.
`set_status(ERROR)` marks the span as failed in visualization tools.

### Context Propagation

The SDK injects W3C Trace Context into training pods, containers, and subprocesses using
the `TRACEPARENT` environment variable. This follows the
[OTel Environment Carriers spec](https://opentelemetry.io/docs/specs/otel/context/env-carriers/)
(currently at Alpha status in the OTel specification).

```python
# kubeflow/common/telemetry/propagation.py

from opentelemetry import context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

def inject_trace_context() -> dict[str, str]:
    """Extract current trace context as environment variables.

    Returns:
        Dict of env vars (e.g., {"TRACEPARENT": "00-<trace_id>-<span_id>-01"})
        to inject into pod/container/subprocess environments.
    """
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
# Inside the training pod user writes this if they want end-to-end tracing
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

High-cardinality values (`kubeflow.job.name`, `kubeflow.runtime.name`) are never put on
metrics. They belong on spans only.

### Configuration

The SDK exposes no telemetry configuration of its own. Users configure everything through
standard OTel mechanisms:

- `TracerProvider` / `MeterProvider` in code
- `OTEL_TRACES_EXPORTER`, `OTEL_METRICS_EXPORTER` env vars
- `OTEL_SERVICE_NAME` env var

Resource attributes (`service.name`, `service.version`) are the user's responsibility. The
SDK documents recommended resource configuration in examples.

### Telemetry Module API

```python
# kubeflow/common/telemetry/__init__.py

from opentelemetry import trace, metrics

def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer for the given module name.

    Args:
        name: Tracer name, typically the module path (e.g., "kubeflow.trainer").

    Returns:
        A Tracer instance. Returns a no-op tracer if no TracerProvider is configured.
    """
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Get a meter for the given module name.

    Args:
        name: Meter name, typically the module path (e.g., "kubeflow.trainer").

    Returns:
        A Meter instance. Returns a no-op meter if no MeterProvider is configured.
    """
    return metrics.get_meter(name)
```

### PipelinesClient Instrumentation (Planned)

`PipelinesClient` ([PR #343](https://github.com/kubeflow/sdk/pull/343), pending merge)
wraps `kfp.Client` for pipeline operations. It follows the same instrumentation pattern:

- Each public I/O method gets a span (`INTERNAL` kind)
- Unlike `TrainerClient`, there is no backend layer `PipelinesClient` delegates directly
  to `kfp.Client`, which is a third-party library
- Tracing the `kfp.Client` HTTP calls would require a separate instrumentation library or
  upstream KFP support
- Attributes follow the same `kubeflow.*` prefix:

```python
KUBEFLOW_PIPELINE_NAME = "kubeflow.pipeline.name"
KUBEFLOW_PIPELINE_VERSION_ID = "kubeflow.pipeline.version_id"
KUBEFLOW_EXPERIMENT_NAME = "kubeflow.experiment.name"
KUBEFLOW_RUN_ID = "kubeflow.run.id"
KUBEFLOW_RUN_STATUS = "kubeflow.run.status"
```

Implementation is deferred until `PipelinesClient` is merged and stable.

### OptimizerClient, ModelRegistryClient, SparkClient (Planned)

These clients follow the same pattern established by `TrainerClient`:

| Client | Backend Architecture | SpanKind |
|---|---|---|
| `OptimizerClient` | Kubernetes backend only | INTERNAL / CLIENT |
| `ModelRegistryClient` | REST API wrapper | INTERNAL (single level) |
| `SparkClient` | Kubernetes backend only | INTERNAL / CLIENT |

They all use the same `kubeflow/common/telemetry/` module and follow the same two-level
tracing pattern with `kubeflow.*` attributes.


### Test Plan

`opentelemetry-sdk` is added to the `dev` dependency group for `InMemorySpanExporter`.

A shared helper goes in `kubeflow/trainer/test/common.py`:

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

Test structure (follows existing SDK patterns):

```python
@dataclass
class OtelTestCase(TestCase):
    expected_spans: list[str] = field(default_factory=list)
    expected_attributes: dict[str, Any] = field(default_factory=dict)
    expected_span_status: str = "OK"


@pytest.mark.parametrize(
    "test_case",
    [
        OtelTestCase(
            name="train creates client and backend spans",
            expected_status=SUCCESS,
            config={...},
            expected_spans=["TrainerClient.train", "KubernetesBackend.train"],
            expected_attributes={
                "kubeflow.backend.type": "kubernetes",
                "kubeflow.trainer.type": "CustomTrainer",
            },
        ),
        OtelTestCase(
            name="train records error on failure",
            expected_status=FAILED,
            config={...},
            expected_error=RuntimeError,
            expected_spans=["TrainerClient.train", "KubernetesBackend.train"],
            expected_span_status="ERROR",
        ),
    ],
)
def test_train_telemetry(test_case, kubernetes_backend):
    print("Executing test:", test_case.name)
    exporter = setup_test_telemetry()

    try:
        result = kubernetes_backend.train(**test_case.config)
        assert test_case.expected_status == SUCCESS
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    for expected in test_case.expected_spans:
        assert expected in span_names

    print("test execution complete")
```

### Graduation Criteria

- **Alpha:** TrainerClient + all three backends instrumented with traces. Shared telemetry
  module complete. Unit tests passing. Documentation and examples.
- **Beta:** Metrics implemented. PipelinesClient instrumented (after PR #343 merges).
  OptimizerClient instrumented.
- **Stable:** All SDK clients instrumented. E2E tests with OTel Collector. Community
  feedback addressed.

## Implementation Plan

| Phase | Scope | Deliverables |
|---|---|---|
| **Phase 1** | Telemetry module | `kubeflow/common/telemetry/` with `get_tracer()`, `get_meter()`, attribute constants, propagation helpers. `pyproject.toml` dependency changes. Unit tests. |
| **Phase 2** | TrainerClient tracing | Spans on all 10 public methods. Error recording. Unit tests with `InMemorySpanExporter`. |
| **Phase 3** | Backend tracing | Spans on all 3 backends. `TRACEPARENT` injection. Span events for sub-operations. Unit tests. |
| **Phase 4** | Metrics | 4 metrics with attribute dimensions. Unit tests with `InMemoryMetricReader`. |
| **Phase 5** | Documentation | Getting started guide, configuration reference, examples with Jaeger + Prometheus. |
| **Phase 6** | PipelinesClient | Instrumentation for PipelinesClient (after PR #343 merges). |
| **Phase 7** | Bonus clients | OptimizerClient, ModelRegistryClient, SparkClient. |

## Implementation History

- 2025-11-18: [Issue #164](https://github.com/kubeflow/sdk/issues/164) created
- 2026-03-12: KEP created

## Drawbacks

- Two new core dependencies (`opentelemetry-api`, `opentelemetry-semantic-conventions`).
  Both are lightweight, but they still increase the dependency surface.
- Instrumentation code adds some visual noise to method bodies. The context manager
  pattern wraps business logic in `with` blocks, adding one level of indentation.
- Attribute names and span conventions need to stay consistent across multiple clients
  as the SDK evolves. That takes discipline during code review.

## Alternatives

### Alternative 1: Separate instrumentation package

Create a standalone `opentelemetry-instrumentation-kubeflow` package that monkey-patches
SDK methods from outside.

We rejected this because the
[OTel native instrumentation guidelines](https://opentelemetry.io/docs/concepts/instrumentation/libraries/)
recommend built-in instrumentation for libraries you own. Monkey-patching also breaks
silently when SDK method signatures change, and it can't easily inject `TRACEPARENT` into
pod specs. Two packages to maintain and keep in sync is unnecessary overhead when we own
the code.

### Alternative 2: `opentelemetry-api` as optional dependency

Make OTel an optional extra (`pip install kubeflow[telemetry]`) with `try/except` import
guards.

We rejected this because every instrumentation call would need an `if _tracer:` guard,
duplicating control flow throughout the codebase. The OTel API is under 100KB
compressed, with only `typing-extensions` as a transitive dependency, and no-op by default. The SDK already depends on `kubernetes` which is much heavier. The OTel spec
explicitly says libraries should depend on the API.

### Alternative 3: Decorator-based instrumentation

Use `@tracer.start_as_current_span("method_name")` decorators instead of context managers.

We rejected this because decorators don't give you a handle to the span inside the method
body. You can't set attributes from return values (e.g., `job_name` from `train()`). The
workaround (`trace.get_current_span()`) mixes both styles for no benefit. Context managers
give one consistent pattern with full control.

### Alternative 4: Level 3 granularity (Client + Backend + Internal Operations)

Trace every internal method within backends (resolve runtime, build spec, create CR).

Deferred. Two levels (Client + Backend) give enough debugging value for an initial
release. Level 3 creates 5-7 spans per operation vs. 2, which adds noise and storage cost.
Internal method names are implementation details that may change between releases. If users
need finer-grained visibility, we can add it later.

### Alternative 5: Kubeflow-specific configuration

Add `TelemetryConfig` parameter to client constructors or `KUBEFLOW_OTEL_*` env vars.

Deferred. OTel already has configuration via `TracerProvider`, `MeterProvider`, and
standard `OTEL_*` env vars. Adding a Kubeflow-specific layer on top creates confusion
about which config takes precedence. Can be added as a non-breaking change later if users
ask for it.
