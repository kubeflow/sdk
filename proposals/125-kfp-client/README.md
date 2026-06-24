# KEP-125: PipelinesClient for Kubeflow SDK

| | |
| --- | --- |
| **Authors** | [MStokluska](https://github.com/MStokluska) |
| **Created** | 2026-03-02 |
| **Relevant Issues** | https://github.com/kubeflow/sdk/issues/125 |

## Table of Contents

- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [User Stories](#user-stories)
  - [Story 1: Quick one-off run](#story-1-quick-one-off-run)
  - [Story 2: Upload once, run many times](#story-2-upload-once-run-many-times)
  - [Story 3: Monitor and wait for specific states](#story-3-monitor-and-wait-for-specific-states)
  - [Story 4: Kubernetes-native pipeline with PVC and secrets](#story-4-kubernetes-native-pipeline-with-pvc-and-secrets)
  - [Story 5: Multi-component pipeline — Spark, Trainer, and Model Registry](#story-5-multi-component-pipeline--spark-trainer-and-model-registry)
- [Proposal](#proposal)
  - [Architecture](#architecture)
  - [Dependency](#dependency)
  - [Constructor](#constructor)
    - [`PipelinesBackendConfig`](#pipelinesbackendconfig)
  - [Escape hatch (`kfp.Client`)](#escape-hatch-kfpclient)
  - [DSL Re-exports](#dsl-re-exports)
  - [Phased API](#phased-api)
    - [Core workflow](#core-workflow)
    - [Phase 1 summary](#phase-1-summary)
- [Design Details](#design-details)
  - [Type aliases](#type-aliases)
  - [KFP-side package structure](#kfp-side-package-structure)
  - [SDK-side package structure](#sdk-side-package-structure)
  - [Error Handling](#error-handling)
  - [Test Plan](#test-plan)
- [Implementation Plan](#implementation-plan)
- [Migration](#migration)
- [Implementation History](#implementation-history)
- [Alternatives](#alternatives)

## Summary

Add a `PipelinesClient` to the Kubeflow SDK that gives users the full
author → compile → upload → run → monitor pipeline workflow from a single
kubeflow import.

The KEP focuses on Phase 1 (MVP) - the API users
get first and how it maps from today’s `kfp.Client`.

Following discussions with the KFP team, the client will be implemented in
the KFP repository at a proposed location, `kfp.kubeflow_client.pipelines_client`, and
re-exported by the Kubeflow SDK. This means:

- **KFP team maintains the client** — KFP owns the implementation, tests, and
  releases.
- **Kubeflow SDK re-exports it** — we import from `kfp.kubeflow_client.pipelines_client` and
  expose it at `kubeflow.pipelines.PipelinesClient`.
- **The client is additive** — `kfp.Client` remains fully supported. The new
  client provides a simplified, name-first API alongside it, in a phased approach.

```bash
pip install 'kubeflow[pipelines]'
```

```python
from kubeflow.pipelines import PipelinesClient, dsl, compiler, components, kubernetes
```

## Motivation

Today, Kubeflow users who want to orchestrate ML pipelines must install and use
`kfp` separately from the Kubeflow SDK. This creates friction:

- **Two SDKs, two import styles** — users mix `from kfp import ...` with
  `from kubeflow.trainer import ...`.
- **Inconsistent constructor** — `kfp.Client(host=..., existing_token=...,
  namespace="kubeflow")` vs the `backend_config` pattern used by `TrainerClient`,
  `SparkClient`, and `OptimizerClient`.
- **ID-centric API** — `kfp.Client` requires pipeline IDs and experiment IDs
  before triggering a run. Other SDK clients are name-first.
- **Too many methods for simple tasks** — uploading a pipeline requires choosing
  between 4 methods depending on source type and whether it's new or a version.
  Running a pipeline requires choosing between 3 methods.
- **Scattered wait semantics** — `kfp.Client.wait_for_run_completion` hardcodes
  terminal states and cannot wait for intermediate states like `Running`.

### Goals

1. Expose KFP pipeline management through the Kubeflow SDK with
   `pip install 'kubeflow[pipelines]'`.
2. Provide a simplified, name-first API that unifies upload variants and
   reduces the number of methods users need to learn.
3. Re-export `kfp.dsl`, `kfp.compiler`, `kfp.components`, and `kfp.kubernetes`
   at the `kubeflow.pipelines` module level.
4. Add `wait_for_run_status` with a flexible status set and callbacks (matching
   `TrainerClient.wait_for_job_status`).
5. Adopt the `backend_config` pattern (matching `TrainerClient`, `SparkClient`,
   `OptimizerClient`) via `PipelinesBackendConfig` for KFP connection
   parameters, enabling future local/container backends without breaking changes.
6. Deliver the API in phases — Phase 1 covers the core workflow with
   minimal methods; later phases add management, observability, and advanced
   features.
7. Have the client implemented in the KFP repository and re-exported by
   the Kubeflow SDK, ensuring KFP team ownership.

### Non-Goals

- Replacing `kfp.Client`. It remains fully supported. Power users who need
  advanced auth (IAP, cookies, proxy) or raw generated APIs continue to use
  `kfp` directly.
- Wrapping the DSL. `dsl`, `compiler`, `components`, and `kubernetes` are
  re-exported as-is — no additional abstraction layer.
- Supporting KFP control-plane provisioning. This client is data-plane only.
- Implementing task logs or run events in early phases. These require upstream
  KFP changes.
- Thread safety or async support. The client follows the same synchronous,
  single-threaded model as `kfp.Client` and other SDK clients.

---

## User Stories

All examples use the proposed Phase 1 API.

### Story 1: Quick one-off run

As a data scientist, I want to define a pipeline, upload it, and run it with
minimal boilerplate.

```python
from kubeflow.pipelines import PipelinesClient, dsl

@dsl.component
def train(epochs: int) -> str:
    return f"trained for {epochs} epochs"

@dsl.pipeline
def my_pipeline(epochs: int = 10):
    train(epochs=epochs)

client = PipelinesClient()

# Upload (compiles automatically from function)
client.upload_pipeline(my_pipeline, name="training-pipeline")

# Run the uploaded pipeline
run = client.run("training-pipeline", params={"epochs": 5})

# Wait for completion
completed = client.wait_for_run_status(run, timeout=3600)
print(f"Finished: {completed.state}")
```

No IDs, no experiment setup, no "which upload_pipeline/run method do I use?" decisions.

### Story 2: Upload once, run many times

As an ML engineer, I want to upload a pipeline once, then trigger runs with
different parameters.

```python
from kubeflow.pipelines import PipelinesClient, dsl

@dsl.component
def preprocess(data_path: str) -> str:
    return f"preprocessed {data_path}"

@dsl.component
def train(data_path: str, lr: float, epochs: int) -> str:
    return f"trained on {data_path} lr={lr} epochs={epochs}"

@dsl.pipeline
def training_pipeline(data_path: str = "/data", lr: float = 0.001, epochs: int = 10):
    data = preprocess(data_path=data_path)
    train(data_path=data.output, lr=lr, epochs=epochs)

client = PipelinesClient()

# Upload once
client.upload_pipeline(training_pipeline, name="training-pipeline")

# Create experiment
client.create_experiment("hyperparameter-sweep")

# Run with different parameters
run1 = client.run(
    "training-pipeline",
    name="lr-0.01",
    experiment="hyperparameter-sweep",
    params={"lr": 0.01, "epochs": 20},
)

run2 = client.run(
    "training-pipeline",
    name="lr-0.001",
    experiment="hyperparameter-sweep",
    params={"lr": 0.001, "epochs": 50},
)

# Wait for both
client.wait_for_run_status(run1, timeout=3600)
client.wait_for_run_status(run2, timeout=3600)
```

No ID juggling. Pipeline and experiment resolved by name. Latest pipeline
version used automatically.

**Experiments:** Phase 1 includes full experiment CRUD: `create_experiment`,
`list_experiments`, `get_experiment`, `delete_experiment`. The `experiment`
parameter on `run` requires the experiment to already exist; it will not be
auto-created. Note: the experiment API may evolve after RunGroups rename and
MLflow integration ([kubeflow/community#892](https://github.com/kubeflow/community/issues/892)).

### Story 3: Monitor and wait for specific states

As a data scientist, I want to wait until my run starts executing (not just
completes) so I can start tailing external logs.

```python
client = PipelinesClient()

run = client.run("training-pipeline", params={"epochs": 10})

# Wait for the run to start (not complete)
running = client.wait_for_run_status(run, status={"running"}, timeout=120)
print(f"Run is now {running.state} — starting log tail...") # This is just illustrative

# Later, wait only for success
completed = client.wait_for_run_status(run, status={"succeeded"}, timeout=3600)
print(f"Finished with state: {completed.state}")
```

`kfp.Client.wait_for_run_completion` can only wait for all terminal states at
once. `wait_for_run_status` accepts any set of states — terminal or non-terminal.
It always exits immediately on any terminal state regardless of `status`.

### Story 4: Kubernetes-native pipeline with PVC and secrets

As an ML engineer, I want to orchestrate distributed training through a
pipeline that mounts shared storage and injects K8s secrets.

```python
from kubeflow.pipelines import PipelinesClient, dsl, kubernetes

@dsl.component(base_image="python:3.12", packages_to_install=["datasets"])
def download_dataset(output_dir: str, subset_size: int = 1000):
    from datasets import load_dataset
    ds = load_dataset("LipengCS/Table-GPT", split=f"train[:{subset_size}]")
    ds.save_to_disk(output_dir)

@dsl.component(base_image="quay.io/my-org/training-image:latest")
def submit_training(data_path: str, num_epochs: int = 3):
    from kubeflow.trainer import TrainerClient

    client = TrainerClient()
    # Data is already on the PVC from download_dataset; no dataset Initializer
    # needed here.
    job_name = client.train(
        trainer=...,  # e.g. TorchTrainer configured to read from data_path
    )
    client.wait_for_job_status(job_name)

@dsl.pipeline(name="distributed-training")
def training_pipeline(subset_size: int = 1000, num_epochs: int = 3):
    # kubernetes.* helpers return the PipelineTask so PVC and Secret config chain naturally
    download_task = kubernetes.mount_pvc(
        download_dataset(output_dir="/mnt/shared/data", subset_size=subset_size),
        pvc_name="shared-pvc",
        mount_path="/mnt/shared",
    )

    train_task = kubernetes.use_secret_as_env(
        kubernetes.mount_pvc(
            submit_training(data_path="/mnt/shared/data", num_epochs=num_epochs),
            pvc_name="shared-pvc",
            mount_path="/mnt/shared",
        ),
        secret_name="k8s-creds",
        secret_key_to_env={"token": "K8S_TOKEN"},
    )
    train_task.after(download_task)

client = PipelinesClient()
client.upload_pipeline(training_pipeline, name="distributed-training")

run = client.run(
    "distributed-training",
    experiment="distributed-experiments",
    params={"subset_size": 5000, "num_epochs": 5},
)

completed = client.wait_for_run_status(run, timeout=7200)
print(f"Run finished: {completed.state}")
```

`kfp.kubernetes` is re-exported at `kubeflow.pipelines.kubernetes` so users
never need a direct `kfp` import. `kubernetes.mount_pvc` /
`kubernetes.use_secret_as_env` compose by fluent chaining (each helper
returns the `PipelineTask` it configures).

This story keeps `load_dataset` as a pipeline step
because it performs custom preprocessing (subset, format) and writes to a
shared PVC for the training step. If a job only needs “download this S3/HF
prefix as-is,” `train(..., initializer=Initializer(dataset=...))` can be used
instead and a separate download component omitted.

### Story 5: Multi-component pipeline — Spark, Trainer, and Model Registry

As an ML engineer, I want **`PipelinesClient`** to orchestrate one pipeline that
chains **Spark → Trainer → Model Registry**, with full Trainer wiring for
LLM-style fine-tuning: **BuiltinTrainer**, **TorchTune**, and **dataset and model
Initializers** (e.g. Spark output on S3 + Hugging Face base model).

```python
from pathlib import Path

from kubeflow.pipelines import PipelinesClient, dsl

@dsl.component(base_image="quay.io/my-org/spark-image:latest")
def preprocess_data(input_path: str, output_path: str):
    from kubeflow.spark import SparkClient

    client = SparkClient()
    spark = client.connect(
        num_executors=4,
        resources_per_executor={"cpu": "2", "memory": "4Gi"},
    )
    df = spark.read.parquet(input_path)
    df_clean = df.dropna().filter(df["quality"] > 0.5)
    df_clean.write.parquet(output_path)
    spark.stop()

@dsl.component(base_image="quay.io/my-org/training-image:latest")
def train_model(
    data_path: str,
    epochs: int,
    trained_model_uri: dsl.OutputPath(str),
):
    from kubeflow.trainer import TrainerClient
    from kubeflow.trainer.types import types as trainer_types

    trainer = TrainerClient()
    job_name = trainer.train(
        trainer=trainer_types.BuiltinTrainer(
            config=trainer_types.TorchTuneConfig(
                epochs=epochs,
                dataset_preprocess_config=trainer_types.TorchTuneInstructDataset(
                    source=trainer_types.DataFormat.PARQUET,
                ),
            ),
        ),
        initializer=trainer_types.Initializer(
            dataset=trainer_types.S3DatasetInitializer(storage_uri=data_path),
            model=trainer_types.HuggingFaceModelInitializer(
                storage_uri="hf://google-bert/bert-base-uncased",
            ),
        ),
    )
    trainer.wait_for_job_status(job_name)

    model_uri = f"s3://my-org-models/{job_name}/checkpoint"
    Path(trained_model_uri).parent.mkdir(parents=True, exist_ok=True)
    Path(trained_model_uri).write_text(model_uri)

@dsl.component(base_image="quay.io/my-org/registry-image:latest")
def register_model(
    model_name: str,
    version: str,
    trained_model_uri: str,
):
    from kubeflow.hub import ModelRegistryClient

    registry = ModelRegistryClient("https://registry.example.com")
    registry.register_model(
        name=model_name,
        uri=trained_model_uri,
        version=version,
        model_format_name="pytorch",
    )

@dsl.pipeline(name="preprocess-train-register")
def full_pipeline(
    input_path: str = "s3://data/raw",
    output_path: str = "s3://data/processed",
    model_name: str = "my-model",
    version: str = "v1",
    epochs: int = 5,
):
    preprocess = preprocess_data(input_path=input_path, output_path=output_path)
    train = train_model(data_path=output_path, epochs=epochs)
    train.after(preprocess)

    register_model(
        model_name=model_name,
        version=version,
        trained_model_uri=train.outputs["trained_model_uri"],
    )

client = PipelinesClient()

# Compile and submit inline — no upload, no catalog entry
run = client.run(
    full_pipeline,
    params={"model_name": "my-model", "version": "v2", "epochs": 10},
)

completed = client.wait_for_run_status(run, timeout=7200)
print(f"Pipeline finished: {completed.state}")
```

One `pip install 'kubeflow[pipelines,spark,hub]'` for this shape. `TrainerClient`
is in the base `kubeflow` package; `pipelines`, `spark`, and `hub`
are extras.

SDK surfaces: `PipelinesClient` (`run` with a callable = compile and submit inline,
no upload), `kfp` DSL (DAG, `dsl.OutputPath`), and imports for Spark, Trainer,
and Model Registry.

---

## Proposal

### Architecture

The client is implemented in the **KFP repository** at `kfp.kubeflow_client.pipelines_client`
and re-exported by the Kubeflow SDK.

```
KFP repository (github.com/kubeflow/pipelines)
└── sdk/python/kfp/            # Python package root (import name: kfp)
    ├── client/
    │   └── client.py          # existing kfp.Client (unchanged)
    └── kubeflow_client/
        └── pipelines_client.py  # NEW: PipelinesClient (simplified API)
```

```
Kubeflow SDK repository (github.com/kubeflow/sdk)
└── kubeflow/pipelines/
    ├── api/
    │   └── pipelines_client.py    # re-exports PipelinesClient
    └── __init__.py                # re-exports PipelinesClient + dsl/compiler/components/kubernetes
```

### Dependency

On Kubeflow SDK side the pipelines optional dependency will list kfp[kubernetes] only; we will not depend on kfp[notebooks] or kfp[all] by default.

```toml
[project.optional-dependencies]
pipelines = ["kfp[kubernetes]>=X.Y.Z"]  # first kfp release shipping kfp.kubeflow_client.pipelines_client
```

**Why `kfp[kubernetes]` only:**

KFP ships three extras (as of kfp 2.x):

- **`kfp[kubernetes]`** — lightweight. Adds K8s helpers (`mount_pvc`,
  `use_secret_as_env`, `add_node_selector`) that most pipeline users need.
- **`kfp[notebooks]`** — heavy (nbclient, ipykernel). Only needed for
  notebook-as-component use cases.
- **`kfp[all]`** — heaviest (docker + notebooks). Overkill as a default.

### Constructor

Follows the `backend_config` pattern used by `TrainerClient`, `SparkClient`, and
`OptimizerClient`. Phase 1 supports a single backend via `PipelinesBackendConfig`;
future phases can add `PipelinesLocalBackendConfig` and `PipelinesContainerBackendConfig`
without breaking the constructor API.

```python
PipelinesClient(
    backend_config: PipelinesBackendConfig | None = None,
)
```

When `backend_config` is `None`, defaults to `PipelinesBackendConfig()` (zero-arg
construction with auto-discovery).

#### `PipelinesBackendConfig`

```python
@dataclass
class PipelinesBackendConfig:
    base_url: str | None = None
    user_token: str | None = None
    is_secure: bool | None = None
    custom_ca: str | None = None
    namespace: str | None = None
```

| Parameter | Description |
|---|---|
| `base_url` | KFP API server URL including scheme and port (e.g. `https://ml-pipeline.example.com:8080`). If omitted, auto-discovered following `kfp.Client` conventions (in-cluster DNS or kubeconfig proxy) |
| `user_token` | Bearer token for authentication |
| `is_secure` | Inferred from scheme if omitted |
| `custom_ca` | Path to PEM-encoded root certificates |
| `namespace` | K8s namespace. If omitted, auto-detected following `kfp.Client` conventions |

**Usage:**

```python
# Zero-arg (auto-discovery)
client = PipelinesClient()

# Explicit config
client = PipelinesClient(
    backend_config=PipelinesBackendConfig(
        base_url="https://ml-pipeline.example.com",
        user_token="...",
    )
)
```

### Escape hatch (`kfp.Client`)

`PipelinesClient` does not try to wrap every `kfp.Client` knob (IAP, cookies,
proxy, raw generated APIs, etc.). Power users either construct a dedicated
`kfp.Client(...)` alongside `PipelinesClient`, or use a `.kfp_client`
property on `PipelinesClient` to access the underlying `kfp.Client` instance
without duplicating connection configuration. The exact accessor name is
subject to change during KFP implementation.

### DSL Re-exports

`kfp.dsl`, `kfp.compiler`, `kfp.components`, and `kfp.kubernetes` are
re-exported at module level with zero wrapping:

```python
from kubeflow.pipelines import PipelinesClient, dsl, compiler, components, kubernetes
```

The DSL is KFP's domain-specific language — it cannot be meaningfully wrapped.
Re-exporting gives users a single namespace for the entire
author → configure → upload → run flow without any direct `kfp` imports.

### Phased API

**Phase 1 is the MVP** and is what this KEP
specifies in detail.

#### Core workflow

##### `upload_pipeline`

Unified upload that handles functions, files, new pipelines, and new versions.

```python
# From a @dsl.pipeline function — auto-compiles
client.upload_pipeline(my_pipeline, name="training-pipeline")

# From a compiled YAML file
client.upload_pipeline("training-pipeline.yaml", name="training-pipeline")

# New version — same method, auto-detects existing pipeline
client.upload_pipeline(my_pipeline_v2, name="training-pipeline", version="v2-with-caching")
```

The user's intent is always "put this pipeline on the server". The client
handles the implementation details:

- First arg is callable → compile it
- First arg is a string path → use the file
- Pipeline with that name exists → create new version
- Pipeline doesn't exist → create it
- `name` omitted → auto-generate from callable's `@dsl.pipeline(name=...)` value, or the function name; for file paths, use the filename without extension.
  Calling `upload_pipeline` again with the same `name` and no explicit `version`
  creates a **new version** each time (not idempotent)
- `version` omitted → auto-generate version label following the same conventions as the KFP UI (exact format deferred to implementation)

```python
def upload_pipeline(
    self,
    pipeline: Callable | str,
    *,
    name: str | None = None,
    version: str | None = None,
    description: str | None = None,
) -> PipelineVersion:
```

**Why unified:** `kfp.Client` currently has four upload methods based on two
axes (func vs file, new vs version). Users shouldn't need to understand these
implementation distinctions:

| User intent | Current `kfp.Client` (choose one) | New client |
|---|---|---|
| Upload from function | `upload_pipeline_from_pipeline_func` | `upload_pipeline(fn, name=...)` |
| Upload from file | `upload_pipeline` | `upload_pipeline("file.yaml", name=...)` |
| New version from function | `upload_pipeline_version_from_pipeline_func` | `upload_pipeline(fn, name=..., version=...)` |
| New version from file | `upload_pipeline_version` | `upload_pipeline("file.yaml", name=..., version=...)` |

##### `run`

Run a pipeline by name, from a function or YAML file (compile-and-submit inline), or from a pipeline/version object.

```python
# Run an uploaded pipeline by name
run = client.run(
    "training-pipeline",
    params={"epochs": 10, "lr": 0.001},
)

# Quick inline run from function — compile and submit, no upload
run = client.run(
    my_pipeline,
    params={"epochs": 5},
)

# Quick inline run from a compiled YAML file — submit, no upload
run = client.run(
    "training-pipeline.yaml",
    params={"epochs": 5},
)

# Pass the return value of upload_pipeline or get_pipeline directly
pipeline = client.get_pipeline("training-pipeline")
run = client.run(pipeline, params={"epochs": 10})
```

```python
def run(
    self,
    pipeline: str | Callable | Pipeline | PipelineVersion,
    *,
    params: dict[str, Any] | None = None,
    name: str | None = None,
    experiment: str | None = None,
    version: str | None = None,
) -> Run:
```

| Parameter | Description |
|---|---|
| `pipeline` | Pipeline name (`str`), path to a compiled YAML file (`str`), `@dsl.pipeline` function, or `Pipeline`/`PipelineVersion` object. If a callable or file path, **compile and submit inline** — no upload, no catalog entry. If a name, resolves the uploaded pipeline on the server |
| `params` | Pipeline parameters |
| `name` | Run display name (auto-generated if omitted following kfp.Client conventions) |
| `experiment` | Experiment name. If `None`, the server's default experiment is used (typically `"Default"`). If a name is provided and no experiment with that name exists, raises `ValueError` — use `create_experiment()` first |
| `version` | Pipeline version to use (latest if omitted; ignored when `pipeline` is a callable, file path, or `PipelineVersion`) |

**Three workflow patterns:**

- **Upload once, run many** (Story 2): Call `upload_pipeline` separately, then `run`
  by pipeline name. Best when you want versioning and to trigger multiple runs from
  the same pipeline.
- **Quick inline run from function**: Pass a callable directly to `run` (Story 5) — the
  client compiles and submits the spec inline without registering a pipeline resource.
  The pipeline will not appear in the catalog.
- **Quick inline run from file**: Pass a YAML file path to `run` — the client submits
  the compiled spec inline without registering a pipeline resource. Use
  `upload_pipeline` first if you want the pipeline to be rerunnable by name from the
  server.

##### `wait_for_run_status`

Wait for a run to reach a target state. Accepts a run ID string or the
`Run` object returned from `run`.

```python
completed = client.wait_for_run_status(run, timeout=3600)

# Wait for a specific non-terminal state
running = client.wait_for_run_status(run, status={"running"}, timeout=120)

# Use callbacks instead of raise_on_failure
def on_complete(run: Run) -> None:
    print(f"Run finished with state: {run.state}")

completed = client.wait_for_run_status(run, callbacks=[on_complete])

# Also accepts a raw run ID
completed = client.wait_for_run_status("abc-123", timeout=3600)
```

```python
def wait_for_run_status(
    self,
    run: str | Run,
    *,
    status: set[str] = {constants.RUN_COMPLETE},
    timeout: int = 600,
    polling_interval: int = 5,
    callbacks: list[Callable[[types.Run], None]] | None = None,
) -> Run:
```

**Semantics**

- **`status` —** Stop when the run state is in `status`, or sooner on any terminal state (`succeeded`, `failed`, `skipped`, `error`, `cancelled`) or `timeout`. Default is `{constants.RUN_COMPLETE}` (`"succeeded"`).

- **`callbacks` —** Called with the final `Run` object when the wait ends (on any stop condition). Replaces the `raise_on_failure` flag. Callbacks can inspect `run.state` and raise or log as needed.

- **`timeout` —** Maximum seconds to wait. `TimeoutError` if the window expires before a stop condition. Defaults to 600 (10 minutes), matching `TrainerClient.wait_for_job_status`.

**Example:** You asked for `{"running"}` but the run `failed` first → wait stops immediately; any provided callbacks receive the `Run` object.

##### `get_run`

Inspect a run by ID.

```python
run_info = client.get_run("abc-123")
```

```python
def get_run(self, run_id: str) -> Run:
```

##### `get_pipeline`

Inspect a pipeline by name.

```python
pipeline = client.get_pipeline("training-pipeline")
```

```python
def get_pipeline(self, name: str) -> Pipeline:
```

**Name resolution:**
- Exactly one pipeline match → return the `Pipeline` object
- Zero matches → raise `ValueError` ("pipeline not found")
- Multiple matches (possible when mixing clients) → raise `ValueError` with the
  list of ambiguous IDs so the user can fall back to `kfp.Client`

Within `PipelinesClient`'s own workflow there are never duplicate pipeline names because
`upload_pipeline` treats names as unique (adding versions rather than new pipelines).

> **Note**: exception types may be refined in [#458](https://github.com/kubeflow/sdk/issues/458)
(e.g. `NameResolutionError`).

##### `get_pipeline_version`

Retrieve a specific pipeline version by pipeline name and version name.

```python
v2 = client.get_pipeline_version(name="training-pipeline", version="v2-with-caching")
```

```python
def get_pipeline_version(
    self,
    name: str,
    version: str,
) -> PipelineVersion:
```

**Semantics:**
- Resolves the pipeline by `name`, then looks up the version by `version` within
  that pipeline. Raises `ValueError` if the pipeline or version is not found

##### `list_pipelines`

List pipelines available on the server. Follows `kfp.Client.list_pipelines`
pagination — callers receive a page of results plus a token to fetch the next page.

```python
# First page
response = client.list_pipelines()
for pipeline in response.pipelines:
    print(pipeline.display_name)

# Next page
response = client.list_pipelines(page_token=response.next_page_token)
```

```python
def list_pipelines(
    self,
    *,
    page_token: str = '',
    page_size: int = 10,
) -> ListPipelinesResponse:
```

`ListPipelinesResponse` is an alias for `V2beta1ListPipelinesResponse`, which has
`.pipelines: list[Pipeline]` and `.next_page_token: str`.

> **Note:** This matches `kfp.Client` pagination as a safe baseline. The implementation
> team should investigate whether a cleaner approach (e.g. `Iterator`, a `ListResult`
> wrapper, or transparent auto-paging) would be a better fit and propose an improvement
> at implementation stage.

##### `list_pipeline_versions`

List versions of a pipeline by name. Follows `kfp.Client.list_pipeline_versions`
pagination — `pipeline_id` is replaced with a name-first parameter resolved
internally.

```python
# First page
response = client.list_pipeline_versions("training-pipeline")
for version in response.pipeline_versions:
    print(version.display_name)

# Next page
response = client.list_pipeline_versions("training-pipeline", page_token=response.next_page_token)
```

```python
def list_pipeline_versions(
    self,
    name: str,
    *,
    page_token: str = '',
    page_size: int = 10,
) -> ListPipelineVersionsResponse:
```

`ListPipelineVersionsResponse` is an alias for `V2beta1ListPipelineVersionsResponse`,
with `.pipeline_versions: list[PipelineVersion]` and `.next_page_token: str`.

> **Note:** Same pagination investigation note applies as `list_pipelines` / `list_runs`.

##### `list_runs`

List runs, optionally filtered by pipeline name, experiment, or status.
Follows `kfp.Client.list_runs` pagination. Listing is essential for
rediscovering runs after a session restart (e.g. a Jupyter notebook crash).

```python
# First page
response = client.list_runs()
for run in response.runs:
    print(run.display_name)

# Next page
response = client.list_runs(page_token=response.next_page_token)

# Name-first filters (resolved to IDs internally)
response = client.list_runs(pipeline="training-pipeline")
response = client.list_runs(experiment="hyperparameter-sweep")
response = client.list_runs(pipeline="training", experiment="prod", status="succeeded")
```

```python
def list_runs(
    self,
    *,
    pipeline: str | None = None,
    experiment: str | None = None,
    status: str | None = None,
    page_token: str = '',
    page_size: int = 10,
) -> ListRunsResponse:
```

`ListRunsResponse` is an alias for `V2beta1ListRunsResponse`, which has
`.runs: list[Run]` and `.next_page_token: str`. The `pipeline`, `experiment`,
and `status` filters are resolved to IDs internally before calling the KFP API,
keeping the public API name-first while matching `kfp.Client` pagination behaviour.

> **Note:** Same as `list_pipelines` — this matches `kfp.Client` as a baseline;
a cleaner pagination approach should be investigated and proposed at implementation stage.

##### `delete_pipeline`

Delete a pipeline or a specific pipeline version. Existing runs are not
affected — they retain a snapshot of the pipeline spec they were created with.

```python
# Delete entire pipeline by name (raises if >1 version unless force=True)
client.delete_pipeline("training-pipeline")
client.delete_pipeline("training-pipeline", force=True)

# Delete a specific version by name
client.delete_pipeline("training-pipeline", version="v2-with-caching")
```

```python
def delete_pipeline(
    self,
    name: str,
    *,
    version: str | None = None,
    force: bool = False,
) -> None:
```

**Semantics:**
- `version=None` (default): deletes the entire pipeline and all its versions.
  Raises if the pipeline has more than one version unless `force=True`.
- `version="..."`: deletes only that specific version. `force` is ignored.

##### Experiment management

Full name-first CRUD for experiments is included in Phase 1.

```python
# Create
client.create_experiment("hyperparameter-sweep")

# List
response = client.list_experiments()
for exp in response.experiments:
    print(exp.display_name)

# Get
exp = client.get_experiment("hyperparameter-sweep")

# Delete
client.delete_experiment("hyperparameter-sweep")
```

```python
def create_experiment(self, name: str, *, description: str | None = None) -> Experiment: ...
def list_experiments(
    self,
    *,
    page_token: str = '',
    page_size: int = 10,
) -> ListExperimentsResponse: ...
def get_experiment(self, name: str) -> Experiment: ...
def delete_experiment(self, name: str) -> None: ...
```

`ListExperimentsResponse` is an alias for `V2beta1ListExperimentsResponse`, with
`.experiments: list[Experiment]` and `.next_page_token: str` — same pagination
pattern as `list_pipelines` and `list_runs`.

> **Note:** Pagination approach should be investigated and a cleaner design proposed
> at implementation stage (same as `list_pipelines` / `list_runs`).

> **Note**: experiment management may evolve after the RunGroups rename and MLflow
integration ([kubeflow/community#892](https://github.com/kubeflow/community/issues/892)).

##### Phase 1 summary

| Method | What it replaces in `kfp.Client` |
|---|---|
| `upload_pipeline` | `upload_pipeline`, `upload_pipeline_from_pipeline_func`, `upload_pipeline_version`, `upload_pipeline_version_from_pipeline_func` |
| `run` | `run_pipeline`, `create_run_from_pipeline_func`, `create_run_from_pipeline_package` |
| `wait_for_run_status` | `wait_for_run_completion` |
| `get_run` | `get_run` |
| `get_pipeline` | `get_pipeline` |
| `get_pipeline_version` | `get_pipeline_version` (name-first; latest if version omitted) |
| `list_pipelines` | `list_pipelines` (to be reconsidered during implementation) |
| `list_pipeline_versions` | `list_pipeline_versions` (name-first; same pagination) |
| `list_runs` | `list_runs` (to be reconsidered during implementation) |
| `delete_pipeline` | `delete_pipeline`, `delete_pipeline_version`|
| `create_experiment` | `create_experiment` |
| `list_experiments` | `list_experiments` (to be reconsidered during implementation) |
| `get_experiment` | `get_experiment` |
| `delete_experiment` | `delete_experiment` |

Fourteen methods replace 19 methods from `kfp.Client` for the core workflow.

---

## Design Details

Package layout and error tables below are guidance for reviewers; exact
module paths and exceptions may be adjusted in implementation PRs.

### Type aliases

The autogenerated KFP server API types are re-exported with clean aliases via
`pipelines/types/types.py`. All public method signatures use these aliases:

| KFP server API type | Alias |
|---|---|
| `V2beta1Pipeline` | `Pipeline` |
| `V2beta1PipelineVersion` | `PipelineVersion` |
| `V2beta1Run` | `Run` |
| `V2beta1Experiment` | `Experiment` |

```python
from kubeflow.pipelines.types import Pipeline, PipelineVersion, Run, Experiment
```

### KFP-side package structure

The new client lives in the KFP repository alongside the existing `kfp.Client`:

```
KFP repository (github.com/kubeflow/pipelines)
└── sdk/python/kfp/            # Python package root (import name: kfp)
    ├── client/
    │   └── client.py              # existing kfp.Client (unchanged)
    ├── kubeflow_client/
    │   ├── __init__.py
    │   └── pipelines_client.py    # NEW: PipelinesClient
    ├── compiler/
    ├── components/
    ├── dsl/
    └── kubernetes/
```

### SDK-side package structure

The Kubeflow SDK re-exports from `kfp.kubeflow_client`, following the same
`api/` convention as Trainer and Optimizer:

```
Kubeflow SDK repository (github.com/kubeflow/sdk)
└── kubeflow/
    ├── pipelines/
    │   ├── api/
    │   │   └── pipelines_client.py    # re-exports PipelinesClient
    │   └── __init__.py                # re-exports PipelinesClient + dsl/compiler/components/kubernetes
    ├── trainer/                   # existing (trainer/api/trainer_client.py)
    ├── optimizer/                 # existing (optimizer/api/optimizer_client.py)
    └── hub/                       # existing (ModelRegistryClient)
```

### Error Handling

| Exception | When |
|---|---|
| `ImportError` | `kfp` not installed. Message directs user to `pip install 'kubeflow[pipelines]'` |
| `ValueError` | Name resolution fails: pipeline/experiment not found, or zero versions |
| `ValueError` | Name resolution is ambiguous: multiple matches found (IDs listed in the message) |
| `ValueError` | `upload_pipeline` — callable fails to compile (invalid `@dsl.pipeline` function) |
| `ValueError` | `run` — `experiment` is a `str` but no experiment with that name exists |
| `RuntimeError` | Raised inside a `callback` passed to `wait_for_run_status` when the callback itself raises |
| `TimeoutError` | `wait_for_run_status` exceeds timeout without reaching target state |

Note: exception types may be refined in [#458](https://github.com/kubeflow/sdk/issues/458)
(e.g. `NameResolutionError`).

### Test Plan

**KFP:** unit tests against mocked internals for Phase 1 methods; E2E against
a live server when feasible (upload_pipeline → run → wait_for_run_status → get_run). Kubeflow SDK:
integration tests for re-export when `kfp` is installed, and for failure with a
clear `pip install 'kubeflow[pipelines]'` message when `kfp` is absent.
Details belong in test PRs.

---

## Implementation Plan

**Ownership:** `PipelinesClient` will be implemented under
`kfp.kubeflow_client.pipelines_client`. The Kubeflow SDK adds the `pipelines` extra,
`kubeflow.pipelines` re-exports, and tests/docs per [Design Details](#design-details)
(including the missing-`kfp` Requirement).

**What ships when:** API scope per phase is defined in [Phased API](#phased-api).

**Tasks:** Version pins, file layout, sequencing, and per-PR checklists are
worked out in implementation PRs (KFP and Kubeflow SDK), not duplicated here.

---

## Migration

### Existing `kfp.Client` users

Adoption is optional and incremental. `kfp.Client` remains fully supported.

| `kfp.Client` | `PipelinesClient` |
|---|---|
| `Client(host="...", existing_token="...")` | `PipelinesClient(backend_config=PipelinesBackendConfig(base_url="...", user_token="..."))` or `PipelinesClient()` (auto-discovery) |
| `upload_pipeline_from_pipeline_func(fn, pipeline_name="X")` | `upload_pipeline(fn, name="X")` |
| `get_pipeline_id("X")` then `run_pipeline(pipeline_id=...)` | `run("X", params={...})` |
| `create_run_from_pipeline_func(fn, ...)` | `run(fn, params={...})` (compile and submit inline) |
| `get_run(run_id)` | `get_run(run_id)` |
| `get_pipeline(pipeline_id)` | `get_pipeline("name")` |
| `get_pipeline_version(pipeline_id, version_id)` | `get_pipeline_version("name")` (latest) or `get_pipeline_version("name", version="v1")` |
| `list_pipeline_versions(pipeline_id=...)` | `list_pipeline_versions("name")` |
| `list_pipelines()` | `list_pipelines()` |
| `list_runs(experiment_id=...)` | `list_runs(experiment="name")` (name-first; same pagination) |
| `wait_for_run_completion(run_id)` | `wait_for_run_status(run)` or `wait_for_run_status(run_id)` |
| `delete_pipeline(pipeline_id)` | `delete_pipeline("name")` or `delete_pipeline("name", version="v1")` |
| `create_experiment(name)` | `create_experiment(name)` |
| `get_experiment(experiment_name=...)` | `get_experiment("name")` |
| `list_experiments()` | `list_experiments()` |
| `delete_experiment(experiment_id)` | `delete_experiment("name")` |
| `create_recurring_run(...)` | `create_recurring_run(...)` (Further phases) |

`kfp.Client` features with no `PipelinesClient` equivalent (use `kfp.Client` directly):

| `kfp.Client` method | Why not included |
|---|---|
| `archive_experiment` / `unarchive_experiment` | Rare organizational operation |
| `archive_run` / `unarchive_run` | Further phases covers `archive`; `unarchive` deferred |
| `delete_run` | Runs are historical records; deletion is uncommon |

### When KFP SDK packaging consolidation lands (kfp 3.x)

We bump to `kfp[kubernetes]>=3.0.0`. No wrapper code changes needed — the
re-export points to the same `kfp.kubeflow_client.pipelines_client` module.

---

## Implementation History

- 2025-02-18: Initial KEP creation (wrapper-in-SDK approach)
- 2026-03-24: Refactored to reflect KFP team collaboration — client in KFP
  repo, phased API, unified upload
- 2026-04: Upstream alignment — single KEP for SDK + KFP integration (no
  separate KFP-repo KEP); refactor to make the KEP more concise and easier to review
- 2026-04-27: Addressing comments and agreements during upstream call

---

## Alternatives

### Alternative 1: Wrapper client in the Kubeflow SDK repo (original approach)

The original version of this KEP proposed implementing `PipelinesClient` as a
thin wrapper around `kfp.Client` directly in the Kubeflow SDK repository.

**How it worked:**

- `PipelinesClient` lived at `kubeflow/pipelines/api/pipelines_client.py`
- Every method delegated to `kfp.Client` after resolving names to IDs
- ~30 methods covering the full `kfp.Client` surface
- Kubeflow SDK team owned the wrapper code

**Why it was superseded:** The KFP team proposed hosting the client in the KFP
repo instead. This is a higher-value approach because:

- **KFP team ownership** eliminates the maintenance burden on the SDK team and
  ensures the client stays aligned with KFP internals.
- **Freedom to implement at any level** — the KFP team can call service APIs
  directly or wrap `kfp.Client`, whichever is more reasonable. The SDK wrapper
  was constrained to wrapping `kfp.Client`'s public methods.
- **Simplified API surface** — with access to KFP internals, methods like
  `upload_pipeline` can unify four separate operations that the wrapper had to delegate
  to four different `kfp.Client` methods.
- **No upstream coordination bottleneck** for client-level features. The KFP
  team implements and releases on their own cadence.

The original wrapper approach remains a viable fallback if the KFP
implementation is significantly delayed.

### Alternative 2: Re-export `kfp.Client` as-is

Simply re-export `kfp.Client` under `kubeflow.pipelines.Client` without any
wrapping or simplification.

**Rejected:** Misses the core value — constructor alignment, name-first API,
unified `upload_pipeline`, and consistent `wait_for_run_status` semantics. Users would still deal with
`host=`, `existing_token=`, and ID-centric methods.

### Alternative 3: Migrate KFP SDK codebase into the Kubeflow SDK

Absorb the KFP SDK code (client, DSL, compiler, components) directly into the
Kubeflow SDK repository.

**Pros:**

- Single Kubeflow SDK for all components.
- Full alignment with the Trainer/Optimizer model.
- No upstream coordination bottleneck.
- Simpler dependency graph.

**Cons:**

- Requires KFP community buy-in to move development.
- Significant multi-quarter migration effort.
- Breaking change for existing `kfp` users.
- KFP SDK serves non-Kubeflow use cases (e.g. Vertex AI Pipelines).
- Decouples the KFP operator from its SDK. Other Kubeflow components
  (Trainer, Spark) already have this split, but the KFP SDK is significantly
  larger and more tightly coupled to its backend.

**Current stance:** The re-export approach delivers the "one SDK" experience
without the migration. If communities converge on a shared SDK vision in the
future, this becomes the natural next step.

### Alternative 4: Git submodule

Include the KFP SDK as a git submodule in the Kubeflow SDK repo.

**Rejected:** Submodule complexity for contributors, build/packaging friction,
two sources of truth, and it doesn't solve the API simplification problem.
Standard pip dependency management is simpler.
