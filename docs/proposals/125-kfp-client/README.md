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
  - [Escape hatch (`kfp.Client`)](#escape-hatch-kfpclient)
  - [DSL Re-exports](#dsl-re-exports)
  - [Phased API](#phased-api)
    - [Core workflow](#core-workflow)
    - [Phase 1 summary](#phase-1-summary)
- [Open Questions](#open-questions)
  - [1. Name-first resolution — client-side or server-side](#1-name-first-resolution--client-side-or-server-side)
  - [2. Task logs](#2-task-logs)
  - [3. Run events](#3-run-events)
  - [4. Runs tracked by ID](#4-runs-tracked-by-id)
  - [5. Version auto-generation strategy](#5-version-auto-generation-strategy)
  - [6. `run_pipeline()` with callable — implicit upload side-effect](#6-run_pipeline-with-callable--implicit-upload-side-effect)
  - [7. `delete_pipeline` cascading behavior](#7-delete_pipeline-cascading-behavior)
  - [8. `upload_pipeline` return type union](#8-upload_pipeline-return-type-union)
  - [9. `wait_for_run` default terminal states — `cancelled` and `raise_on_failure`](#9-wait_for_run-default-terminal-states--cancelled-and-raise_on_failure)
- [Design Details](#design-details)
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
the KFP repository at a proposed location, `kfp.kubeflow.client`, and
re-exported by the Kubeflow SDK. This means:

- **KFP team maintains the client** — KFP owns the implementation, tests, and
  releases.
- **Kubeflow SDK re-exports it** — we import from `kfp.kubeflow.client` and
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
  namespace="kubeflow")` vs `ModelRegistryClient(base_url=..., user_token=...)`.
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
4. Add `wait_for_run` with a flexible status set (matching
   `TrainerClient.wait_for_job_status`).
5. Align the constructor with `ModelRegistryClient` (`base_url`, `port`,
   `user_token`, `is_secure`, `custom_ca`) and add `namespace` for KFP's
   multi-user deployments.
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
  KFP changes (see [Open Questions](#open-questions)).
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

client = PipelinesClient("https://ml-pipeline.example.com")

# Upload (compiles automatically from function)
client.upload_pipeline(my_pipeline, name="training-pipeline")

# Run and wait inline
run = client.run_pipeline("training-pipeline", params={"epochs": 5}, timeout=3600)
print(f"Finished: {run.state}")
```

No IDs, no experiment setup, no "which upload_pipeline/run_pipeline method do I use?" decisions.

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

client = PipelinesClient("https://ml-pipeline.example.com")

# Upload once
client.upload_pipeline(training_pipeline, name="training-pipeline")

# Run with different parameters
run1 = client.run_pipeline(
    "training-pipeline",
    name="lr-0.01",
    experiment="hyperparameter-sweep",
    params={"lr": 0.01, "epochs": 20},
)

run2 = client.run_pipeline(
    "training-pipeline",
    name="lr-0.001",
    experiment="hyperparameter-sweep",
    params={"lr": 0.001, "epochs": 50},
)

# Wait for both
client.wait_for_run(run1, timeout=3600)
client.wait_for_run(run2, timeout=3600)
```

No ID juggling. Pipeline and experiment resolved by name. Latest pipeline
version used automatically.

**Experiments:** Phase 1 `run_pipeline` accepts an `experiment`
name; if no experiment with that name exists, it is auto-created, aligned
with `kfp.Client`. Further phases can add experiment management — explicit
`create_experiment`, `list_experiments`, filtering `list_runs` by experiment,
and related lifecycle operations.

### Story 3: Monitor and wait for specific states

As a data scientist, I want to wait until my run starts executing (not just
completes) so I can start tailing external logs.

```python
client = PipelinesClient("https://ml-pipeline.example.com")

run = client.run_pipeline("training-pipeline", params={"epochs": 10})

# Wait for the run to start (not complete)
running = client.wait_for_run(run, status={"running"}, timeout=120)
print(f"Run is now {running.state} — starting log tail...") # This is just illustrative

# Later, wait only for success
completed = client.wait_for_run(run, status={"succeeded"}, timeout=3600)
print(f"Finished with state: {completed.state}")
```

`kfp.Client.wait_for_run_completion` can only wait for all terminal states at
once. `wait_for_run` accepts any set of states — terminal or non-terminal.

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

client = PipelinesClient("https://ml-pipeline.example.com")
client.upload_pipeline(training_pipeline, name="distributed-training")

run = client.run_pipeline(
    "distributed-training",
    experiment="distributed-experiments",
    params={"subset_size": 5000, "num_epochs": 5},
)

completed = client.wait_for_run(run, timeout=7200)
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
LLM-style fine-tuning: **BuiltinTrainer**, **TorchTune**, **dataset and model
Initializers** (e.g. Spark output on S3 + Hugging Face base model), and remote
checkpoints.

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

client = PipelinesClient("https://ml-pipeline.example.com")

run = client.run_pipeline(
    full_pipeline,
    params={"model_name": "my-model", "version": "v2", "epochs": 10},
)

completed = client.wait_for_run(run, timeout=7200)
print(f"Pipeline finished: {completed.state}")
```

One `pip install 'kubeflow[pipelines,spark,hub]'` for this shape. `TrainerClient`
is in the base `kubeflow` package; `pipelines`, `spark`, and `hub`
are extras.

SDK surfaces: `PipelinesClient` (`run_pipeline` with a callable = compile →
upload → run per Phase 1), `kfp` DSL (DAG, `dsl.OutputPath`), and imports
for Spark, Trainer, and Model Registry.

---

## Proposal

### Architecture

The client is implemented in the **KFP repository** at `kfp.kubeflow.client`
and re-exported by the Kubeflow SDK.

```
KFP repository (github.com/kubeflow/pipelines)
└── sdk/python/kfp/            # Python package root (import name: kfp)
    ├── client/
    │   └── client.py          # existing kfp.Client (unchanged)
    └── kubeflow/
        └── client.py          # NEW: PipelinesClient (simplified API)
```

```
Kubeflow SDK repository (github.com/kubeflow/sdk)
└── kubeflow/pipelines/
    └── __init__.py            # re-exports PipelinesClient + dsl/compiler/components/kubernetes
```

```
User code
    │
    ▼
kubeflow.pipelines.PipelinesClient   (re-export)
    │
    ▼
kfp.kubeflow.client.PipelinesClient  (implementation in KFP repo)
    │
    ▼
kfp.Client / KFP service APIs       (underlying KFP internals)
    │
    ▼
KFP REST API server
```

### Dependency

On Kubeflow SDK side the pipelines optional dependency will list kfp[kubernetes] only; we will not depend on kfp[notebooks] or kfp[all] by default.

```toml
[project.optional-dependencies]
pipelines = ["kfp[kubernetes]>=X.Y.Z"]  # first kfp release shipping kfp.kubeflow.client
```

**Why `kfp[kubernetes]` only:**

KFP ships three extras (as of kfp 2.x):

- **`kfp[kubernetes]`** — lightweight. Adds K8s helpers (`mount_pvc`,
  `use_secret_as_env`, `add_node_selector`) that most pipeline users need.
- **`kfp[notebooks]`** — heavy (nbclient, ipykernel). Only needed for
  notebook-as-component use cases.
- **`kfp[all]`** — heaviest (docker + notebooks). Overkill as a default.

### Constructor

Aligned with `ModelRegistryClient` (`base_url`, `port`, `user_token`,
`is_secure`, `custom_ca` are shared). `PipelinesClient` adds `namespace` for
KFP's multi-user deployment model.

```python
PipelinesClient(
    base_url: str,
    port: int | None = None,
    *,
    user_token: str | None = None,
    is_secure: bool | None = None,
    custom_ca: str | None = None,
    namespace: str | None = None,
)
```

| Parameter | Description |
|---|---|
| `base_url` | KFP API server URL including scheme |
| `port` | Inferred from scheme if omitted (443 for https, 8080 for http). The http default is 8080 (not 80) because KFP's API server listens on 8080 by convention |
| `user_token` | Bearer token for authentication |
| `is_secure` | Inferred from scheme if omitted |
| `custom_ca` | Path to PEM-encoded root certificates |
| `namespace` | K8s namespace when needed. Default `None` (unset): no namespace is assumed here—effective behavior follows the server and deployment. |

**Namespace default:** `PipelinesClient` defaults `namespace=None` so callers who don’t care can omit it. That differs from `kfp.Client`, which commonly defaults to `"kubeflow"` on the client. Per-cluster behavior belongs in KFP/server docs and the shipping client docstring—not in this KEP.

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
- Version label omitted → auto-generate (following KFP UI conventions).
  Calling `upload_pipeline` again with the same `name` and no explicit `version`
  creates a **new version** each time (not idempotent)

```python
def upload_pipeline(
    self,
    pipeline: Callable | str,
    *,
    name: str,
    version: str | None = None,
    description: str | None = None,
) -> V2beta1Pipeline | V2beta1PipelineVersion:  # see Open Question 8
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

##### `run_pipeline`

Run a pipeline by name, or directly from a function.

```python
# Run an uploaded pipeline
run = client.run_pipeline(
    "training-pipeline",
    params={"epochs": 10, "lr": 0.001},
)

# Quick run from function — compile, upload (pipeline on server), then run
run = client.run_pipeline(
    my_pipeline,
    params={"epochs": 5},
)

# Run and wait inline (timeout makes it synchronous)
run = client.run_pipeline(
    "training-pipeline",
    params={"epochs": 10},
    timeout=3600,
)
```

```python
def run_pipeline(
    self,
    pipeline: str | Callable,
    *,
    params: dict[str, Any] | None = None,
    name: str | None = None,
    experiment: str | None = None,
    version: str | None = None,
    timeout: int | None = None,
) -> V2beta1Run:
```

| Parameter | Description |
|---|---|
| `pipeline` | Pipeline name (`str`) or `@dsl.pipeline` function — if a function, **compile → upload → run** (pipeline registered on the server; see [Open Question 6](#6-run_pipeline-with-callable--implicit-upload-side-effect)) |
| `params` | Pipeline parameters |
| `name` | Run display name (auto-generated if omitted) |
| `experiment` | Experiment name. If `None`, the server's default experiment is used (typically `"Default"`). Auto-created if it doesn't exist, matching `kfp.Client` behavior. |
| `version` | Pipeline version (latest if omitted) |
| `timeout` | If set, `run_pipeline` blocks until the run is terminal or the inner wait times out. If omitted, returns right after the run is created (usually still pending) |

`run_pipeline` + `timeout` design note: The convenience of `timeout` on `run_pipeline()` merges
“create run” and “wait for completion” into one call. Callers who inspect
`run.state` must know whether `timeout` was set — without `timeout`, the state
is typically not terminal. Prefer `run_pipeline()` then `wait_for_run()` for clearer
control flow.

When `pipeline` is a callable:   The client compiles the function, uploads
it to the API server (creating the pipeline or a new version if it already
exists, using the `@dsl.pipeline(name=...)` name), then creates a run. That
matches the “pipeline on the server + run” story without a separate `upload_pipeline`
call.

> **Note:** That implicit upload surprises some users who expected a one-off
> run with no catalog entry. `kfp.Client.create_run_from_pipeline_func`
> can submit inline spec only. See [Open Question 6](#6-run_pipeline-with-callable--implicit-upload-side-effect).

**Two workflow patterns:**

- **Upload once, run many** (Story 2): Call `upload_pipeline` separately, then `run_pipeline`
  by pipeline name. Best when you want to control versioning and trigger
  multiple runs from the same pipeline.
- **Quick one-shot**: Story 1 uses explicit `upload_pipeline` then `run_pipeline` by pipeline
  name. Story 5 passes a callable to `run_pipeline` (compile + upload + run in
  one call — pipeline ends up on the server) — fewest lines for notebooks and
  iterative development.

##### `wait_for_run`

Wait for a run to reach a target state. Accepts a run ID string or the
`V2beta1Run` object returned from `run_pipeline`.

```python
completed = client.wait_for_run(run, timeout=3600)

# Wait for a specific non-terminal state
running = client.wait_for_run(run, status={"running"}, timeout=120)

# Don't raise on failure — inspect the result instead
result = client.wait_for_run(run, raise_on_failure=False)
if result.state == "failed":
    print(f"Run failed: {result.error}")

# Also accepts a raw run ID
completed = client.wait_for_run("abc-123", timeout=3600)
```

```python
def wait_for_run(
    self,
    run: str | V2beta1Run,
    *,
    status: set[str] | None = None,
    timeout: int | None = None,
    poll_interval: int = 5,
    raise_on_failure: bool = True,
) -> V2beta1Run:
```

**Semantics**

- **`status` —** *Omitted:* stop on the first default terminal state (`succeeded`, `failed`, `skipped`, `error`, `cancelled` — so a UI cancel does not leave you polling forever). *Set* (e.g. `{"running"}`): stop when the state is in `status`, or sooner on `cancelled`, `failed`/`error`, or `timeout`.

- **`raise_on_failure` —** Applies only to `failed` and `error`. `True` (default): those raise `RuntimeError` when they end the wait. `False`: return `V2beta1Run`. `cancelled` / `skipped`: always return the run (never raised via this flag).

- **`timeout` —** If set (seconds), `TimeoutError` if the window expires before a stop condition. If omitted, poll until a stop condition.

**Example:** You asked for `{"running"}` but the run `failed` first → `raise_on_failure=True` → `RuntimeError`; `False` → inspect `.state` on the returned run.

##### `get_run`

Inspect a run by ID or by the `V2beta1Run` object returned from `run_pipeline`.

```python
run_info = client.get_run(run)  # accepts the run object directly
run_info = client.get_run("abc-123")  # or a raw run ID
```

```python
def get_run(self, run: str | V2beta1Run) -> V2beta1Run:
```

##### `get_pipeline`

Inspect a pipeline by name.

```python
pipeline = client.get_pipeline("training-pipeline")
```

```python
def get_pipeline(self, name: str) -> V2beta1Pipeline:
```

##### `list_pipelines`

List pipelines available on the server.

```python
pipelines = client.list_pipelines()
```

```python
def list_pipelines(
    self,
    *,
    limit: int = 100,
) -> list[V2beta1Pipeline]:
```

##### `list_runs`

List runs, optionally filtered by pipeline name.

```python
# All recent runs
runs = client.list_runs()

# Runs for a specific pipeline
runs = client.list_runs(pipeline="training-pipeline")
```

```python
def list_runs(
    self,
    *,
    pipeline: str | None = None,
    limit: int = 100,
) -> list[V2beta1Run]:
```

Listing is essential for rediscovering runs after a session restart (e.g. a
Jupyter notebook crash). Without it, users would need the KFP UI or a raw
`kfp.Client` call to find run IDs.

Pagination: Phase 1 list methods return a simple list with a default
`limit=100` to prevent accidentally fetching thousands of records on busy
clusters.

##### `delete_pipeline`

Delete a pipeline by name. This cascades to all pipeline versions (versions are
children of the pipeline in KFP's data model). Existing runs are not
affected — they retain a snapshot of the pipeline spec they were created with.

```python
client.delete_pipeline("training-pipeline")
```

```python
def delete_pipeline(self, name: str) -> None:
```

> **Safety consideration:** Cascading to all versions could be destructive if a
> pipeline has many versions. Consider adding a `force: bool = False` parameter
> that raises if the pipeline has more than one version unless `force=True`. See
> [Open Question 7](#7-delete_pipeline-cascading-behavior).

##### Phase 1 summary

| Method | What it replaces in `kfp.Client` |
|---|---|
| `upload_pipeline` | `upload_pipeline`, `upload_pipeline_from_pipeline_func`, `upload_pipeline_version`, `upload_pipeline_version_from_pipeline_func` |
| `run_pipeline` | `run_pipeline`, `create_run_from_pipeline_func`, `create_run_from_pipeline_package` |
| `wait_for_run` | `wait_for_run_completion` (with flexible status set) |
| `get_run` | `get_run` |
| `get_pipeline` | `get_pipeline` |
| `list_pipelines` | `list_pipelines` |
| `list_runs` | `list_runs` (with name-first pipeline filter) |
| `delete_pipeline` | `delete_pipeline` |

Eight methods replace ~14 methods from `kfp.Client` for the core workflow.

---

## Open Questions

Each item below uses the same shape so reviewers can focus on Proposal
(accept, adjust, or reject). Problem states why the question exists.

### 1. Name-first resolution — client-side or server-side

**Problem:** Kubeflow SDK clients are name-first; KFP service APIs are
ID-centric (pipelines, experiments, versions). Users should not have to
pass raw IDs for the common path.

**Proposal:** Ship client-side resolution in Phase 1 (internal lookups /
list+filter) so name-first `run_pipeline` / `upload_pipeline` work without server changes. Treat
server-side name parameters or atomic resolve-in-one-call APIs as a later
optimization when/if the KFP API supports them. Pipelines and experiments stay
name-first in the public API; runs stay ID-based after creation.

### 2. Task logs

**Problem:** `TrainerClient` exposes job logs. KFP’s Python SDK has no
first-class `get_task_logs`; logs live in Pods and the UI uses paths that
are awkward to replicate in a thin client without Kubernetes API access
(kubeconfig or in-cluster identity), which would mix auth models.

**Proposal:** Defer `PipelinesClient.logs(...)` to a later phase and/or
upstream API.

### 3. Run events

**Problem:** Same as logs: other SDKs expose K8s-style events; KFP does not
expose them in the same way, and client-side emulation has the same hybrid-auth
concerns.

**Proposal:** Defer `events(run_id)` to a later phase, same as logs.

### 4. Runs tracked by ID

**Problem:** Other SDK surfaces are name-centric for “jobs,” but KFP run
display names are not unique, and runs are not always tied to a single
pipeline version—so there is no stable natural key besides **`run_id`**.
Recurring runs will have the same issue.

**Proposal:** Expose runs by `run_id` (and accept
`V2beta1Run` objects in `wait_for_run` / `get_run` so callers rarely touch raw IDs in
the same session). Use `list_runs` to rediscover runs after restarts.

### 5. Version auto-generation strategy

**Problem:** When `upload_pipeline` is called without an explicit `version` and
the pipeline already exists, the server needs a version label — either
auto-generated or user-supplied.

**Proposal:** Auto-generate new version labels using the same conventions
as the KFP UI for auto-generated names, with an optional explicit `version`
parameter for users who want full control.

### 6. `run_pipeline()` with callable — implicit upload side-effect

**Problem:** For `run_pipeline(callable)`, the implementation can (**a**) submit an
inline pipeline spec with no persistent pipeline resource (like
`kfp.Client.create_run_from_pipeline_func` today), or (**b**) compile →
register on the server → run, which matches “pipeline in the catalog” and reuse
by name but can feel like a hidden upload.

**Proposal:** Adopt (**b**) for Phase 1: `compile → upload (register) → run`
so the pipeline appears in the UI and can be rerun by name. Users who want
inline-only behavior use `kfp.Client` or we revisit in a follow-up if
demand is high.

### 7. `delete_pipeline` cascading behavior

**Problem:** `delete_pipeline("training-pipeline")` may cascade to all versions,
which is destructive if many versions exist.

**Proposal:** Cascade delete all versions for the pipeline name (as in the
Phase 1 sketch). Add `force: bool = False`: when `False` and the pipeline
has more than one version, raise; require `force=True` to confirm
multi-version delete. (Implementation PR can tune messaging.)

### 8. `upload_pipeline` return type union

**Problem:** Returning `V2beta1Pipeline | V2beta1PipelineVersion` forces
callers to narrow types.

**Proposal:** Prefer a single concrete type where possible—e.g. always return
`V2beta1PipelineVersion` (creating a pipeline implies a first version), or
a small dataclass (`pipeline_id`, `version_id`, `name`) built from API
responses. Final shape aligned with KFP maintainers in implementation.

### 9. `wait_for_run` default terminal states — `cancelled` and `raise_on_failure`

**Problem:** If `cancelled` is not treated as terminal, `wait_for_run` may poll
forever after a user cancels a run. `raise_on_failure` must not treat user
cancel like `failed`.

**Proposal:** Include `cancelled` in the default terminal set (together
with `succeeded`, `failed`, `skipped`, `error`—exact strings per
target KFP version). `cancelled` and `skipped`: return the run
object; do not raise via `raise_on_failure`. `raise_on_failure`
applies to `failed` / `error` only. Confirm whether `skipped` should
ever raise under `raise_on_failure=True` for specific deployments (optional
narrowing in implementation).

---

## Design Details

Package layout and error tables below are guidance for reviewers; exact
module paths and exceptions may be adjusted in implementation PRs.

### KFP-side package structure

The new client lives in the KFP repository alongside the existing `kfp.Client`:

```
KFP repository (github.com/kubeflow/pipelines)
└── sdk/python/kfp/            # Python package root (import name: kfp)
    ├── client/
    │   └── client.py              # existing kfp.Client (unchanged)
    ├── kubeflow/
    │   ├── __init__.py
    │   └── client.py              # NEW: PipelinesClient
    ├── compiler/
    ├── components/
    ├── dsl/
    └── kubernetes/
```

### SDK-side package structure

The Kubeflow SDK re-exports from `kfp.kubeflow`:

```
Kubeflow SDK repository (github.com/kubeflow/sdk)
└── kubeflow/
    ├── pipelines/
    │   └── __init__.py            # re-exports PipelinesClient + dsl/compiler/components/kubernetes
    ├── trainer/                   # existing
    ├── optimizer/                 # existing
    └── hub/                       # existing (ModelRegistryClient)
```

### Error Handling

| Exception | When |
|---|---|
| `ImportError` | `kfp` not installed. Message directs user to `pip install 'kubeflow[pipelines]'` |
| `ValueError` | Name resolution fails (pipeline/experiment not found, no versions) |
| `ValueError` | `upload_pipeline` — callable fails to compile (invalid `@dsl.pipeline` function) |
| `RuntimeError` | `wait_for_run` — run reaches `failed` or `error` while `raise_on_failure=True` (default). `cancelled` / `skipped` return the run; they do not raise |
| `TimeoutError` | `wait_for_run` exceeds timeout without reaching target state |

### Test Plan

**KFP:** unit tests against mocked internals for Phase 1 methods; E2E against
a live server when feasible (upload_pipeline → run_pipeline → wait_for_run → get_run). Kubeflow SDK:
integration tests for re-export when `kfp` is installed, and for failure with a
clear `pip install 'kubeflow[pipelines]'` message when `kfp` is absent.
Details belong in test PRs.

---

## Implementation Plan

**Ownership:** The KFP team implements `PipelinesClient` under
`kfp.kubeflow.client`. The Kubeflow SDK adds the `pipelines` extra,
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
| `Client(host="...", existing_token="...")` | `PipelinesClient(base_url="...", user_token="...")` |
| `upload_pipeline_from_pipeline_func(fn, pipeline_name="X")` | `upload_pipeline(fn, name="X")` |
| `get_pipeline_id("X")` then `run_pipeline(pipeline_id=...)` | `run_pipeline("X", params={...})` |
| `get_run(run_id)` | `get_run(run_id)` or `get_run(run)` |
| `get_pipeline(pipeline_id)` | `get_pipeline("name")` |
| `list_pipelines()` | `list_pipelines()` |
| `list_runs(experiment_id=...)` | `list_runs(pipeline="name")` |
| `wait_for_run_completion(run_id)` | `wait_for_run(run)` or `wait_for_run(run_id)` |
| `delete_pipeline(pipeline_id)` | `delete_pipeline("name")` |
| `create_recurring_run(...)` | `create_recurring_run(...)` (Further phases) |

`kfp.Client` features with no `PipelinesClient` equivalent (use `kfp.Client` directly):

| `kfp.Client` method | Why not included |
|---|---|
| `archive_experiment` / `unarchive_experiment` | Rare organizational operation |
| `delete_experiment` | Rare — experiments are lightweight metadata |
| `archive_run` / `unarchive_run` | Further phases covers `archive`; `unarchive` deferred |
| `delete_run` | Runs are historical records; deletion is uncommon |
| `get_pipeline_version` / `delete_pipeline_version` | Version-level operations deferred to further phases |

### When KFP SDK packaging consolidation lands (kfp 3.x)

We bump to `kfp[kubernetes]>=3.0.0`. No wrapper code changes needed — the
re-export points to the same `kfp.kubeflow.client` module.

---

## Implementation History

- 2025-02-18: Initial KEP creation (wrapper-in-SDK approach)
- 2026-03-24: Refactored to reflect KFP team collaboration — client in KFP
  repo, phased API, unified upload
- 2026-04: Upstream alignment — single KEP for SDK + KFP integration (no
  separate KFP-repo KEP); refactor to make the KEP more concise and easier to review

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
unified `upload_pipeline`, and consistent `wait_for_run` semantics. Users would still deal with
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
