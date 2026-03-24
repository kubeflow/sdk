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
  - [Story 5: Multi-component pipeline — Spark, Trainer, and Model Registry (simplified)](#story-5-multi-component-pipeline--spark-trainer-and-model-registry-simplified)
  - [Story 6: Advanced — TorchTune, initializers, and remote checkpoints](#story-6-advanced--torchtune-initializers-and-remote-checkpoints)
- [Proposal](#proposal)
  - [Architecture](#architecture)
  - [Dependency](#dependency)
  - [Constructor](#constructor)
  - [Deviations from Other SDK Clients](#deviations-from-other-sdk-clients)
  - [DSL Re-exports](#dsl-re-exports)
  - [Phased API](#phased-api)
    - [Phase 1: Core workflow](#phase-1-core-workflow)
    - [Phase 2: Experiments, scheduling, and run lifecycle](#phase-2-experiments-scheduling-and-run-lifecycle)
    - [Phase 3: Health, observability, and upstream enhancements](#phase-3-health-observability-and-upstream-enhancements)
  - [Name Resolution](#name-resolution)
  - [Current kfp.Client API Reference](#current-kfpclient-api-reference)
- [Open Questions](#open-questions)
  - [1. Name-first resolution — client-side or server-side](#1-name-first-resolution--client-side-or-server-side)
  - [2. Task logs](#2-task-logs)
  - [3. Run events](#3-run-events)
  - [4. Runs tracked by ID (decided)](#4-runs-tracked-by-id-decided)
  - [5. Version auto-generation strategy](#5-version-auto-generation-strategy)
  - [6. `run()` with callable — implicit upload side-effect](#6-run-with-callable--implicit-upload-side-effect)
  - [7. `delete` cascading behavior](#7-delete-cascading-behavior)
  - [8. `upload` return type union](#8-upload-return-type-union)
  - [9. `wait` default terminal states — `cancelled` and `raise_on_failure`](#9-wait-default-terminal-states--cancelled-and-raise_on_failure)
- [Design Details](#design-details)
  - [KFP-side package structure](#kfp-side-package-structure)
  - [SDK-side package structure](#sdk-side-package-structure)
  - [Error Handling](#error-handling)
  - [Test Plan](#test-plan)
- [Implementation Plan](#implementation-plan)
  - [SDK-side implementation](#sdk-side-implementation)
  - [KFP-side implementation (proposed)](#kfp-side-implementation-proposed)
- [Implementation History](#implementation-history)
- [Migration](#migration)
- [Alternatives](#alternatives)

## Summary

Add a `PipelinesClient` to the Kubeflow SDK that gives users the full
author → compile → upload → run → monitor pipeline workflow from a single
`kubeflow` import.

Following discussions with the KFP team, the client will be **implemented in
the KFP repository** at a proposed location, **`kfp.kubeflow.client`**, and
**re-exported** by the Kubeflow SDK. This means:

- **KFP team maintains the client** — they own the implementation, tests, and
  releases.
- **Kubeflow SDK re-exports it** — we import from `kfp.kubeflow.client` and
  expose it at `kubeflow.pipelines.PipelinesClient`.
- **The client is additive** — `kfp.Client` remains fully supported. The new
  client provides a simplified, name-first API alongside it.

The API is delivered in phases, starting with the smallest set of methods that
cover the core upload → run → monitor → clean up workflow, expanding
incrementally as users adopt it.

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
2. Provide a **simplified, name-first API** that unifies upload variants and
   reduces the number of methods users need to learn.
3. Re-export `kfp.dsl`, `kfp.compiler`, `kfp.components`, and `kfp.kubernetes`
   at the `kubeflow.pipelines` module level.
4. Add `wait` with a flexible status set (matching
   `TrainerClient.wait_for_job_status`).
5. Align the constructor with `ModelRegistryClient` (`base_url`, `port`,
   `user_token`, `is_secure`, `custom_ca`) and add `namespace` for KFP's
   multi-user deployments.
6. Deliver the API in **phases** — Phase 1 covers the core workflow with
   minimal methods; later phases add management, observability, and advanced
   features.
7. Have the client **implemented in the KFP repository** and re-exported by
   the Kubeflow SDK, ensuring KFP team ownership and reducing SDK maintenance
   burden.

### Non-Goals

- Replacing `kfp.Client`. It remains fully supported. Power users who need
  advanced auth (IAP, cookies, proxy) or raw generated APIs continue to use
  `kfp` directly.
- Wrapping the DSL. `dsl`, `compiler`, `components`, and `kubernetes` are
  re-exported as-is — no additional abstraction layer.
- Supporting KFP control-plane provisioning. This client is data-plane only.
- Implementing task logs or run events in early phases. These require upstream
  KFP changes (see [Open Questions](#open-questions)).
- Building the KFP-side KEP. This KEP covers the SDK perspective. A separate
  KEP against the KFP repo will propose the `kfp.kubeflow.client`
  implementation.
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
client.upload(my_pipeline, name="training-pipeline")

# Run and wait inline
run = client.run("training-pipeline", params={"epochs": 5}, timeout=3600)
print(f"Finished: {run.state}")
```

No IDs, no experiment setup, no "which upload/run method do I use?" decisions.

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
client.upload(training_pipeline, name="training-pipeline")

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
client.wait(run1, timeout=3600)
client.wait(run2, timeout=3600)
```

No ID juggling. Pipeline and experiment resolved by name. Latest pipeline
version used automatically.

**Experiments (Phase 1 vs Phase 2):** Phase 1 `run` accepts an `experiment`
name; if no experiment with that name exists, it is **auto-created**, aligned
with `kfp.Client`. Phase 2 adds **experiment management** — explicit
`create_experiment`, `list_experiments`, filtering `list_runs` by experiment,
and related lifecycle operations. That split keeps Phase 1 focused on the core
upload → run workflow while still allowing named grouping of runs. The later
phase benefits operators and power users who need visibility (what experiments
exist), governance (create or archive before large campaigns), and
cross-run discovery without relying on the KFP UI alone.

### Story 3: Monitor and wait for specific states

As a data scientist, I want to wait until my run starts executing (not just
completes) so I can start tailing external logs.

```python
client = PipelinesClient("https://ml-pipeline.example.com")

run = client.run("training-pipeline", params={"epochs": 10})

# Wait for the run to start (not complete)
running = client.wait(run, status={"running"}, timeout=120)
print(f"Run is now {running.state} — starting log tail...") # This is just illustrative

# Later, wait only for success
completed = client.wait(run, status={"succeeded"}, timeout=3600)
print(f"Finished with state: {completed.state}")
```

`kfp.Client.wait_for_run_completion` can only wait for all terminal states at
once. `wait` accepts any set of states — terminal or non-terminal — matching
the `TrainerClient.wait_for_job_status(name, status={...})` pattern.

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
client.upload(training_pipeline, name="distributed-training")

run = client.run(
    "distributed-training",
    experiment="distributed-experiments",
    params={"subset_size": 5000, "num_epochs": 5},
)

completed = client.wait(run, timeout=7200)
print(f"Run finished: {completed.state}")
```

`kfp.kubernetes` is re-exported at `kubeflow.pipelines.kubernetes` so users
never need a direct `kfp` import. **`kubernetes.mount_pvc` /
`kubernetes.use_secret_as_env`** compose by fluent chaining (each helper
returns the `PipelineTask` it configures).

**Trainer `Initializer` vs a pipeline download step:** Initializers **pull**
dataset (or model) artifacts into the TrainJob from remote URIs (`s3://`,
`hf://`, …). They replace **remote fetch inside the TrainJob**, not arbitrary
in-pipeline logic. This story keeps **`load_dataset`** as a **pipeline step**
because it performs custom preprocessing (subset, format) and writes to a
shared PVC for the training step. If a job only needs “download this S3/HF
prefix as-is,” `train(..., initializer=Initializer(dataset=...))` can be used
instead and a separate download component omitted.

### Story 5: Multi-component pipeline — Spark, Trainer, and Model Registry (simplified)

As an ML engineer, I want **`PipelinesClient`** to orchestrate **one pipeline**
that chains **Spark → Trainer → Model Registry**.

Component bodies are **stubbed** on purpose, to simplify the example. **Story 6**
fills in **BuiltinTrainer**, **TorchTune**, and **Initializers** for LLM-style
fine-tuning.

```python
from pathlib import Path

from kubeflow.pipelines import PipelinesClient, dsl

@dsl.component(base_image="quay.io/my-org/spark-image:latest")
def preprocess_data(input_path: str, output_path: str):
    """Preprocess with Spark (SparkClient + SparkConnect)."""
    from kubeflow.spark import SparkClient

    # ... SparkConnect session: read/write, then spark.stop() — see Spark SDK docs ...
    pass


@dsl.component(base_image="quay.io/my-org/training-image:latest")
def train_model(
    data_path: str,
    epochs: int,
    trained_model_uri: dsl.OutputPath(str),
):
    """Submit a TrainJob (TrainerClient). Use CustomTrainer or Story 6’s BuiltinTrainer."""
    from kubeflow.trainer import TrainerClient
    from kubeflow.trainer.types import CustomTrainer

    def train_fn():
        pass  # Replace with real training; data_path is Spark output (or PVC).

    trainer = TrainerClient()
    job_name = trainer.train(trainer=CustomTrainer(func=train_fn))
    trainer.wait_for_job_status(job_name)

    # KFP OutputPath: write the URI string to disk so the compiler wires it
    # to downstream tasks (required KFP plumbing — cannot be stubbed).
    uri = f"s3://my-org-models/{job_name}/checkpoint"
    Path(trained_model_uri).parent.mkdir(parents=True, exist_ok=True)
    Path(trained_model_uri).write_text(uri)


@dsl.component(base_image="quay.io/my-org/registry-image:latest")
def register_model(
    model_name: str,
    version: str,
    trained_model_uri: str,
):
    """Register in Model Registry (ModelRegistryClient)."""
    from kubeflow.hub import ModelRegistryClient

    registry = ModelRegistryClient(
        "https://registry.example.com",
        author="pipeline",
    )
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

run = client.run(
    full_pipeline,
    params={"model_name": "my-model", "version": "v2", "epochs": 10},
)

completed = client.wait(run, timeout=7200)
print(f"Pipeline finished: {completed.state}")
```

One `pip install 'kubeflow[pipelines,spark,hub]'` for this shape. **`TrainerClient`**
is in the base `kubeflow` package; **`pipelines`**, **`spark`**, and **`hub`**
are extras.

**SDK surfaces:** `PipelinesClient` (**`run`** with a **callable** = compile →
upload → run per Phase 1), **`kfp` DSL** (DAG, `dsl.OutputPath`), and **imports**
for Spark, Trainer, and Model Registry.

### Story 6: Advanced — TorchTune, initializers, and remote checkpoints

Same **preprocess → train → register** DAG as Story 5, with **full** Trainer
wiring for **LLM-style fine-tuning**: **BuiltinTrainer** + **TorchTune** +
**dataset and model Initializers** (e.g. Spark output on S3 + Hugging Face base
model). Refer to this story if Story 5's stubs need concrete detail.

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

run = client.run(
    full_pipeline,
    params={"model_name": "my-model", "version": "v2", "epochs": 10},
)

completed = client.wait(run, timeout=7200)
print(f"Pipeline finished: {completed.state}")
```

See Story 5 for the simplified version of this pattern. The pipeline definition
and `PipelinesClient` usage are identical — only the component bodies differ.

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

**Why this architecture:**

- **KFP team owns the implementation.** They know the internals, the API
  server, and the service APIs best. They maintain and release it.
- **Kubeflow SDK is a thin re-export.** Minimal maintenance — no wrapper code,
  no mapping layers, no risk of drifting from upstream.
- **Additive.** `kfp.Client` remains unchanged. The new client exists alongside
  it at `kfp.kubeflow.client`.
- **The KFP team has freedom** to implement at whatever level makes sense —
  wrapping `kfp.Client` or calling service APIs directly — whichever is more
  reasonable for the phased approach.

### Dependency

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
| `namespace` | K8s namespace for multi-user deployments. Default `None` — omit sending a client-side default so behavior follows **server** and **deployment** configuration (see below) |

**Namespace default — `None` instead of `"kubeflow"`:** KFP's raw client
defaults to `"kubeflow"`, which might be confusing for single-user deployments. We
default to `None` so users who don't care about namespaces simply omit the
parameter. With `namespace=None`, **what the API server uses** depends on how KFP
is deployed — e.g. **multi-user** mode may tie runs to the **authenticated user’s
namespace**; **single-user** profiles may treat a missing namespace like the
server’s configured default (often `kubeflow` in older docs). Consult the **KFP
deployment guide** for the target cluster rather than assuming one global rule.

### Deviations from Other SDK Clients

| Capability | TrainerClient | ModelRegistryClient | PipelinesClient | Gap? |
|---|---|---|---|---|
| Name-centric identity | `name` everywhere | `name` everywhere | `name` for create; `run_id` after | Yes — runs |
| Logs | `get_job_logs(name)` | N/A | Phase 3 | Yes |
| Events | `get_job_events(name)` | N/A | Phase 3 | Yes |
| Wait with status set | `wait_for_job_status(name, status={...})` | N/A | `wait(run, status={...})` | Parity (except name vs ID) |
| Wait raise on failure | Raises on failure | N/A | `raise_on_failure=True` (default) | Parity with Trainer |
| Return types | Hand-crafted Pydantic | Passthrough (`model_registry.types`) | Passthrough (`kfp_server_api.V2beta1*`) | Same as registry |
| Auth model | K8s API | REST API | REST API | Aligned with registry |
| Constructor | `backend_config=` | `base_url, port, user_token, ...` | `base_url, port, user_token, ...` + `namespace` | Aligned with registry |

### Escape hatch (`kfp.Client`)

`PipelinesClient` does not try to wrap every `kfp.Client` knob (IAP, cookies,
proxy, raw generated APIs, etc.). **Power users** either construct a dedicated
`kfp.Client(...)` alongside `PipelinesClient`, or use a **`.kfp_client`**
property on `PipelinesClient` to access the underlying `kfp.Client` instance
without duplicating connection configuration. The exact accessor name is
subject to change during KFP-side implementation.

### DSL Re-exports

`kfp.dsl`, `kfp.compiler`, `kfp.components`, and `kfp.kubernetes` are
re-exported at module level with zero wrapping:

```python
from kubeflow.pipelines import PipelinesClient, dsl, compiler, components, kubernetes
```

The DSL is KFP's domain-specific language — it cannot be meaningfully wrapped.
Re-exporting gives users a single namespace for the entire
author → configure → upload → run flow without any direct `kfp` imports.

If `kfp` is not installed, avoid binding `PipelinesClient = None` (which would
make `from kubeflow.pipelines import PipelinesClient` silently give `None`).
Instead: attempt the import; on failure, leave the names undefined and use a
module-level `__getattr__` so the first access to any export raises an
actionable `ImportError`. See the concrete pattern in
[SDK-side package structure](#sdk-side-package-structure).

### Phased API

The API is delivered in phases. Phase 1 covers the core workflow with a minimal
set of methods. Each subsequent phase is additive and non-breaking.

#### Phase 1: Core workflow

**Goal:** Earn user buy-in with a simple yet powerful API. Eight methods +
constructor cover: connect → upload → run → monitor → list → inspect → clean up.

##### `upload`

Unified upload that handles functions, files, new pipelines, and new versions.

```python
# From a @dsl.pipeline function — auto-compiles
client.upload(my_pipeline, name="training-pipeline")

# From a compiled YAML file
client.upload("training-pipeline.yaml", name="training-pipeline")

# New version — same method, auto-detects existing pipeline
client.upload(my_pipeline_v2, name="training-pipeline", version="v2-with-caching")
```

The user's intent is always "put this pipeline on the server". The client
handles the implementation details:

- First arg is callable → compile it
- First arg is a string path → use the file
- Pipeline with that name exists → create new version
- Pipeline doesn't exist → create it
- Version label omitted → auto-generate (following KFP UI conventions).
  Calling `upload` again with the same `name` and no explicit `version`
  creates a **new version** each time (not idempotent)

```python
def upload(
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
| Upload from function | `upload_pipeline_from_pipeline_func` | `upload(fn, name=...)` |
| Upload from file | `upload_pipeline` | `upload("file.yaml", name=...)` |
| New version from function | `upload_pipeline_version_from_pipeline_func` | `upload(fn, name=..., version=...)` |
| New version from file | `upload_pipeline_version` | `upload("file.yaml", name=..., version=...)` |

##### `run`

Run a pipeline by name, or directly from a function.

```python
# Run an uploaded pipeline
run = client.run(
    "training-pipeline",
    params={"epochs": 10, "lr": 0.001},
)

# Quick run from function — compile, upload (pipeline on server), then run
run = client.run(
    my_pipeline,
    params={"epochs": 5},
)

# Run and wait inline (timeout makes it synchronous)
run = client.run(
    "training-pipeline",
    params={"epochs": 10},
    timeout=3600,
)
```

```python
def run(
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
| `pipeline` | Pipeline name (`str`) or `@dsl.pipeline` function — if a function, **compile → upload → run** (pipeline registered on the server; see [Open Question 6](#6-run-with-callable--implicit-upload-side-effect)) |
| `params` | Pipeline parameters |
| `name` | Run display name (auto-generated if omitted) |
| `experiment` | Experiment name. If `None`, the server's default experiment is used (typically `"Default"`). Auto-created if it doesn't exist, matching `kfp.Client` behavior. |
| `version` | Pipeline version (latest if omitted) |
| `timeout` | If set, `run` blocks until the run is terminal or the inner wait times out. If omitted, returns right after the run is **created** (usually still pending) |

**`run` + `timeout` design note:** The convenience of `timeout` on `run()` merges
“create run” and “wait for completion” into one call. Callers who inspect
`run.state` must know whether `timeout` was set — without `timeout`, the state
is typically **not** terminal. Prefer **`run()` then `wait()`** for clearer
control flow (see [Story 1](#story-1-quick-one-off-run) for `timeout` on `run` vs explicit `wait`).

**When `pipeline` is a callable:** The client compiles the function, **uploads**
it to the API server (creating the pipeline or a **new version** if it already
exists, using the `@dsl.pipeline(name=...)` name), then creates a run. That
matches the “pipeline on the server + run” story without a separate `upload`
call.

> **Note:** That **implicit upload** surprises some users who expected a one-off
> run with **no** catalog entry. **`kfp.Client.create_run_from_pipeline_func`**
> can submit **inline spec** only. See [Open Question 6](#6-run-with-callable--implicit-upload-side-effect).

**Two workflow patterns:**

- **Upload once, run many** (Story 2): Call `upload` separately, then `run`
  by pipeline name. Best when you want to control versioning and trigger
  multiple runs from the same pipeline.
- **Quick one-shot**: Story 1 uses explicit `upload` then `run` **by pipeline
  name**. Stories **5 and 6** pass a **callable** to `run` (compile + upload + run
  in one call — pipeline ends up on the server) — fewest lines for notebooks and
  iterative development.

**Recurring runs are separate** (Phase 2). Preference is to keep
recurring runs distinct because they are a one-time configuration operation
(set up a schedule), not a common interactive workflow. They also have a
different return type (`V2beta1RecurringRun` vs `V2beta1Run`) and different
lifecycle.

##### `wait`

Wait for a run to reach a target state. Accepts a run ID string or the
`V2beta1Run` object returned from `run`.

```python
completed = client.wait(run, timeout=3600)

# Wait for a specific non-terminal state
running = client.wait(run, status={"running"}, timeout=120)

# Don't raise on failure — inspect the result instead
result = client.wait(run, raise_on_failure=False)
if result.state == "failed":
    print(f"Run failed: {result.error}")

# Also accepts a raw run ID
completed = client.wait("abc-123", timeout=3600)
```

```python
def wait(
    self,
    run: str | V2beta1Run,
    *,
    status: set[str] | None = None,
    timeout: int | None = None,
    poll_interval: int = 5,
    raise_on_failure: bool = True,
) -> V2beta1Run:
```

**Time limit:** If `timeout` is `None`, polling runs until a stop condition below
is met. If `timeout` is a number of seconds, **`TimeoutError`** is raised when the
wait does not finish in time (see end of this section).

**At a glance**

| Knob | Role |
|------|------|
| `status` | Omit → stop on **any default terminal** state. Set → stop when run state **`status`** (e.g. `{"running"}`, `{"succeeded"}`). |
| `raise_on_failure` | Only affects **`failed`** and **`error`**: raise `RuntimeError` vs return the run for inspection. |
| `timeout` | Optional wall-clock limit; can raise **`TimeoutError`**. |

**1. When does polling stop?**

- **`status` omitted:** Stop when the run reaches **any** of the default terminal
  states: `succeeded`, `failed`, `skipped`, `error`, `cancelled`.
- **`status` set** (e.g. `{"running"}`): Stop when the run’s state is **in
  `status`**, **or** when the waiter must end anyway — e.g. **`cancelled`** (so
  a user cancel does not leave you polling forever), **`failed` / `error`**
  (see `raise_on_failure`), or **`timeout`**.

Including **`cancelled`** in the default terminal set avoids polling forever
after a user cancels a run in the UI. See
[Open Question 9](#9-wait-default-terminal-states--cancelled-and-raise_on_failure)
for **`skipped`** and deployment-specific expectations.

**2. After stop: raise or return? (`raise_on_failure`)**

`raise_on_failure` does **not** turn `cancelled` or `skipped` into exceptions;
those always return a `V2beta1Run` so you can branch on `.state`. It only
governs **`failed`** and **`error`**.

| Situation | `raise_on_failure=True` (default) | `raise_on_failure=False` |
|-----------|-----------------------------------|---------------------------|
| Run reaches **`failed`** or **`error`** while the requested `status` is **not** yet satisfied (e.g. you asked for `{"running"}` but the run failed first) | Raises **`RuntimeError`** | Returns the run |
| Run reaches **`cancelled`** or **`skipped`** | Returns the run (never treated as `raise_on_failure`) | Returns the run |

**3. `TimeoutError`**

Raised when `timeout` is set and the waiter does not finish within the window
— typically because the run never reached the requested `status` and did not
otherwise stop in a way that ended the wait. If `raise_on_failure=True`, a
**`failed`** / **`error`** run usually ends the wait earlier via `RuntimeError`
instead of waiting for the full timeout.

Set **`raise_on_failure=False`** to always get a `V2beta1Run` back for
**`failed`** / **`error`** (and inspect `.state` yourself), including when you
omit `timeout` or use a long window.

##### `get_run`

Inspect a run by ID or by the `V2beta1Run` object returned from `run`.

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

**Pagination:** Phase 1 list methods return a simple list with a default
`limit=100` to prevent accidentally fetching thousands of records on busy
clusters. Full pagination support (`page_size`, `page_token`) will be added in
Phase 2 alongside the other list methods (`list_experiments`,
`list_versions`).

##### `delete`

Delete a pipeline by name. This cascades to all pipeline versions (versions are
children of the pipeline in KFP's data model). Existing runs are **not**
affected — they retain a snapshot of the pipeline spec they were created with.

```python
client.delete("training-pipeline")
```

```python
def delete(self, name: str) -> None:
```

> **Safety consideration:** Cascading to all versions could be destructive if a
> pipeline has many versions. Consider adding a `force: bool = False` parameter
> that raises if the pipeline has more than one version unless `force=True`. See
> [Open Question 7](#7-delete-cascading-behavior).

##### Phase 1 summary

| Method | What it replaces in `kfp.Client` |
|---|---|
| `upload` | `upload_pipeline`, `upload_pipeline_from_pipeline_func`, `upload_pipeline_version`, `upload_pipeline_version_from_pipeline_func` |
| `run` | `run_pipeline`, `create_run_from_pipeline_func`, `create_run_from_pipeline_package` |
| `wait` | `wait_for_run_completion` (with flexible status set) |
| `get_run` | `get_run` |
| `get_pipeline` | `get_pipeline` |
| `list_pipelines` | `list_pipelines` |
| `list_runs` | `list_runs` (with name-first pipeline filter) |
| `delete` | `delete_pipeline` |

Eight methods replace ~14 methods from `kfp.Client` for the core workflow.

---

#### Phase 2: Experiments, scheduling, and run lifecycle

For users who need experiment organization, recurring runs, and run lifecycle
management beyond the core workflow.

```python
# Experiments
client.create_experiment("hyperparameter-sweep")
experiments = client.list_experiments()

# Filter runs by experiment
runs = client.list_runs(experiment="hyperparameter-sweep")

# Run lifecycle
client.terminate(run_id)
client.archive(run_id)

# Pipeline versions
versions = client.list_versions("training-pipeline")

# Recurring runs (separate from one-off runs per KFP team preference)
client.create_recurring_run(
    pipeline="training-pipeline",
    name="nightly-retrain",
    experiment="production",
    cron_expression="0 0 * * *",
    params={"data_path": "/data/latest"},
)
client.list_recurring_runs()
client.disable_recurring_run(recurring_run_id)
client.enable_recurring_run(recurring_run_id)
client.delete_recurring_run(recurring_run_id)
```

**Why Phase 2:** Experiment management, recurring runs, and run lifecycle
(terminate, archive) are organizational operations. Phase 1 covers the full
create → run → list → inspect → delete workflow. Phase 2 adds the tooling to
manage runs and pipelines at scale.

---

#### Phase 3: Health, observability, and upstream enhancements

Health checks, namespace management, and features that may require KFP API
server changes or deeper integration.

```python
# Server health
health = client.health()

# Namespace management (multi-user)
client.set_namespace("team-ml")
ns = client.get_namespace()

# Upstream enhancements (require KFP API server changes)
logs = client.logs(run_id, task="train")
events = client.events(run_id)
completed = client.wait(run_id, callbacks=[my_callback])
```

**Upstream enhancements:**

1. **Task logs** — `client.logs(run_id, task="train")`
2. **Run events** — `client.events(run_id)`
3. **Callbacks on wait** — `client.wait(run_id, callbacks=[my_callback])`
4. **Server-side name resolution** — eliminates extra API calls
    from client-side name → ID resolution

**Why Phase 3:** Health checks and namespace management are infrastructure
concerns most users never need. The upstream enhancements depend on KFP API
server changes that are outside the client's control.

---

### Name Resolution

The new client uses name-first parameters (`pipeline="training-pipeline"`,
`experiment="hyperparameter-sweep"`) while the underlying KFP APIs are
ID-centric.

Name → ID resolution will be handled internally. The exact approach — whether
client-side (extra API calls) or server-side (new API parameters) — is an
[open question](#1-name-first-resolution--client-side-or-server-side) to be
resolved during the KFP-side KEP.

**Client-side approach** (current implementation):

```python
# Internal helper — not public API
def _resolve_pipeline_id(self, name):
    resolved = self._client.get_pipeline_id(name)
    if resolved is None:
        raise ValueError(f"Pipeline {name!r} not found.")
    return resolved
```

Each name-first call triggers 1-3 additional API calls for resolution (pipeline
name → ID, experiment name → ID, latest version lookup). This works but adds
round-trips.

**Server-side approach** (ideal, requires API server changes):

REST API endpoints accept `display_name` parameters directly, resolving names
atomically in a single request. Eliminates extra round-trips and race
conditions.

### Current kfp.Client API Reference

For context, the full `kfp.Client` API that the new client simplifies:

| Category | `kfp.Client` methods |
|---|---|
| **Health/Context** | `get_kfp_healthz`, `set_user_namespace`, `get_user_namespace` |
| **Experiments** | `create_experiment`, `get_experiment`, `list_experiments`, `get_pipeline_id`, `archive_experiment`, `unarchive_experiment`, `delete_experiment` |
| **Pipelines** | `get_pipeline`, `list_pipelines`, `delete_pipeline`, `list_pipeline_versions`, `get_pipeline_version`, `delete_pipeline_version` |
| **Upload** | `upload_pipeline`, `upload_pipeline_from_pipeline_func`, `upload_pipeline_version`, `upload_pipeline_version_from_pipeline_func` |
| **Runs** | `run_pipeline`, `create_run_from_pipeline_func`, `create_run_from_pipeline_package`, `get_run`, `list_runs`, `wait_for_run_completion`, `archive_run`, `unarchive_run`, `delete_run`, `terminate_run` |
| **Recurring Runs** | `create_recurring_run`, `get_recurring_run`, `list_recurring_runs`, `delete_recurring_run`, `enable_recurring_run`, `disable_recurring_run` |

~30 methods total. The new client's Phase 1 covers the core workflow with 8
methods, growing incrementally through later phases.

---

## Open Questions

### 1. Name-first resolution — client-side or server-side

**Context:** The Kubeflow SDK pattern is name-first (Trainer, Model Registry,
Optimizer all use names). KFP APIs are ID-centric. Name-first resolution is
important and agreed upon.

**Client-side:** Extra API calls per request. Works today without API server changes.

**Server-side:** Atomic resolution. No extra round-trips. Requires API server
changes.

**Decision needed:** To be resolved during the KFP-side KEP.

### 2. Task logs

**Gap:** `TrainerClient` exposes `get_job_logs(name)`. KFP's Python SDK has no
`get_task_logs` — logs live in Kubernetes pods, and the KFP UI fetches them via
the K8s API behind the scenes.

**Proposal:** Out of scope for Phase 1-2. Propose to KFP as a native
`logs(run_id, task_name)` method in Phase 3. Implementing at the client level
would require K8s API access (kubeconfig or in-cluster service account),
creating a hybrid auth model.

### 3. Run events

**Gap:** `TrainerClient.get_job_events(name)` and
`OptimizerClient.get_job_events(name)` expose K8s events. KFP's API server does
not surface them. Same hybrid auth concern as logs.

**Proposal:** Out of scope for Phase 1-2. Same approach as logs — propose
upstream in Phase 3.

### 4. Runs tracked by ID

**Gap:** Other SDK clients are fully name-centric. `PipelinesClient` tracks
runs by `run_id` after creation because KFP run display names are not unique.

**Decision:** Runs remain ID-based. There is no clean unique key for runs —
display names are not unique, and runs are not always tied to a specific
pipeline version. This is a fundamental difference from Kubernetes resources
(where `metadata.name` is unique within a namespace) because KFP runs are
database records with auto-generated UUIDs.

The same applies to **recurring runs** (Phase 2). Methods like
`disable_recurring_run(recurring_run_id)` and
`delete_recurring_run(recurring_run_id)` use IDs for the same reason —
recurring run display names are not unique in KFP. Users discover recurring
run IDs via `list_recurring_runs()`, which returns objects carrying the ID.

Phase 1 mitigates the ID requirement through:
- **`list_runs`** — rediscover runs after a session restart (e.g. Jupyter
  notebook crash) without needing to remember UUIDs.
- **Accepting the `V2beta1Run` object** in `wait` and `get_run` — so in the
  common same-session workflow, users never need to extract `.run_id` manually.

### 5. Version auto-generation strategy

**Context:** When `upload` is called without an explicit `version` parameter
and the pipeline already exists, a version label needs to be auto-generated.

**Proposal:** Follow the approach used by the KFP UI for auto-generated
names, with the option for users to manually set the version label.

**Decision needed:** Adopt the KFP UI–style auto-versioning described above,
or replace it with an explicit alternative.

### 6. `run()` with callable — implicit upload side-effect

**Problem:** For **`run(callable)`**, we can either (1) **only** start a run
with an **inline** pipeline spec, or (2) **also** create/update a **pipeline on
the server** and then run. (2) matches this KEP’s Phase 1 flow (pipelines show up
in the UI; reuse by name is easy) but can feel like a hidden **upload**. (1)
matches **`kfp.Client.create_run_from_pipeline_func`** — no new pipeline
resource unless the user **uploads** separately.

**Proposal (this KEP):** Same as Phase 1 **`run`** above: **compile → upload →
run** — register the pipeline (or a new version), then create the run.

**Decision needed:** **Adopt** that proposal **or** default **`run(callable)`**
to **inline-spec-only** runs (and require **`upload` + `run(name)`** when users
want a server-side pipeline).

### 7. `delete` cascading behavior

**Context:** `delete("training-pipeline")` cascades to all pipeline versions.
A user could accidentally delete a pipeline with many versions and no way to
recover them.

**Options:**
- Add `force: bool = False` — raises if the pipeline has more than one version
  unless `force=True`.
- Always cascade (current proposal) but log a warning showing the number of
  versions being deleted.

**Decision needed:** Determine whether a `force` guard or a confirmation
mechanism is warranted.

### 8. `upload` return type union

**Context:** `upload` returns `V2beta1Pipeline | V2beta1PipelineVersion`.
Union return types are harder for callers to use without type-narrowing.

**Options:**
- Always return `V2beta1PipelineVersion` — a new pipeline implicitly creates
  its first version, so a version object is always available.
- Return a simpler result type (e.g. a dataclass with `pipeline_id`,
  `version_id`, `name`).

**Decision needed:** Align with the KFP team on whether a consistent return
type is feasible.

### 9. `wait` default terminal states — `cancelled` and `raise_on_failure`

**Context:** KFP runs can be cancelled (`CANCELED` state). If `cancelled` is
not in the default terminal set, `wait()` would poll indefinitely when a run
is cancelled mid-flight.

**Proposed resolution:** Include `cancelled` in the default terminal state set
(`succeeded`, `failed`, `skipped`, `error`, `cancelled`) so polling stops.
Also confirm the exact state string used by the target KFP version (`CANCELED` vs
`cancelled`).

**`raise_on_failure` vs `cancelled`:** User- or operator-initiated **cancel** is
not necessarily a “failure” for exception semantics. The KEP proposes that
**`cancelled`** (and likely **`skipped`**) **return** the terminal run rather
than raising `RuntimeError`, while **`failed` / `error`** raise when
`raise_on_failure=True`.

**Decision needed:** Confirm with the KFP team whether **`skipped`** should ever
raise under `raise_on_failure=True` (deployment-specific expectations).

---

## Design Details

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

Same [Architecture](#architecture): one physical prefix **`sdk/python/`**, then
the **`kfp`** package as installed / imported.

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

```python
# kubeflow/pipelines/__init__.py
_KFP_EXPORTS = frozenset(
    {"PipelinesClient", "dsl", "compiler", "components", "kubernetes"}
)

try:
    from kfp.kubeflow.client import PipelinesClient
    from kfp import compiler, components, dsl, kubernetes
except ImportError:
    _KFP_IMPORT_FAILED = True
else:
    _KFP_IMPORT_FAILED = False

__all__ = list(_KFP_EXPORTS)

def __getattr__(name):
    # Only fires for names not already bound at module level.
    # On import failure the names are never defined, so access triggers this.
    if name in _KFP_EXPORTS and _KFP_IMPORT_FAILED:
        raise ImportError(
            f"'{name}' requires kfp. Install it with: "
            f"pip install 'kubeflow[pipelines]'"
        ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Comparison with Trainer / Model Registry clients: see
[Deviations from Other SDK Clients](#deviations-from-other-sdk-clients) under Proposal.

### Error Handling

| Exception | When |
|---|---|
| `ImportError` | `kfp` not installed. Message directs user to `pip install 'kubeflow[pipelines]'` |
| `ValueError` | Name resolution fails (pipeline/experiment not found, no versions) |
| `ValueError` | `upload` — callable fails to compile (invalid `@dsl.pipeline` function) |
| `RuntimeError` | `wait` — run reaches **`failed`** or **`error`** while `raise_on_failure=True` (default). **`cancelled`** / **`skipped`** return the run; they do not raise |
| `TimeoutError` | `wait` exceeds timeout without reaching target state |

### Test Plan

- **Unit tests (KFP repo):** All `PipelinesClient` methods tested against
  mocked KFP internals. Covers name resolution, auto-version resolution,
  unified upload logic, and error cases.
- **E2E tests (KFP repo):** Against a live KFP server if possible. Cover
  the full upload → run → wait → get flow.
- **Integration tests (Kubeflow SDK repository):** Verify re-export works correctly.
  `from kubeflow.pipelines import PipelinesClient` resolves to the KFP
  implementation.

---

## Implementation Plan

### SDK-side implementation

The Kubeflow SDK work is minimal — it's a re-export.

1. Add `pipelines = ["kfp[kubernetes]>=X.Y.Z"]` to `pyproject.toml` (pinned to
   the `kfp` version that includes `kfp.kubeflow.client`)
2. Create `kubeflow/pipelines/__init__.py` with re-export of `PipelinesClient`
   and DSL re-exports (`dsl`, `compiler`, `components`, `kubernetes`)
3. Integration tests verifying re-export
4. Documentation on sdk.kubeflow.org

### KFP-side implementation (proposed)

The KFP team owns this implementation but of course, we can contribute. This section outlines what we propose
for a separate KEP against the KFP repo.

**Phase 1: Core workflow**

1. Create `sdk/python/kfp/kubeflow/client.py` with `PipelinesClient` class
2. Implement constructor with `base_url`→`host` mapping, `user_token`→`existing_token`,
   `namespace` defaulting to `None`
3. Implement unified `upload` — detect callable vs path, detect new vs existing
   pipeline, auto-generate version label following KFP UI conventions
4. Implement `run` — name-first pipeline and experiment resolution, optional
   `timeout` for inline wait, support for passing `@dsl.pipeline` function
   directly (**compile → upload → run**; see [Open Question 6](#6-run-with-callable--implicit-upload-side-effect))
5. Implement `wait` — flexible status set, client-side polling
6. Implement `get_run` (by run ID or `V2beta1Run` object) and `get_pipeline` (by name)
7. Implement `list_pipelines` and `list_runs` (with optional pipeline name filter)
8. Implement `delete` — by pipeline name
9. Unit tests for all Phase 1 methods

**Phase 2: Experiments, scheduling, and run lifecycle**

1. Implement experiment management (`create_experiment`, `list_experiments`)
2. Add experiment filter to `list_runs`
3. Implement run lifecycle (`terminate`, `archive`)
4. Implement `list_versions` for pipeline version listing
5. Implement recurring run methods (`create_recurring_run`, `list_recurring_runs`,
   `disable_recurring_run`, `enable_recurring_run`, `delete_recurring_run`)
6. Unit tests

**Phase 3: Health, observability, and upstream enhancements**

1. Implement `health`
2. Implement `set_namespace`, `get_namespace`
3. Task logs — `logs(run_id, task_name)` (requires KFP API server changes)
4. Run events — `events(run_id)` (requires KFP API server changes)
5. Callbacks on `wait`
6. Server-side name resolution (if pursued)
7. E2E tests
8. Documentation

---

## Implementation History

- 2025-02-18: Initial KEP creation (wrapper-in-SDK approach)
- 2026-03-24: Refactored to reflect KFP team collaboration — client in KFP
  repo, phased API, unified upload

---

## Migration

### Existing `kfp.Client` users

Adoption is optional and incremental. `kfp.Client` remains fully supported.

| `kfp.Client` | `PipelinesClient` |
|---|---|
| `Client(host="...", existing_token="...")` | `PipelinesClient(base_url="...", user_token="...")` |
| `upload_pipeline_from_pipeline_func(fn, pipeline_name="X")` | `upload(fn, name="X")` |
| `get_pipeline_id("X")` then `run_pipeline(pipeline_id=...)` | `run("X", params={...})` |
| `get_run(run_id)` | `get_run(run_id)` or `get_run(run)` |
| `get_pipeline(pipeline_id)` | `get_pipeline("name")` |
| `list_pipelines()` | `list_pipelines()` |
| `list_runs(experiment_id=...)` | `list_runs(pipeline="name")` |
| `wait_for_run_completion(run_id)` | `wait(run)` or `wait(run_id)` |
| `delete_pipeline(pipeline_id)` | `delete("name")` |
| `create_recurring_run(...)` | `create_recurring_run(...)` (Phase 2) |

**`kfp.Client` features with no `PipelinesClient` equivalent (use `kfp.Client` directly):**

| `kfp.Client` method | Why not included |
|---|---|
| `archive_experiment` / `unarchive_experiment` | Rare organizational operation |
| `delete_experiment` | Rare — experiments are lightweight metadata |
| `archive_run` / `unarchive_run` | Phase 2 covers `archive`; `unarchive` deferred |
| `delete_run` | Runs are historical records; deletion is uncommon |
| `get_pipeline_version` / `delete_pipeline_version` | Version-level operations deferred to Phase 2+ |

### When KFP SDK packaging consolidation lands (kfp 3.x)

We bump to `kfp[kubernetes]>=3.0.0`. No wrapper code changes needed — the
re-export points to the same `kfp.kubeflow.client` module.

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
  `upload` can unify four separate operations that the wrapper had to delegate
  to four different `kfp.Client` methods.
- **No upstream coordination bottleneck** for client-level features. The KFP
  team implements and releases on their own cadence.

The original wrapper approach remains a viable fallback if the KFP-side
implementation is significantly delayed.

### Alternative 2: Re-export `kfp.Client` as-is

Simply re-export `kfp.Client` under `kubeflow.pipelines.Client` without any
wrapping or simplification.

**Rejected:** Misses the core value — constructor alignment, name-first API,
unified upload, and consistent wait semantics. Users would still deal with
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
