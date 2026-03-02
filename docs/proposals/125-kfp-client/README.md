# KEP-125: PipelinesClient for Kubeflow SDK

|                     |                                             |
| ------------------- | ------------------------------------------- |
| **Authors**         | [MStokluska](https://github.com/MStokluska) |
| **Created**         | 2026-03-02                                  |
| **Relevant Issues** | https://github.com/kubeflow/sdk/issues/125  |

## Table of Contents

- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [User Stories](#user-stories)
  - [Story 1: Quick one-off run](#story-1-quick-one-off-run-simplest-path)
  - [Story 2: Upload once, run many times](#story-2-upload-once-run-many-times)
  - [Story 3: Production recurring schedule](#story-3-production-recurring-schedule)
  - [Story 4: Monitoring and waiting for specific states](#story-4-monitoring-and-waiting-for-specific-states)
  - [Story 5: Kubernetes-native pipeline with PVC and secrets](#story-5-kubernetes-native-pipeline-with-pvc-and-secrets)
  - [Story 6: Pipeline versioning](#story-6-pipeline-versioning)
  - [Story 7: End-to-end ML workflow — Pipelines + Trainer + Model Registry](#story-7-end-to-end-ml-workflow--pipelines--trainer--model-registry)
- [Proposal](#proposal)
  - [Architecture](#architecture)
  - [Dependency](#dependency)
  - [Constructor](#constructor)
  - [DSL Re-exports](#dsl-re-exports)
  - [Exposed API](#exposed-api)
  - [Deferred APIs](#deferred-apis)
  - [Name Resolution Internals](#name-resolution-internals)
- [Open Questions](#open-questions)
  - [1. Task logs](#1-task-logs)
  - [2. Run events](#2-run-events)
  - [3. Runs tracked by ID after creation](#3-runs-tracked-by-id-after-creation)
  - [4. wait_for_run_status callbacks](#4-wait_for_run_status-callbacks)
  - [5. Return types — passthrough vs hand-crafted](#5-return-types--passthrough-vs-hand-crafted)
  - [6. Upstream KFP improvements](#6-upstream-kfp-improvements)
  - [7. Ownership and long-term maintenance](#7-ownership-and-long-term-maintenance)
- [Design Details](#design-details)
  - [Package Structure](#package-structure)
  - [Deviations from Other SDK Clients](#deviations-from-other-sdk-clients)
  - [Error Handling](#error-handling)
  - [Test Plan](#test-plan)
- [Implementation Plan](#implementation-plan)
- [Migration](#migration)
- [Alternatives](#alternatives)

## Summary

Add a `PipelinesClient` to the Kubeflow SDK that wraps `kfp.Client`, giving
users the full author → compile → upload → run → monitor pipeline workflow from
a single `kubeflow` import. The client follows the same patterns as
`ModelRegistryClient` (constructor, auth, naming conventions) and re-exports
KFP's DSL, compiler, and components so users never need to mix `kfp` and
`kubeflow` imports.

```bash
pip install 'kubeflow[pipelines]'
```

```python
from kubeflow.pipelines import PipelinesClient, dsl, compiler, components, kubernetes
```

## Motivation

Today, Kubeflow users who want to orchestrate ML pipelines must install and use
`kfp` separately from the Kubeflow SDK. This creates friction:

- **Two SDKs, two import styles** — users mix `from kfp import ...` with `from kubeflow.trainer import ...`.
- **Inconsistent constructor** — `kfp.Client(host=..., existing_token=..., namespace="kubeflow")` vs `ModelRegistryClient(base_url=..., user_token=...)`.
- **ID-centric API** — `kfp.Client` requires users to look up pipeline IDs and experiment IDs before they can trigger a run. Other SDK clients are name-first.
- **Scattered wait semantics** — `kfp.Client.wait_for_run_completion` hardcodes terminal states and cannot wait for intermediate states like `Running`.

### Goals

1. Expose KFP pipeline management through the Kubeflow SDK with `pip install 'kubeflow[pipelines]'`.
2. Provide a name-first API for experiments, pipelines, and pipeline versions — matching the conventions of `ModelRegistryClient`, `TrainerClient`, and `OptimizerClient`.
3. Re-export `kfp.dsl`, `kfp.compiler`, `kfp.components`, and `kfp.kubernetes` at the `kubeflow.pipelines` module level.
4. Add `wait_for_run_status` with a flexible status set (matching `TrainerClient.wait_for_job_status`).
5. Align the constructor with `ModelRegistryClient` (`base_url`, `port`, `user_token`, `is_secure`, `custom_ca`) and add `namespace` for KFP's multi-user deployments.

### Non-Goals

- Replacing `kfp.Client`. Power users who need advanced auth (IAP, cookies, proxy) or raw generated APIs continue to use `kfp` directly.
- Wrapping the DSL. `dsl`, `compiler`, `components`, and `kubernetes` are re-exported as-is — no additional abstraction layer.
- Supporting KFP control-plane provisioning. This client is data-plane only, same as `kfp.Client`.
- Implementing task logs or run events in v1. These are deferred (see [Open Questions](#open-questions)).

---

## User Stories

### Story 1: Quick one-off run (simplest path)

As a data scientist, I want to define a simple pipeline, upload it, and run it
in as few lines as possible — without dealing with IDs, versions, or experiment
setup.

```python
from kubeflow.pipelines import PipelinesClient, dsl

@dsl.component
def train(epochs: int) -> str:
    return f"trained for {epochs} epochs"

@dsl.pipeline
def my_pipeline(epochs: int = 10):
    train(epochs=epochs)

client = PipelinesClient("https://ml-pipeline.example.com")

run = client.run_pipeline_from_func(my_pipeline, arguments={"epochs": 5})
print(f"Run started: {run.run_id}")
```

`run_pipeline_from_func` compiles, uploads to the default experiment, and
triggers a run in one call. No experiment creation, no pipeline ID lookup.

### Story 2: Upload once, run many times

As an ML engineer, I want to upload a pipeline once, then trigger runs with
different parameters — referring to the pipeline by name, not by ID.

```python
from kubeflow.pipelines import PipelinesClient, dsl, compiler

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
client.upload_pipeline_from_pipeline_func(
    training_pipeline,
    pipeline_name="training-pipeline",
)

# Create experiment
client.create_experiment("hyperparameter-sweep")

# Run with different parameters — pipeline and experiment resolved by name
run1 = client.run_pipeline(
    job_name="lr-0.01",
    experiment_name="hyperparameter-sweep",
    pipeline_name="training-pipeline",
    params={"lr": 0.01, "epochs": 20},
)

run2 = client.run_pipeline(
    job_name="lr-0.001",
    experiment_name="hyperparameter-sweep",
    pipeline_name="training-pipeline",
    params={"lr": 0.001, "epochs": 50},
)

# Wait for both — uses run ID returned from run_pipeline
client.wait_for_run_status(run1.run_id)
client.wait_for_run_status(run2.run_id)
```

No ID juggling — `pipeline_name` and `experiment_name` are resolved internally.
The latest pipeline version (latest uploaded pipeline) is used automatically; pass `version_id=` to pin a
specific version.

### Story 3: Production recurring schedule

As a platform engineer, I want a pipeline to run every night at midnight and
handle failures gracefully.

```python
from kubeflow.pipelines import PipelinesClient

client = PipelinesClient(
    "https://ml-pipeline.example.com",
    user_token="eyJ...",
)

client.create_recurring_run(
    job_name="nightly-retrain",
    experiment_name="production",
    pipeline_name="training-pipeline",
    cron_expression="0 0 * * *",
    params={"data_path": "/data/latest", "epochs": 30},
    max_concurrency=1,
    no_catchup=True,
)
```

### Story 4: Monitoring and waiting for specific states

As a data scientist, I want to wait until my run starts executing (not just
completes) so I can start tailing external logs.

```python
from kubeflow.pipelines import PipelinesClient

client = PipelinesClient("https://ml-pipeline.example.com")

run = client.run_pipeline(
    job_name="train-run",
    experiment_name="default",
    pipeline_name="training-pipeline",
)

# Wait for the run to start (not complete)
running = client.wait_for_run_status(
    run.run_id,
    status={"running"},
    timeout=120,
)
print(f"Run is now {running.state} — starting log tail...")

# Later, wait only for success (not failed/skipped/error)
completed = client.wait_for_run_status(
    run.run_id,
    status={"succeeded"},
    timeout=3600,
)
print(f"Finished with state: {completed.state}")
```

KFP's built-in `wait_for_run_completion` can only wait for all terminal states
at once (`succeeded`, `failed`, `skipped`, `error`). `wait_for_run_status`
accepts any set of states — terminal or non-terminal — matching the
`TrainerClient.wait_for_job_status(name, status={...})` pattern.

### Story 5: Kubernetes-native pipeline with PVC and secrets

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
def submit_training(model_path: str, num_epochs: int = 3, nproc_per_node: int = 1):
    from kubeflow.trainer import TrainerClient
    client = TrainerClient()
    client.train(...)
    client.wait_for_job_status("train-job")

@dsl.pipeline(name="distributed-training")
def training_pipeline(
    subset_size: int = 1000,
    model_path: str = "/mnt/shared/model",
    num_epochs: int = 3,
):
    download_task = download_dataset(output_dir="/mnt/shared/data", subset_size=subset_size)
    kubernetes.mount_pvc(download_task, pvc_name="shared-pvc", mount_path="/mnt/shared")

    train_task = submit_training(model_path=model_path, num_epochs=num_epochs)
    kubernetes.mount_pvc(train_task, pvc_name="shared-pvc", mount_path="/mnt/shared")
    kubernetes.use_secret_as_env(train_task, secret_name="k8s-creds",
                                 secret_key_to_env={"token": "K8S_TOKEN"})
    train_task.after(download_task)

client = PipelinesClient("https://ml-pipeline.example.com")
pipeline = client.upload_pipeline_from_pipeline_func(
    training_pipeline, pipeline_name="distributed-training"
)

run = client.run_pipeline(
    job_name="training-v1",
    experiment_name="distributed-experiments",
    pipeline_name="distributed-training",
    params={"subset_size": 5000, "num_epochs": 5},
)

completed = client.wait_for_run_status(run.run_id, timeout=7200)
print(f"Run finished: {completed.state}")
```

The `kfp.kubernetes` helpers (`mount_pvc`, `use_secret_as_env`,
`add_node_selector`, etc.) are available because `kubeflow[pipelines]` depends
on `kfp[kubernetes]`.

`kfp.kubernetes` is re-exported at `kubeflow.pipelines.kubernetes` so users
never need a direct `kfp` import — even for K8s-specific helpers like
`mount_pvc` and `use_secret_as_env`.

### Story 6: Pipeline versioning

As a platform engineer, I want to upload a new version of an existing pipeline
and pin runs to a specific version.

```python
client = PipelinesClient("https://ml-pipeline.example.com")

# Upload v2 of an existing pipeline
client.upload_pipeline_version_from_pipeline_func(
    training_pipeline_v2,
    pipeline_version_name="v2-with-caching",
    pipeline_name="training-pipeline",
)

# List versions
versions = client.list_pipeline_versions("training-pipeline")
for v in versions.pipeline_versions:
    print(f"  {v.display_name} — {v.pipeline_version_id}")

# Run pinned to a specific version
run = client.run_pipeline(
    job_name="pinned-run",
    experiment_name="production",
    pipeline_name="training-pipeline",
    version_id="abc-123-specific-version",
)
```

When `version_id` is omitted, the latest version is resolved automatically.
Pass it explicitly to pin.

### Story 7: End-to-end ML workflow — Pipelines + Trainer + Model Registry

As an ML engineer, I want a single pipeline that trains a model with the
Kubeflow Trainer, waits for it to finish, and registers the result in Model
Registry — all from one SDK.

```python
from kubeflow.pipelines import PipelinesClient, dsl

@dsl.component(base_image="quay.io/my-org/training-image:latest")
def train_and_register(model_name: str, version: str, epochs: int = 5):
    from kubeflow.trainer import TrainerClient
    from kubeflow.hub import ModelRegistryClient

    # Train
    trainer = TrainerClient()
    trainer.train(trainer=..., options=...)
    trainer.wait_for_job_status("train-job")

    # Register the trained model
    registry = ModelRegistryClient("https://registry.example.com")
    registry.register_model(
        name=model_name,
        uri="s3://models/trained-model",
        version=version,
        model_format_name="pytorch",
    )

@dsl.pipeline(name="train-and-register")
def train_and_register_pipeline(
    model_name: str = "my-model",
    version: str = "v1",
    epochs: int = 5,
):
    train_and_register(model_name=model_name, version=version, epochs=epochs)

client = PipelinesClient("https://ml-pipeline.example.com")

run = client.run_pipeline_from_func(
    train_and_register_pipeline,
    arguments={"model_name": "my-model", "version": "v2", "epochs": 10},
)

completed = client.wait_for_run_status(run.run_id, timeout=3600)
print(f"Pipeline finished: {completed.state}")
```

One `pip install 'kubeflow[pipelines,hub]'`, one `kubeflow` namespace.
`TrainerClient` is included in the base `kubeflow` package — only `pipelines`
and `hub` need extras. The pipeline orchestrates training and model registration
while the outer script only interacts with `PipelinesClient`.

---

## Proposal

### Architecture

`PipelinesClient` is a thin wrapper around `kfp.Client`. It does not duplicate
any KFP logic — every method delegates to the underlying client after resolving
names to IDs where needed.

```
kubeflow.pipelines
├── __init__.py          # exports PipelinesClient, dsl, compiler, components, kubernetes
└── api/
    └── pipelines_client.py   # PipelinesClient wrapping kfp.Client
```

```
User code
    │
    ▼
PipelinesClient  ──►  kfp.Client  ──►  KFP REST API server
    │
    └── name → ID resolution (experiments, pipelines, versions)
    └── wait_for_run_status (client-side polling with flexible states)
    └── constructor alignment (base_url, user_token, ...)
```

### Dependency

```toml
[project.optional-dependencies]
pipelines = ["kfp[kubernetes]>=2.0.0"]
```

**Why `kfp[kubernetes]` only:**

KFP ships three extras (as of kfp 2.x):

```python
extras_require = {
    'all': docker + kubernetes + notebooks,
    'kubernetes': kubernetes,
    'notebooks': notebooks,
}
```

- **`kfp[kubernetes]`** — lightweight (only `kfp` + `protobuf`). Adds K8s
  helpers (`mount_pvc`, `use_secret_as_env`, `add_node_selector`) that most
  pipeline users need.
- **`kfp[notebooks]`** — heavy (nbclient, ipykernel, jupyter_client). Only
  needed for notebook-as-component use cases.
- **`kfp[all]`** — heaviest (docker + notebooks). Overkill as a default.

Users who need notebook components can `pip install kfp[notebooks]` on top.

After the proposed KFP SDK packaging consolidation (tracked as KEP-12548,
draft), we bump to e.g. `kfp[kubernetes]>=3.0.0` and re-evaluate.

### Constructor

Mostly aligned with `ModelRegistryClient` (`base_url`, `port`, `user_token`,
`is_secure`, `custom_ca` are shared). `PipelinesClient` adds `namespace` for
KFP's multi-user deployment model. `ModelRegistryClient` has an `author`
parameter that does not apply here.

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
| `port` | Inferred from scheme if omitted (443 for https, 8080 for http) |
| `user_token` | Bearer token for authentication |
| `is_secure` | Inferred from scheme if omitted |
| `custom_ca` | Path to PEM-encoded root certificates |
| `namespace` | K8s namespace for multi-user deployments. Default `None` — lets the server decide |

**Namespace default — `None` instead of `"kubeflow"`:** KFP's raw client
defaults to `"kubeflow"`, which is confusing for single-user deployments. We
default to `None` so users who don't care about namespaces simply omit the
parameter.

```python
# Single-user — no namespace needed
client = PipelinesClient("https://ml-pipeline.example.com")

# Multi-user — explicit namespace
client = PipelinesClient("https://ml-pipeline.example.com", namespace="team-ml")

# Self-signed certs
client = PipelinesClient(
    "https://ml-pipeline.internal",
    user_token="eyJ...",
    custom_ca="/etc/ssl/certs/internal-ca.pem",
)
```

### DSL Re-exports

`kfp.dsl`, `kfp.compiler`, `kfp.components`, and `kfp.kubernetes` are
re-exported at module level with zero wrapping:

```python
from kubeflow.pipelines import PipelinesClient, dsl, compiler, components, kubernetes

@dsl.component
def my_step(): ...

@dsl.pipeline
def my_pipeline(): ...

compiler.Compiler().compile(my_pipeline, "pipeline.yaml")
```

The DSL is KFP's domain-specific language — it cannot be meaningfully wrapped.
Re-exporting (including `kfp.kubernetes`) gives users a single namespace for
the entire author → configure → upload → run flow without any direct `kfp`
imports.

If `kfp` is not installed, the re-exports are silently unavailable. Attempting
`from kubeflow.pipelines import dsl` without kfp installed produces an
`ImportError` — the message should be improved to suggest
`pip install 'kubeflow[pipelines]'`.

### Exposed API

#### Health and Context

```python
client.get_health()                 # server reachable? multi-user mode?
client.set_user_namespace("team-b") # switch namespace mid-session
client.get_user_namespace()         # current namespace
```

`get_health` is renamed from KFP's `get_kfp_healthz` for readability.
`set_user_namespace` / `get_user_namespace` keep KFP's names unchanged — they
allow switching namespace mid-session without recreating the client.

#### Experiments (name-first)

```python
exp = client.create_experiment("my-experiment")
exp = client.get_experiment("my-experiment")
# or by ID:
exp = client.get_experiment(experiment_id="abc-123")

exps = client.list_experiments(page_size=20)
client.archive_experiment("my-experiment")
client.unarchive_experiment("my-experiment")
client.delete_experiment("my-experiment")
```

All experiment methods accept `name` as the primary parameter with `experiment_id`
as a keyword fallback. The wrapper resolves names to IDs before delegating.

#### Pipelines (name-first)

```python
pipelines = client.list_pipelines(page_size=20)
p = client.get_pipeline("training-pipeline")
client.delete_pipeline("training-pipeline")

versions = client.list_pipeline_versions("training-pipeline")
v = client.get_pipeline_version("version-id", name="training-pipeline")
client.delete_pipeline_version("version-id", name="training-pipeline")
```

#### Upload

Four methods covering the compile-and-upload matrix:

```python
# From compiled YAML/JSON file
client.upload_pipeline("pipeline.yaml", pipeline_name="my-pipeline")
client.upload_pipeline_version("pipeline.yaml", "v2", pipeline_name="my-pipeline")

# From @dsl.pipeline function — compile + upload in one step
client.upload_pipeline_from_pipeline_func(my_pipeline, pipeline_name="my-pipeline")
client.upload_pipeline_version_from_pipeline_func(my_pipeline, "v2", pipeline_name="my-pipeline")
```

#### Runs (name-first creation, ID-based tracking)

**Creating a run** is name-first:

```python
run = client.run_pipeline(
    job_name="run-1",
    experiment_name="default",
    pipeline_name="training-pipeline",
    params={"epochs": 10},
)
```

- `experiment_name` and `pipeline_name` are resolved to IDs internally.
- Latest pipeline version is used automatically. Pass `version_id=` to pin.
- Returns a `V2beta1Run` object with `run.run_id`.

**After creation**, runs are tracked by `run_id`:

```python
run = client.get_run(run.run_id)
client.wait_for_run_status(run.run_id)
client.archive_run(run.run_id)
client.unarchive_run(run.run_id)
client.delete_run(run.run_id)
client.terminate_run(run.run_id)
runs = client.list_runs(experiment_id=exp.experiment_id)
```

**Why ID-based after creation:** KFP run display names are not unique. Multiple
runs can share the same `job_name`. There is no reliable name → run mapping, unlike
`TrainerClient` where TrainJob names are unique K8s resource names. See
[Open Question 3](#3-runs-tracked-by-id-after-creation) for alternatives
considered.

**Convenience one-shot methods:**

```python
# Compile + run in one step (no separate upload)
result = client.run_pipeline_from_func(my_pipeline, arguments={"epochs": 5})
result = client.run_pipeline_from_package("pipeline.yaml", arguments={"epochs": 5})
```

> **Note:** These return KFP's `RunPipelineResult` (not `V2beta1Run`), since
> they delegate to `kfp.Client.create_run_from_pipeline_func` /
> `create_run_from_pipeline_package` which handle compilation internally.

#### Recurring Runs

Same name-first creation pattern:

```python
recurring = client.create_recurring_run(
    job_name="nightly-retrain",
    experiment_name="production",
    pipeline_name="training-pipeline",
    cron_expression="0 0 * * *",
    params={"data_path": "/data/latest"},
)

# Managed by recurring_run_id after creation
client.get_recurring_run(recurring.recurring_run_id)
client.disable_recurring_run(recurring.recurring_run_id)
client.enable_recurring_run(recurring.recurring_run_id)
client.delete_recurring_run(recurring.recurring_run_id)
client.list_recurring_runs()
```

#### `wait_for_run_status`

```python
def wait_for_run_status(
    self,
    run_id: str,
    timeout: int = 600,
    polling_interval: int = 5,
    status: set[str] | None = None,
) -> V2beta1Run:
```

Default: waits for any terminal state (`succeeded`, `failed`, `skipped`, `error`).
Pass a custom `status` set to wait for specific states:

```python
# Wait for run to start executing
client.wait_for_run_status(run.run_id, status={"running"}, timeout=120)

# Wait only for success (raise TimeoutError if it fails)
client.wait_for_run_status(run.run_id, status={"succeeded"}, timeout=3600)
```

This matches `TrainerClient.wait_for_job_status(name, status={...})` and
`OptimizerClient.wait_for_job_status(name, status={...})`.

### Deferred APIs

| What | Why |
|---|---|
| Generated server APIs (`client.pipelines.*`, `client.runs.*`, etc.) | Our high-level methods call them under the hood; exposing raw layer would duplicate functionality |
| `pipeline_spec` | Internal protobuf IR. Not needed for the normal DSL → compile → upload → run flow |
| Legacy aliases (`delete_job`, `enable_job`, `disable_job`) | Old names for `*_recurring_run` methods. We unify naming |
| Advanced auth (`cookies`, `client_id`, `proxy`, `kube_context`, etc.) | Niche. We cover `user_token`, `is_secure`, `custom_ca`. Rest available via `kfp.Client` |
| UI / notebook helpers (URL printing, IPython display links) | Environment-specific. Can be added later if demanded |

### Name Resolution Internals

The wrapper has three internal helpers that make the name-first API work:

```python
def _resolve_pipeline_id(self, name, pipeline_id):
    """Resolve pipeline name → ID using kfp.Client.get_pipeline_id()."""

def _resolve_experiment_id(self, name, experiment_id, namespace=None):
    """Resolve experiment name → ID using kfp.Client.get_experiment(experiment_name=...)."""

def _resolve_latest_version_id(self, pipeline_id):
    """Fetch the most recent pipeline version using list_pipeline_versions(sort_by='created_at desc')."""
```

These are not part of the public API. If KFP upstream adds name-first
parameters to its public methods, these helpers become unnecessary and we can
remove them.

---

## Open Questions

Each question includes the current proposal and alternatives considered.

### 1. Task logs

**Gap:** `TrainerClient` exposes `get_job_logs(name, follow=False)`.
`PipelinesClient` has no logs method. KFP's Python SDK has no `get_task_logs`
either — logs live in Kubernetes pods, and the KFP UI fetches them via the K8s
API behind the scenes.

**Current proposal:** Out of scope for v1. Propose upstream to KFP as a native
`get_task_logs(run_id, task_name)` on `kfp.Client`. This is the cleanest
long-term path — KFP already knows about pods and can manage K8s auth
internally.

**Interim workaround:** Users can get the pod name from `get_run()` task
details and use `kubectl logs` or the KFP UI.

**If KFP ships it:** Our wrapper delegates to it — one-line addition:

```python
def get_task_logs(self, run_id: str, task_name: str) -> str:
    return self._client.get_task_logs(run_id=run_id, task_name=task_name)
```

**If we need it before KFP ships it:** The alternative is implementing it at
the SDK wrapper level using the Kubernetes Python client. This would change the
auth model — today `PipelinesClient` only needs KFP API credentials
(`base_url` + `user_token`). Fetching pod logs requires K8s API access
(kubeconfig or in-cluster service account), which is a different auth path.
The constructor would need additional config (e.g. `k8s_client=` parameter or
auto-detection via `load_incluster_config()`). This creates a hybrid auth model
that no other SDK client requires.

```python
# Hypothetical SDK-side implementation (if upstream doesn't ship it)
def get_task_logs(self, run_id: str, task_name: str) -> str:
    run = self.get_run(run_id)
    pod_name = run.task_details[task_name].pod_name
    # Requires K8s client — needs kubeconfig or in-cluster SA
    return self._k8s_client.read_namespaced_pod_log(pod_name, namespace=...)
```

**Decision needed:** Go with proposal, wait for upstream, or implement with hybrid auth?

### 2. Run events

**Gap:** `TrainerClient.get_job_events(name)` and
`OptimizerClient.get_job_events(name)` expose K8s events. KFP's API server
does not surface them. Same auth model concern as logs — implementing this
requires K8s API access.

**Current proposal:** Out of scope for v1. Same approach as logs — propose
upstream to KFP. If we solve the auth question for logs, events come
essentially for free.

**If KFP ships it:** Same one-line delegation as logs.

**Decision needed:** Same as logs — proposal, wait for upstream, or implement?

### 3. Runs tracked by ID after creation

**Gap:** `TrainerClient` and `ModelRegistryClient` are fully name-centric.
`PipelinesClient` tracks runs by `run_id` after creation.

**Why:** KFP run display names are not unique. Multiple runs can share the same
`job_name`. `TrainerClient` doesn't have this problem because TrainJob names
are unique K8s resource names.

**Current proposal:** Keep ID-only for v1.

**Alternatives considered:**

| Alternative | Pros | Cons |
|---|---|---|
| Name + "most recent" | Simpler API | Silently returns wrong run if duplicates exist |
| Name + experiment scoping | More precise | Still not guaranteed unique |
| Fluent run handle (`run.wait()`, `run.archive()`) | User never sees IDs | Only works for current session; can't find yesterday's run |
| Keep ID-only (current) | Explicit, unambiguous | Deviates from other clients |

**What would help upstream:** If KFP adds `display_name` filter support for
runs and recurring runs, we could offer name-based lookup as an option. We
propose this upstream.

**Decision needed:** Is ID-centric acceptable, or should we explore the fluent
handle pattern?

### 4. `wait_for_run_status` callbacks

**Gap:** `TrainerClient.wait_for_job_status` and
`OptimizerClient.wait_for_job_status` accept a `callbacks` parameter — a list
of functions invoked after each poll:

```python
# TrainerClient pattern
def log_progress(trainjob):
    print(f"Status: {trainjob.status}")

client.wait_for_job_status("my-job", callbacks=[log_progress])
```

`PipelinesClient.wait_for_run_status` does not support callbacks.

**Current proposal:** Not included in v1. The flexible `status` parameter
covers the main use case. Callbacks can be added later as a non-breaking change:

```python
# Future addition — non-breaking
def wait_for_run_status(
    self,
    run_id: str,
    ...,
    callbacks: list[Callable[[V2beta1Run], None]] | None = None,
) -> V2beta1Run:
```

**Decision needed:** Include callbacks in v1 for parity with Trainer/Optimizer,
or defer?

### 5. Return types — passthrough vs hand-crafted

**Gap:** `PipelinesClient` returns `kfp_server_api.V2beta1*` types
(auto-generated from the KFP OpenAPI spec). `TrainerClient` returns
hand-crafted Pydantic models from `kubeflow.trainer.types`.

**Current proposal:** Passthrough for v1. `ModelRegistryClient` follows the
same pattern — it returns `model_registry.types.*` from the underlying library,
not hand-crafted Kubeflow SDK types. This is zero-maintenance and familiar to
existing KFP users.

**Trade-off:** Hand-crafted wrapper types (e.g. `pipelines.types.Run`) would
give a cleaner API surface but add significant work, a mapping layer, and
potential upstream divergence.

**Decision needed:** Is passthrough acceptable, or do we want wrapper types?

### 6. Upstream KFP improvements

Several of our design decisions are bridges that become unnecessary if KFP
upstream adopts improvements. We propose the following to the KFP project:

| Improvement | What it solves for us | Status |
|---|---|---|
| `get_task_logs(run_id, task_name)` | Task logs without hybrid auth | To be proposed |
| `get_run_events(run_id)` | Run events without hybrid auth | To be proposed |
| `display_name` filter for runs/recurring runs | Name-based run lookup | To be proposed |
| Name-first public methods for pipelines/experiments | Our `_resolve_*` helpers become unnecessary | To be proposed |
| `wait_for_run_status(run_id, status={...})` | Our custom polling loop becomes unnecessary | To be proposed |

**If KFP adopts all of these**, our wrapper becomes very thin:

- Constructor alignment (`base_url` → `host` mapping)
- DSL re-exports (`kubeflow.pipelines.dsl`, etc.)
- "One SDK" packaging (`pip install 'kubeflow[pipelines]'`)

**Until then**, the wrapper implements name-first resolution, auto-version, and
flexible wait as a bridge.

### 7. Ownership and long-term maintenance

**Context:** For components that fully migrate into the Kubeflow SDK (e.g.
Trainer, Optimizer), ownership naturally sits with the component maintainers —
they own both the underlying CRD/controller and the SDK client. `PipelinesClient`
is different: it wraps an external SDK (`kfp.Client`) maintained by the KFP
project.

**What ownership means here:** The implementation can be contributed by anyone,
but ownership is primarily about **ongoing maintenance**:

- **Code maintenance** — keeping the wrapper functional as `kfp.Client` evolves
  (method signatures, deprecations, breaking changes across major versions).
- **API alignment** — ensuring the methods exposed by the wrapper stay up to
  date with what KFP offers and what other SDK clients (`TrainerClient`,
  `ModelRegistryClient`) expect in terms of naming and patterns.
- **Upstream coordination** — collaborating with KFP maintainers on the
  proposals in [Phase 4](#phase-4-upstream-kfp-proposals) (task logs, run
  events, name-first API) and adapting the wrapper as KFP ships them.
- **Testing** — maintaining unit and E2E tests, especially during KFP major
  version transitions (2.x → 3.x).

**The question:** This wrapper needs KFP maintainer buy-in to be sustainable
long-term. Bug fixes in KFP behavior need to go through the KFP project. The
upstream proposals require KFP collaboration. Without that buy-in, the wrapper
carries more bridge logic indefinitely and risks drifting from upstream.

**Decision needed:** Who owns this wrapper long-term — Kubeflow SDK maintainers,
KFP maintainers, or shared? And is the KFP team willing to collaborate on the
upstream improvements that would make the wrapper thinner over time?

**Current proposal:** Add KFP maintainers as sub-owners of the
`kubeflow/pipelines/` wrapper, with KFP maintainers as the **primary owners**
in collaboration with the Kubeflow SDK maintainers.

Rationale:

1. **Precedent in the Kubeflow SDK proposal.** The initial Kubeflow SDK design
   envisioned component teams owning their respective SDK clients. KFP
   maintainers are the component team for Pipelines — this follows the same
   model.
2. **Review and objection rights.** Making KFP maintainers primary owners gives
   them visibility into every change to the wrapper. They can review PRs, raise
   objections if the wrapper diverges from upstream conventions, and ensure the
   wrapper stays aligned with the direction of `kfp.Client`.
3. **Shared responsibility.** Kubeflow SDK maintainers contribute to the wrapper
   and keep it consistent with the broader SDK patterns (constructor alignment,
   naming conventions), while KFP maintainers ensure it stays correct and
   current with the underlying KFP SDK.

---

## Design Details

### Package Structure

```
sdk/kubeflow/
├── pipelines/
│   ├── __init__.py                    # PipelinesClient + dsl/compiler/components/kubernetes
│   └── api/
│       ├── __init__.py
│       ├── pipelines_client.py        # PipelinesClient implementation
│       └── pipelines_client_test.py   # Tests
├── trainer/    # existing
├── optimizer/  # existing
└── hub/        # existing (ModelRegistryClient)
```

### pyproject.toml

```toml
[project.optional-dependencies]
pipelines = ["kfp[kubernetes]>=2.0.0"]
```

### Deviations from Other SDK Clients

| Capability | TrainerClient | ModelRegistryClient | PipelinesClient | Gap? |
|---|---|---|---|---|
| Name-centric identity | `name` everywhere | `name` everywhere | `name` for create; `run_id` after | Yes — runs |
| Logs | `get_job_logs(name)` | N/A | Not implemented | Yes |
| Events | `get_job_events(name)` | N/A | Not implemented | Yes |
| Wait with status set | `wait_for_job_status(name, status={...})` | N/A | `wait_for_run_status(run_id, status={...})` | Parity (except name vs ID) |
| Wait with callbacks | `callbacks=[...]` | N/A | Not supported (v1) | Minor gap |
| Return types | Hand-crafted Pydantic | Passthrough (`model_registry.types`) | Passthrough (`kfp_server_api.V2beta1*`) | Same pattern as registry |
| Auth model | K8s API | REST API | REST API (K8s needed if logs/events added) | Hybrid only if logs added |
| Constructor | `backend_config=` | `base_url, port, user_token, ...` | `base_url, port, user_token, ...` + `namespace` | Mostly aligned with registry |

### Error Handling

The client raises the following exceptions:

| Exception | When |
|---|---|
| `ImportError` | `kfp` not installed. Message directs user to `pip install 'kubeflow[pipelines]'` |
| `ValueError` | Name resolution fails (pipeline/experiment not found, no versions available) |
| `TimeoutError` | `wait_for_run_status` exceeds timeout without reaching target state |

All other exceptions (network errors, auth failures, server errors) propagate
from `kfp.Client` unchanged.

### Logging and Polling

The client will log at `DEBUG` level during name resolution and status polling
(e.g. `logger.debug(f"Run {run_id}: {current_state}")`), providing visibility
without cluttering default output. This logging will be added in Phase 1.
Thread safety follows `kfp.Client`'s guarantees — consult KFP documentation
for concurrent usage patterns.

### Test Plan

- **Unit tests:** All `PipelinesClient` methods tested against a mocked
  `kfp.Client`. Tests cover name-first resolution, auto-version resolution,
  experiment resolution, `wait_for_run_status` with custom status sets, and
  error cases (pipeline not found, no versions, timeout).
- **E2E tests:** Against a live KFP server. Cover the full upload → run →
  wait → get_run flow with real pipeline compilation.

---

## Migration

### Existing `kfp.Client` users

Adoption is optional and incremental. `kfp.Client` remains fully supported.

| `kfp.Client` | `PipelinesClient` |
|---|---|
| `Client(host="...", existing_token="...")` | `PipelinesClient(base_url="...", user_token="...")` |
| `client.get_pipeline_id("name")` then `client.run_pipeline(pipeline_id=...)` | `client.run_pipeline(pipeline_name="name", ...)` |
| `client.wait_for_run_completion(run_id)` | `client.wait_for_run_status(run_id)` |
| `client.create_run_from_pipeline_func(fn)` | `client.run_pipeline_from_func(fn)` |

### When KFP SDK packaging consolidation lands (unified `kfp` 3.x)

We bump to `kfp[kubernetes]>=3.0.0`. No wrapper code changes needed — our
wrapper only calls high-level Client methods and doesn't depend on
`kfp_server_api` or `kfp_pipeline_spec` import paths.

---

## Implementation Plan

### Phase 1: Core wrapper and constructor

1. Add `pipelines = ["kfp[kubernetes]>=2.0.0"]` to `sdk/pyproject.toml`
2. Create `kubeflow/pipelines/__init__.py` with `PipelinesClient` export and `dsl`/`compiler`/`components`/`kubernetes` re-exports
3. Implement `PipelinesClient` constructor with `base_url`→`host` mapping, `user_token`→`existing_token`, `namespace` defaulting to `None`
4. Add `_resolve_pipeline_id`, `_resolve_experiment_id`, `_resolve_latest_version_id` internal helpers
5. Implement `wait_for_run_status` with flexible status set and client-side polling
6. Unit tests for constructor edge cases, name resolution, and wait logic

### Phase 2: Exposed API methods

1. Implement health/context methods (`get_health`, `set_user_namespace`, `get_user_namespace`)
2. Implement experiment methods (create, get, list, archive, unarchive, delete) with name-first resolution
3. Implement pipeline methods (get, list, delete, version CRUD) with name-first resolution
4. Implement upload methods (from file, from function, version variants)
5. Implement run methods (`run_pipeline`, `run_pipeline_from_func`, `run_pipeline_from_package`, list, get, archive, unarchive, delete, terminate) with auto-version resolution
6. Implement recurring run methods (create, get, list, delete, enable, disable) with auto-version resolution
7. Unit tests for all methods against a mocked `kfp.Client`

### Phase 3: Testing, documentation, and parity

1. E2E tests against a live KFP server covering upload → run → wait → get_run
2. Update SDK documentation on sdk.kubeflow.org with usage guide and examples
3. Improve `ImportError` messaging for DSL re-exports (suggest `pip install 'kubeflow[pipelines]'`)
4. Add migration guide examples

### Phase 4: Upstream KFP proposals

1. Propose `get_task_logs(run_id, task_name)` to KFP — native task log fetching without hybrid auth
2. Propose `get_run_events(run_id)` to KFP — native event fetching
3. Propose `display_name` filter for runs and recurring runs — enable name-based lookup
4. Propose `wait_for_run_status` with flexible status set and `callbacks` support to KFP — matching `TrainerClient`/`OptimizerClient`'s `wait_for_job_status` pattern
5. Propose namespace default of `None` instead of `"kubeflow"`
6. As KFP ships these, delegate from wrapper and remove bridge code (`_resolve_*` helpers, custom polling loop)

## Alternatives

### Alternative 1: Full re-implementation instead of wrapper

Build a `PipelinesClient` that talks to the KFP REST API directly (using
`kfp_server_api` or raw HTTP) instead of wrapping `kfp.Client`.

**Rejected:** Significant duplication of logic already in `kfp.Client`
(pipeline compilation, parameter serialization, auth handling). High
maintenance burden to keep in sync with KFP API changes.

### Alternative 2: Re-export `kfp.Client` as-is

Simply re-export `kfp.Client` under `kubeflow.pipelines.Client` without any
wrapping.

**Rejected:** Misses the core value — constructor alignment, name-first API,
and consistent wait semantics. Users would still deal with `host=`,
`existing_token=`, and ID-centric methods.

### Alternative 3: Minimal API surface

Ship only the methods needed for the simplest end-to-end flow and grow from
there. The minimal set:

```python
# The entire minimal API
client = PipelinesClient("https://ml-pipeline.example.com")

pipeline = client.upload_pipeline_from_pipeline_func(my_pipeline, pipeline_name="my-pipeline")
run = client.run_pipeline(job_name="run-1", pipeline_name="my-pipeline", experiment_name="default")
run = client.wait_for_run_status(run.run_id)
run = client.get_run(run.run_id)
client.delete_pipeline("my-pipeline")
```

Five methods + constructor. A user can upload, run, monitor, inspect, and clean
up without learning any other API surface.

**Comparison:**

| Category | Minimal (5 methods) | Full proposal (~30 methods) |
|---|---|---|
| **Upload** | `upload_pipeline_from_pipeline_func` | + `upload_pipeline`, `upload_pipeline_version`, `upload_pipeline_version_from_pipeline_func` |
| **Run** | `run_pipeline` | + `run_pipeline_from_func`, `run_pipeline_from_package` |
| **Monitor** | `wait_for_run_status`, `get_run` | + `list_runs`, `archive_run`, `unarchive_run`, `delete_run`, `terminate_run` |
| **Experiments** | *(auto-create default)* | `create_experiment`, `get_experiment`, `list_experiments`, `archive_experiment`, `unarchive_experiment`, `delete_experiment` |
| **Pipelines** | `delete_pipeline` | + `get_pipeline`, `list_pipelines`, `list_pipeline_versions`, `get_pipeline_version`, `delete_pipeline_version` |
| **Recurring runs** | *(not included)* | `create_recurring_run`, `get_recurring_run`, `list_recurring_runs`, `delete_recurring_run`, `enable_recurring_run`, `disable_recurring_run` |
| **Context** | *(not included)* | `get_health`, `set_user_namespace`, `get_user_namespace` |

**Arguments for minimal:**

- Smaller review surface, faster to ship.
- Forces the "simplest path" as the primary user experience.
- Additional methods can be added incrementally without breaking changes.

**Arguments for full (current proposal):**

- Every method beyond the minimal five is a one-line delegation to
  `kfp.Client` with no added logic — the implementation cost is trivial.
- Shipping without experiment management, pipeline listing, or recurring runs
  means users must drop down to `kfp.Client` for common operations, defeating
  the "one SDK" goal.

**Current stance:** Ship the full set. The marginal cost of each additional
method is near-zero (one-line delegations), while the marginal value is
significant — users stay within a single client for the entire pipeline
lifecycle. The minimal set is useful as a "getting started" guide rather than
an API boundary.

### Alternative 4: Migrate the KFP SDK codebase into the Kubeflow SDK

Instead of wrapping `kfp.Client`, absorb the KFP SDK code (client, DSL,
compiler, components) directly into the Kubeflow SDK repository — the same
model used by Trainer, where the training SDK lives entirely in the `kubeflow`
repo.

**How it would work:**

- The `kfp` Python package (`kfp.Client`, `kfp.dsl`, `kfp.compiler`,
  `kfp.components`, `kfp.kubernetes`) would be developed and released from the
  Kubeflow SDK repository.
- `PipelinesClient` would call into code that lives in the same repo, not an
  external dependency.
- DSL, compiler, and components would no longer need re-exports — they would
  already be under the `kubeflow` namespace.

**Pros:**

- **Single Kubeflow SDK for all components.** Pipelines would join Trainer,
  Optimizer, Spark, and Model Registry as a first-class citizen of the SDK —
  one package, one set of conventions, one place to contribute. Users
  `pip install kubeflow` and get everything.
- **Full alignment with Trainer/Optimizer model.** One repo, one release cycle,
  clearly defined maintainers of each part of the SDK.
- **No upstream coordination bottleneck.** Features like task logs, run events,
  name-first API, and flexible `wait_for_run_status` could be implemented
  directly without waiting for KFP to adopt upstream proposals.
- **Simpler dependency graph.** `pip install 'kubeflow[pipelines]'` would not
  pull in a separate `kfp` package — fewer version conflicts, no need to track
  `kfp` releases.
- **Consistent patterns.** Constructor signatures, error handling, return types,
  and `wait_for_*` semantics would be native rather than adapted from a
  different SDK's conventions.

**Cons:**

- **Requires KFP community buy-in.** The KFP SDK is actively maintained by
  the KFP team with its own release cadence, contributor base, and roadmap.
  Migrating it would need agreement from KFP maintainers to either move
  development into the Kubeflow SDK repo or maintain it as a shared effort.
  Without that agreement, this is not feasible.
- **Significant migration effort.** The KFP SDK is a large codebase — the
  client, DSL, compiler, and components are tightly coupled and have their own
  test suites, CI pipelines, and documentation. Absorbing all of that is a
  huge effort.
- **Breaking change for existing `kfp` users.** Users who `import kfp` today
  would need a migration path. Either the `kfp` package becomes a thin
  compatibility shim pointing to `kubeflow.pipelines`, or users must update
  all imports — both are disruptive.
- **Dual maintenance risk during transition.** Until the migration is complete,
  bug fixes and features would need to land in both repos, creating a painful
  transition period.
- **KFP SDK serves non-Kubeflow use cases.** Some users run KFP standalone
  without the broader Kubeflow platform. Tying the SDK to the Kubeflow repo
  could create friction for those users.
- **Decouples the KFP operator from its SDK.** Today the KFP backend
  (API server, controllers, compiler) and the KFP SDK live in the same project
  and evolve together. Migrating the SDK into the Kubeflow repo separates these
  two — the operator stays in the KFP repo while the SDK moves out. This makes
  it harder to coordinate API contract changes, test SDK against backend in CI,
  and release them in lockstep. Other Kubeflow components (Trainer, Spark)
  already have this operator/SDK split (operators live in their own repos), but
  the KFP SDK is significantly larger and more tightly coupled to its backend
  (DSL, compiler, components) than the thin client layers in Trainer or Spark.

**Current stance:** The wrapper approach is the pragmatic path forward — it
delivers the "one SDK" user experience today without requiring the migration.
If the Kubeflow and KFP communities converge on a shared SDK vision in the
future, this alternative becomes the natural next step, and the thin wrapper
makes migration easier because the public API (`PipelinesClient`) would not
change.

### Alternative 5: Git submodule — pin the KFP SDK inside the Kubeflow SDK repo

Instead of depending on `kfp` as a PyPI package or fully migrating the code,
include the KFP SDK repository as a git submodule within the Kubeflow SDK repo.
The wrapper would import from the submodule-pinned copy of `kfp` rather than
an externally installed package.

**Pros:**

- **Pinned to an exact commit.** The Kubeflow SDK controls precisely which
  version of the KFP SDK it builds and tests against — no surprises from
  upstream releases. CI runs against the pinned commit, not whatever `>=2.0.0`
  resolves to.
- **No code duplication.** The KFP source stays in its own repo with its own
  history and contributors. The submodule is a pointer, not a fork.
- **Easier to patch locally.** If the wrapper needs a temporary fix in KFP
  code (e.g. a bug blocking a release), it can pin to a fork branch or a
  specific commit without waiting for an upstream release.
- **Incremental path toward full migration.** If the community later decides
  to absorb KFP into the Kubeflow SDK (Alternative 4), the submodule makes the
  transition smoother — the code is already checked out in the repo.

**Cons:**

- **Submodule complexity.** Git submodules are notoriously painful for
  contributors — `git clone` doesn't fetch them by default (`--recurse-submodules`
  is needed), `git status` behaves differently, and updating the pin requires
  explicit `git submodule update` commands. This raises the contribution barrier.
- **Build and packaging friction.** The `kfp` package normally installs via
  pip with its own `pyproject.toml` and dependencies. Building it from a
  submodule requires custom packaging logic to include the submodule source in
  the Kubeflow SDK wheel — or users still `pip install kfp` separately,
  defeating the purpose.
- **Two sources of truth.** The submodule pin can drift from the latest KFP
  release. Someone has to actively bump the submodule — if forgotten, the
  wrapper tests against stale code while users install the latest `kfp` from
  PyPI.
- **Doesn't solve the wrapper problem.** The wrapper still delegates to
  `kfp.Client` — the submodule only controls which version is used at build
  and test time. The runtime behavior is identical to the current proposal
  unless the packaging also bundles the submodule source, which adds
  significant complexity.
- **CI complexity.** Testing requires checking out the submodule, installing
  it in the test environment, and keeping the pin in sync with the KFP version
  declared in `pyproject.toml`. This is more moving parts than a simple pip
  dependency.

**Current stance:** The added complexity is not justified. The wrapper's
dependency on `kfp>=2.0.0` via pip is simpler for contributors, simpler for
packaging, and achieves the same result. Version pinning concerns are better
addressed with standard dependency management (lock files, CI version matrices)
than with submodules.
