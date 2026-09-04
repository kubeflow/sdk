# TrainJob Lifecycle: SDK Call to Cluster CR

## Context

Users call `TrainerClient(backend_config=KubernetesBackendConfig(...)).train(...)`.
On the Kubernetes backend, that call ends with a `TrainJob` custom resource in the API
server. Training pods are **not** started by the SDK itself.

## End-to-end flow (Kubernetes backend)

```mermaid
sequenceDiagram
    participant User
    participant TC as TrainerClient
    participant KB as KubernetesBackend
    participant Opt as options[]
    participant Spec as _get_trainjob_spec
    participant API as K8s API server
    participant Ctrl as Trainer controller

    User->>TC: train(runtime, initializer, trainer, options)
    TC->>KB: backend.train(...)
    loop each option
        KB->>Opt: option(job_spec, trainer, backend)
        Opt-->>KB: mutates job_spec dict
    end
    KB->>KB: generate train_job_name
    KB->>Spec: _get_trainjob_spec(...)
    Spec-->>KB: TrainerV1alpha1TrainJobSpec
    KB->>API: create_namespaced_custom_object(TrainJob)
    API-->>KB: created
    KB-->>TC: train_job_name
    TC-->>User: train_job_name
    Note over Ctrl: Out of SDK scope
    Ctrl->>API: reconcile TrainJob → Pods
```

## Step-by-step (from code)

### 1. Client delegation

`TrainerClient.train()` (`kubeflow/trainer/api/trainer_client.py`) forwards all arguments
to `self.backend.train()` unchanged.

### 2. Apply options

`KubernetesBackend.train()` (`kubeflow/trainer/backends/kubernetes/backend.py`):

1. Initializes empty `job_spec = {}`
2. For each `option` in `options`: `option(job_spec, trainer, self)`
3. Extracts from `job_spec`:
   - `metadata`: `labels`, `annotations`, `name`
   - `spec`: `labels`, `annotations`, `trainer` overrides, `podTemplateOverrides`

Common keys are consumed by backend-specific option classes in `kubeflow/trainer/options/`.

### 3. Job name

```text
train_job_name = metadata.name OR (random lowercase letter + uuid hex[:JOB_NAME_UUID_LENGTH])
```

Constant `JOB_NAME_UUID_LENGTH` is in `kubeflow/trainer/constants/constants.py`.

### 4. Build TrainJob spec — `_get_trainjob_spec()`

| Input | Behavior |
|-------|----------|
| `runtime is None` | `get_runtime(DEFAULT_TRAINING_RUNTIME)` — env override, default `torch-distributed` |
| `runtime` is `str` | Resolved via `get_runtime(name)` |
| `trainer` type | Builds `TrainerV1alpha1Trainer` CR fragment (see below) |
| `trainer_overrides` | Patches `command` / `args` on trainer CR when set via options |
| `initializer` | Adds `TrainerV1alpha1Initializer` for dataset/model when present. The SDK passes `storage_uri` (e.g., `hf://`, `s3://`, `cache://`) to the TrainJob spec; the controller mounts storage and runs initializer pods before training starts. |

#### Trainer → Trainer CR mapping

| Trainer type | Builder | Runtime constraint |
|--------------|---------|-------------------|
| `CustomTrainer` / `CustomTrainerContainer` | `utils.get_trainer_cr_from_custom_trainer` | `runtime.trainer.trainer_type == CUSTOM_TRAINER` |
| `BuiltinTrainer` | `utils.get_trainer_cr_from_builtin_trainer` | `runtime.trainer.trainer_type == BUILTIN_TRAINER` |

#### Assembled spec fields

`TrainerV1alpha1TrainJobSpec`:

- `runtimeRef.name` — from resolved `Runtime`
- `trainer` — CR fragment (or `None` if empty default)
- `labels` / `annotations` — spec-level from options
- `pod_template_overrides` — from options
- `initializer` — optional dataset/model init

### 5. Create TrainJob CR

Builds `TrainerV1alpha1TrainJob`:

- `apiVersion`: `trainer.kubeflow.org/v1alpha1`
- `kind`: `TrainJob`
- `metadata.name`: `train_job_name`
- `spec`: from step 4

Creates via:

```python
custom_api.create_namespaced_custom_object(
    GROUP, VERSION, namespace, TRAINJOB_PLURAL, train_job.to_dict()
)
```

Constants (`GROUP`, `VERSION`, `TRAINJOB_PLURAL`) live in `kubeflow/trainer/constants/constants.py`.

On success, returns `train_job_name`. SDK does **not** wait for pods unless the caller
invokes `wait_for_job_status()`.

### 6. Post-create operations (SDK)

| Method | Mechanism |
|--------|-----------|
| `get_job()` | `get_namespaced_custom_object` → map CR to `types.TrainJob` |
| `get_job_logs()` | Resolve pod for step (e.g. `node-0`) → stream container logs |
| `wait_for_job_status()` | Poll `get_job()` until status in expected set |
| `delete_job()` | Delete TrainJob CR |

Status mapping (`__get_trainjob_from_cr`) reads CR `status.conditions` for
`Complete` / `Failed`, else infers `Running` when all node steps are running/succeeded.

## Out of SDK scope (controller reconciliation)

Once the SDK creates the TrainJob CR (steps 1–5 above), the **Kubeflow Trainer controller**
owns reconciliation. The SDK does not create pods, JobSets, or PodGroups — it only submits
the CR and optionally polls status via `get_job()` / `wait_for_job_status()`.

The controller watches TrainJob resources, loads the referenced **TrainingRuntime** (or
**ClusterTrainingRuntime**) template, runs it through a **plugin pipeline**, and materializes
a **JobSet** that manages initializer and trainer pods on GPU nodes.

### Controller lifecycle

1. **Kubernetes validates and stores the CR** — The API server validates the TrainJob against
   the CRD schema and persists it in etcd.
2. **Controller detects the TrainJob** — The Trainer controller reconciles: compares desired
   state (TrainJob spec) with observed state (JobSet, pods, conditions).
3. **Controller loads the runtime template** — Reads `spec.runtimeRef.name` and fetches the
   matching runtime. If not found, the TrainJob fails with `TrainingRuntimeNotSupported`.
4. **Plugins process the template** — Each plugin injects framework- or scheduling-specific
   config into the runtime template (see [Plugin pipeline](#plugin-pipeline) below).
5. **Controller creates a JobSet** — The JobSet groups multiple Jobs (initializer + trainer
   nodes) as a single unit.
6. **Initializer runs first** (when configured) — Dataset/model init pods download artifacts
   to a shared PVC once; trainer jobs start after initializers succeed.
7. **Training pods run** — Each trainer pod runs the user's code with distributed env vars
   (e.g. `PET_MASTER_ADDR`, `PET_NNODES`) pre-configured by the Torch plugin. The controller
   monitors JobSet/pod status and updates TrainJob conditions.

### Plugin pipeline

Plugins run in order during reconciliation. Which plugins activate depends on the runtime's
`mlPolicy` and `podGroupPolicy`:

| Plugin | Type | Activated when | What it does |
|--------|------|----------------|--------------|
| **Torch** | ML policy | `mlPolicy.torch` is set | Injects env vars (`PET_NNODES`, `PET_MASTER_ADDR`, …) for `torchrun` coordination |
| **MPI** | ML policy | `mlPolicy.mpi` is set | Generates SSH keys and hostfile for MPI-based training |
| **PlainML** | ML policy | No framework specified | Pass-through — no framework-specific config |
| **Coscheduling** | Pod group | `podGroupPolicy.coscheduling` is set | Creates a PodGroup for gang-scheduling (all pods start together) |
| **JobSet** | Infra (always) | Always | Builds the final JobSet from the runtime template with all plugin config applied |
| **TrainJobStatus** | Infra (always) | Always | Aggregates JobSet status back onto the TrainJob (`Suspended`, `Complete`, `Failed`) |

Reference: [`kubeflow/trainer` — `pkg/runtime/framework/plugins/`](https://github.com/kubeflow/trainer/tree/master/pkg/runtime/framework/plugins)

### TrainJob status

The controller surfaces three primary conditions on a TrainJob:

| Condition | Meaning |
|-----------|---------|
| **Suspended** | Training is paused (`spec.suspend: true`) |
| **Complete** | All jobs in the JobSet finished successfully |
| **Failed** | A job failed or a deadline was exceeded |

The SDK maps these when polling: `get_job()` and `wait_for_job_status()` read
`status.conditions` on the TrainJob CR (see step 6 above).

### SDK vs controller boundary

| Responsibility | Owner |
|----------------|-------|
| Build TrainJob spec from trainer + runtime + options | SDK (`KubernetesBackend`) |
| Create TrainJob CR in API server | SDK |
| Load TrainingRuntime, run plugins, create JobSet | Trainer controller |
| Schedule pods, run initializers, distributed env setup | Trainer controller + JobSet |
| Update TrainJob conditions | Trainer controller |
| Poll status, stream logs, delete CR | SDK |

- **TrainingRuntime** / **ClusterTrainingRuntime** CRs — platform-managed templates (images,
  resources, framework policy)
- **TrainJob** CR — the user's submission artifact; what `TrainerClient.train()` creates
- Pod scheduling, failure recovery, and condition updates — controller responsibilities

## Local backends (contrast)

For comparison, non-Kubernetes backends do **not** create TrainJob CRs:

| Backend | `train()` effect |
|---------|------------------|
| **Container** | Creates Docker/Podman containers with inline training script |
| **LocalProcess** | Starts host subprocess with generated script in temp venv |

Both still return a job name string and emulate `TrainJob`/`Step` models in memory for
`list_jobs()` / `get_job()`.

## Decision

The Kubernetes backend submits a **`TrainJob` custom resource** as the unit of work — not a bare
`batch/v1` `Job`, and not a v1 framework job such as `PyTorchJob`.

A TrainJob references a platform-managed **`TrainingRuntime`** (or **`ClusterTrainingRuntime`**
) blueprint by name. The SDK fills in the trainer fragment (code, image, or algorithm config); the
Trainer controller resolves the runtime, runs the plugin pipeline, and materializes a **JobSet**
that owns initializer and training pods.

| Submission unit | Used by SDK? | Role |
|-----------------|--------------|------|
| **`TrainJob` CR** | Yes (Kubernetes backend) | User-facing submission — *what* to train + runtime reference |
| **`TrainingRuntime` CR** | Referenced, not created by SDK | Platform blueprint — *how* to run (image, GPUs, plugins, topology) |
| **`JobSet` / `Job` / Pods** | No (controller creates) | Infrastructure — created and reconciled by Trainer controller |
| **`PyTorchJob` (v1)** | No | Legacy Training Operator CR; superseded by Trainer v2 |

This split keeps `TrainerClient.train()` stable while platform teams evolve runtimes
(CUDA/ROCm images, gang-scheduling, initializer templates) independently of user code.

## Rationale

### Why TrainJob + TrainingRuntime (not bare Job)

Bare Kubernetes `Job` objects expose pod templates, resource requests, env vars, and distributed
coordination directly in every submission. That forces data scientists to understand infrastructure
details on each run and duplicates platform configuration across notebooks and CI jobs.

TrainJob inverts that:

- **Platform admins** publish tested `TrainingRuntime` / `ClusterTrainingRuntime` templates.
- **Data scientists** call `TrainerClient.train(runtime="torch-distributed", trainer=…)` and
  only specify training intent.
- **Controller** owns JobSet creation, initializer ordering, framework plugins (Torch, MPI,
  Coscheduling), and status conditions.

The SDK therefore stops at CR submission; it does not schedule pods or manage distributed env
vars — see [Out of SDK scope](#out-of-sdk-scope-controller-reconciliation).

### Why TrainJob (not PyTorchJob / v1 Training Operator)

Kubeflow Training v1 used framework-specific CRDs (`PyTorchJob`, `MPIJob`, `TFJob`, …). Each
carried both training intent and infrastructure wiring. The v1 operator also created low-level
Pods and Services per rank and re-implemented batch Job behaviors (suspend, failure recovery)
that Kubernetes now provides through **JobSet** and enhanced `batch/v1` `Job` APIs.

[KEP-2170 (Kubeflow Training V2)](https://github.com/kubeflow/training-operator/blob/bd82422a15742f73d7567b3be665f7b54be11d81/docs/proposals/2170-kubeflow-training-v2/README.md)
introduced a single **`TrainJob`** API plus reusable runtime blueprints to:

- Reduce operator maintenance by delegating infra to JobSet instead of custom Pod/Service blocks
- Hide Kubernetes complexity for data scientists who know ML APIs but not pod topology
- Support batch workload features (PodFailurePolicy, disruption handling) via the JobSet layer
- Enable Kueue / MultiKueue integration for queue-managed training

Trainer v2 (`TrainJob`) is the supported production path going forward.

### Why one CR across the product stack

Using TrainJob as the submission unit keeps downstream components aligned:

- **Katib / Optimizer** — each trial creates a TrainJob (same controller, same status model)
- **Dashboard / monitoring** — reads TrainJob conditions
- **SDK** — `get_job()`, `wait_for_job_status()`, and `list_jobs()` map one CR shape

A bare Job or legacy PyTorchJob would not participate in this stack without adapter layers.

### Rejected alternatives

- **Direct `Job` / Pod creation from SDK** — bypasses TrainingRuntime templates, plugin pipeline,
  and controller lifecycle; breaks platform-managed runtimes and Katib trial integration
- **Continue v1 `PyTorchJob` as primary API** — framework-specific CRs, duplicated Job
  semantics in operator, deprecated product trajectory
- **SDK creates JobSet directly** — leaks infra concerns into client; bypasses runtime
  validation, webhooks, and controller reconciliation

## Invariants

- Kubernetes `train()` must build the full spec through `_get_trainjob_spec()`; do not
  construct `TrainerV1alpha1TrainJob` fragments ad hoc in other modules.
- Trainer type must match `Runtime.trainer.trainer_type` or `ValueError` is raised before
  API call.
- `runtimeRef.name` always references an existing TrainingRuntime the backend can list.

## Preconditions

- Target namespace exists and caller has `create` on `trainjobs.trainer.kubeflow.org`
- Referenced `TrainingRuntime` (or cluster-scoped equivalent) is installed

## Related code

- `kubeflow/trainer/api/trainer_client.py` — `train()`
- `kubeflow/trainer/backends/kubernetes/backend.py` — `train()`, `_get_trainjob_spec()`
- `kubeflow/trainer/backends/kubernetes/utils.py` — custom/builtin trainer CR builders
- `kubeflow/trainer/constants/constants.py` — API group, kinds, status strings
- [backend-abstraction.md](backend-abstraction.md) — why Kubernetes backend creates TrainJob CRs
