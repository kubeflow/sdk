# Backend Abstraction Design

## Context

`TrainerClient` exposes one Python API for training jobs regardless of where work runs.
The concrete execution environment is selected at client construction time via
`backend_config`. See [Decision](#decision) and [Rationale](#rationale) for why three
backends exist and how they map to user journeys.

## Architecture

### Entry point

`TrainerClient` (`kubeflow/trainer/api/trainer_client.py`) accepts exactly one of:

| Config class | Backend instance | Default? |
|--------------|------------------|----------|
| `KubernetesBackendConfig` (`kubeflow/common/types.py`) | `KubernetesBackend` | Yes (`None` config) |
| `LocalProcessBackendConfig` | `LocalProcessBackend` | |
| `ContainerBackendConfig` | `ContainerBackend` | |

All public methods (`list_runtimes`, `train`, `get_job_logs`, `wait_for_job_status`, …)
delegate to `self.backend` with no backend-specific branching in the client.

### Abstract interface

`RuntimeBackend` (`kubeflow/trainer/backends/base.py`) defines the contract:

- `list_runtimes()` / `get_runtime()` / `get_runtime_packages()`
- `train(runtime, initializer, trainer, options) -> str` (job name)
- `list_jobs()` / `get_job()` / `delete_job()`
- `get_job_logs()` / `get_job_events()` / `wait_for_job_status()`

Each backend implements the same method signatures. Return types use shared SDK models
in `kubeflow.trainer.types` (e.g. `TrainJob`, `Runtime`) so callers see a uniform shape.

### Runtime discovery (differs per backend)

| Backend | Runtime source |
|---------|----------------|
| **Kubernetes** | `TrainingRuntime` / `ClusterTrainingRuntime` CRs on the cluster |
| **Container** | Built-in defaults + `ContainerBackendConfig.runtime_source` (github/https/file URLs) |
| **LocalProcess** | Hardcoded `local_runtimes` in `backends/localprocess/constants.py` only |

### Backend implementations

| Backend | Package | Execution model | `train()` creates |
|---------|---------|-----------------|-------------------|
| **Kubernetes** | `backends/kubernetes/` | K8s API (`CustomObjectsApi`) | `TrainJob` CR in target namespace |
| **Container** | `backends/container/` | Docker or Podman via adapter | One or more local containers on a per-job network |
| **LocalProcess** | `backends/localprocess/` | Host subprocess + temp venv | `LocalJob` subprocess (in-memory registry) |

#### Kubernetes

- Loads kubeconfig or in-cluster config from `KubernetesBackendConfig`
- Lists `TrainingRuntime` / `ClusterTrainingRuntime` CRs
- Creates `trainer.kubeflow.org/v1alpha1` `TrainJob` objects
- See [train-job-lifecycle.md](train-job-lifecycle.md) for the full `.train()` → TrainJob CR flow

#### Container

- Auto-detects Docker then Podman (override via `ContainerBackendConfig`)
- Serializes `CustomTrainer.func` into inline script (heredoc in container command)
- Supports multi-node via per-job network + `torchrun`
- Runs dataset/model **initializers** as containers before training when `initializer=` is set
- Module docstring: v1 supports **CustomTrainer only**

#### LocalProcess

- Uses built-in `local_runtimes` definitions (not cluster CRs)
- Requires explicit `runtime` argument
- Creates temp venv dir, builds shell command, starts `LocalJob` subprocess
- Supports **CustomTrainer only**
- Accepts `initializer` in the `train()` signature but **does not run it** (ignored today)
- `get_job_events()` raises `NotImplementedError`

### Initializer support

| Backend | `initializer` in `train()` |
|---------|---------------------------|
| **Kubernetes** | Embedded in TrainJob spec; controller runs init pods |
| **Container** | Runs dataset/model init containers locally before training |
| **LocalProcess** | Accepted in signature but ignored |

### Trainer type support matrix (from validation in backends)

| Trainer type | Kubernetes | Container | LocalProcess |
|--------------|------------|-----------|--------------|
| `CustomTrainer` | Yes (runtime must be `CUSTOM_TRAINER`) | Yes | Yes |
| `CustomTrainerContainer` | Yes | No | No |
| `BuiltinTrainer` | Yes (runtime must be `BUILTIN_TRAINER`) | No | No |

`BuiltinTrainer` and `CustomTrainerContainer` are Kubernetes-only; local backends accept
`CustomTrainer` only.

### Options pattern

`options` is a list of **callable** objects passed to `train()`. Each option implements
`__call__(job_spec: dict, trainer, backend: RuntimeBackend)`.

- **Common options** (`kubeflow/trainer/options/common.py`): e.g. `Name` — work on all backends
- **Kubernetes options** (`kubeflow/trainer/options/kubernetes.py`): e.g. pod overrides, labels —
  each checks `isinstance(backend, KubernetesBackend)` and raises `ValueError` otherwise
- **LocalProcess options** (`kubeflow/trainer/options/localprocess.py`): placeholder only — use
  `Name` + `LocalProcessBackendConfig`; planned options (`ProcessTimeout`, `WorkingDirectory`)
  not implemented yet

**Options scope by backend:**

| Backend | Options consumed from `job_spec` |
|---------|--------------------------------|
| **Kubernetes** | Full merge: metadata, spec labels/annotations, trainer overrides, pod templates |
| **Container** | `metadata.name` only |
| **LocalProcess** | `metadata.name` only |

Backends iterate `options` and merge results into a mutable `job_spec` dict before building the job.

## Decision

When you call `TrainerClient.train()`, the SDK needs to decide where to run your training.
This is controlled by the backend. Most of the time you'll use the Kubernetes backend as the
default; however, there are two other options available with local development in mind.

- **Kubernetes Backend** creates a TrainJob in your cluster. This is the production path.
  Once the TrainJob is submitted, your Python script can exit — the controller manages
  execution. Use this when you have a cluster with GPU nodes.
- **Container Backend** runs training in local Docker or Podman containers. No cluster
  needed. The containers get the same environment variables they would in the cluster, so
  you can test dataset loading, training logic, and checkpoint saving locally before
  submitting to a cluster.
- **LocalProcess Backend** runs your training function as a Python subprocess via
  `subprocess.Popen`. No containers, no cluster — just direct execution. The fastest way to
  check if your function runs at all, but it doesn't support distributed training.

| Backend | Where it runs | Distributed? | Best for |
|---------|---------------|--------------|----------|
| Kubernetes | K8s cluster | Yes | Production training |
| Container | Local Docker/Podman | Yes (multi-container via torchrun) | Testing distributed training locally without a cluster |
| LocalProcess | Python subprocess | No | Quick function validation |

## Rationale

The backend layer bridges the gap between data science workflows and Kubernetes
infrastructure. Most practitioners want to write Python training code — not TrainJob YAML,
pod templates, or distributed coordination env vars. The SDK hides that complexity behind
`TrainerClient` while still producing correct cluster resources when the Kubernetes backend is selected.

### Single client, pluggable backends

The design uses one `TrainerClient` with config-driven backend selection
(`KubernetesBackendConfig`, `ContainerBackendConfig`, `LocalProcessBackendConfig`) rather
than separate entry points (e.g. a hypothetical `DockerTrainerClient`). At construction
time the client instantiates the matching `RuntimeBackend` subclass; all public methods
delegate to `self.backend` with no backend-specific branching in the facade.

Benefits:

- Notebooks, examples, and CI scripts swap `backend_config` without rewriting training calls
- New execution targets add a backend + config pair; the client API stays stable
- Shared types (`TrainJob`, `Runtime`) keep job listing, logs, and status vocabulary
  consistent where implemented

Rejected alternatives:

- **Separate clients per environment** — duplicates API surface and drifts examples
- **Kubernetes-only SDK** — forces Kind/minikube or cloud spend for every syntax check
- **Direct Pod/Job creation** — bypasses TrainJob CR and TrainingRuntime templates
- **Single local backend only** — cannot test container env parity or multi-node `torchrun`
  before cluster submit

### Why local backends exist

| Pain point | How local backends help |
|------------|-------------------------|
| Long wait for pod schedule / image pull | LocalProcess runs in seconds on the host |
| Cloud cost for throwaway experiments | No cluster quota or GPU node time for smoke tests |
| Kubernetes complexity for many data scientists | Container and LocalProcess hide CRDs during inner-loop dev |

**Container backend** sits between production and quick checks: same container image and
env-var semantics as cluster pods, supports multi-node via `torchrun`, and can run
dataset/model initializers locally — without kubeconfig or TrainJob reconciliation.

**LocalProcess backend** is the fastest path: host `subprocess.Popen`, temp venv,
`CustomTrainer` only, no distributed training.

## Invariants

- `TrainerClient` must not contain backend-specific job-creation logic; new execution
  targets add a `RuntimeBackend` subclass + config type.
- Options that require cluster APIs must validate backend type in `__call__` (see
  `kubeflow/trainer/options/kubernetes.py`).
- Backends return the same `types.TrainJob` status vocabulary (`Created`, `Running`,
  `Complete`, `Failed`) where implemented.
- Kubernetes `train()` builds Trainer CR fragments only through `_get_trainjob_spec()` and
  `kubeflow/trainer/backends/kubernetes/utils.py`.

## Preconditions

- **Kubernetes:** Valid kubeconfig or in-cluster credentials; Trainer control plane installed
  (SDK warns if version ConfigMap missing; does not block).
- **Container:** Docker or Podman socket available (`ContainerBackendConfig.container_host`
  optional).
- **LocalProcess:** `runtime` must be provided; trainer must be `CustomTrainer`.

## Trade-offs

### Per-backend comparison

| Dimension | Kubernetes | Container | LocalProcess |
|-----------|------------|-----------|--------------|
| **Setup cost** | High — cluster, Trainer CRDs, kubeconfig | Medium — Docker or Podman socket | Low — host Python + temp venv |
| **Iteration speed** | Slow — schedule, image pull, controller reconcile | Medium — local pull/start | Fast — subprocess start |
| **Cluster fidelity** | Full — TrainJob CR, controller lifecycle, TrainingRuntime CRs | Partial — same images/env vars; no CR or controller | Low — host subprocess |
| **Trainer surface** | Full — Custom, Builtin, container trainers | `CustomTrainer` only (v1) | `CustomTrainer` only |
| **Distributed training** | Yes | Yes — multi-container via `torchrun` | No |
| **Options support** | Full `job_spec` merge | `Name` only | `Name` only |
| **Initializers** | Controller runs init pods | Local init containers before training | Ignored in `train()` today |

**Kubernetes** is the source of truth for production: only path that creates TrainJob CRs,
supports the full trainer matrix, and Kubernetes-specific options.

**Container** is the best pre-submit check when you need image parity or multi-node behavior
without a cluster. It does not exercise the Trainer controller or TrainJob status
reconciliation.

**LocalProcess** optimizes for the innermost loop: syntax, imports, and basic training logic
on `CustomTrainer`.

### Architectural patterns (chosen vs alternatives)

| Approach | Chosen? | Pros | Cons |
|----------|---------|------|------|
| Shared `RuntimeBackend` ABC | Yes | Uniform `TrainerClient` surface | Each backend must emulate shared types where features differ |
| Config-driven backend selection | Yes | Swap `backend_config` without API changes | No runtime backend switching on an existing client |
| Backend-specific trainer restrictions | Yes | Local backends stay simple and testable | Users must switch backend for Builtin trainers |
| Options as callables mutating `job_spec` | Yes | Extensible without changing `train()` signature | Kubernetes options fail on local backends unless guarded |
| Adapter pattern (Container) | Yes | Docker vs Podman differences isolated | Extra indirection in container lifecycle code |
| Separate clients per environment | No | Duplicated API, divergent examples | |
| Kubernetes-only SDK | No | Blocks fast iteration | |

## Related code

- `kubeflow/trainer/api/trainer_client.py` — facade
- `kubeflow/common/types.py` — backend config types
- `kubeflow/trainer/backends/base.py` — ABC
- `kubeflow/trainer/backends/kubernetes/backend.py`
- `kubeflow/trainer/backends/container/backend.py`
- `kubeflow/trainer/backends/localprocess/backend.py`
- `kubeflow/trainer/options/` — job customization callables
- [train-job-lifecycle.md](train-job-lifecycle.md) — Kubernetes `train()` detail
