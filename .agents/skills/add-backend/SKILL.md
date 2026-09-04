---
name: add-backend
description: >-
  Add or extend an execution backend for the Kubeflow SDK. Use when asked to
  create a new backend, extend an existing backend, or implement a RuntimeBackend
  method.
---

# Add or Extend an Execution Backend

Execution backends implement the `RuntimeBackend` ABC and are responsible for
job lifecycle (create, list, get, logs, events, wait, delete). The SDK currently
has three backends: Kubernetes, Container (Docker/Podman), and LocalProcess.

## Before you start

1. Read the ABC contract — every backend must implement these methods:

```
kubeflow/trainer/backends/base.py
```

2. Read the canonical Kubernetes backend for the full implementation pattern:

```
kubeflow/trainer/backends/kubernetes/backend.py
```

3. Read the backend abstraction design doc for architectural context:

```
docs/design/backend-abstraction.md
```

4. Decide whether you are:
   - **Extending an existing backend** (adding a method or modifying behavior)
   - **Creating a new backend** (new directory under `backends/`)

## Extending an existing backend

### Step 1: Modify the backend

Edit the backend file (e.g., `kubeflow/trainer/backends/kubernetes/backend.py`).

If adding a new public method, also add the abstract signature to
`kubeflow/trainer/backends/base.py` so all backends must implement it.

### Step 2: Add tests

Add test cases in the co-located `*_test.py` file. Follow the parametrized
`TestCase` pattern from `kubeflow/trainer/backends/kubernetes/backend_test.py`:

```python
@dataclasses.dataclass
class TestCase:
    name: str
    # ... test-specific fields
    expected_status: str
    expected_error: str | None = None
```

Key rules:
- Mock K8s API only — no real cluster or network calls.
- Use fixtures from `kubeflow/trainer/test/common.py`.
- Test both success and error paths.

### Step 3: Validate

```bash
uv run ruff check kubeflow/trainer/backends/<name>/backend.py
uv run pytest -q kubeflow/trainer/backends/<name>/backend_test.py
make lint-imports
```

## Creating a new backend

### Step 1: Create the directory structure

```
kubeflow/<component>/backends/<name>/
├── __init__.py
├── backend.py        # RuntimeBackend subclass
├── backend_test.py   # Parametrized tests
├── constants.py      # CRD group/version, defaults (if K8s-based)
└── utils.py          # Spec building helpers (if needed)
```

### Step 2: Implement `RuntimeBackend`

In `backend.py`, subclass `RuntimeBackend` and implement all abstract methods:

- `list_runtimes()` -> `list[types.Runtime]`
- `get_runtime(name)` -> `types.Runtime`
- `get_runtime_packages(runtime)`
- `train(runtime, initializer, trainer, options)` -> `str`
- `list_jobs(runtime)` -> `list[types.TrainJob]`
- `get_job(name)` -> `types.TrainJob`
- `get_job_logs(name, follow, step)` -> `Iterator[str]`
- `get_job_events(name)` -> `list[types.Event]`
- `wait_for_job_status(name, status, timeout, polling_interval, callbacks)` -> `types.TrainJob`
- `delete_job(name)`

Use shared status constants from `kubeflow/trainer/constants/constants.py`.

### Step 3: Create a backend config

Add a config dataclass in `kubeflow/common/types.py` following the existing
`KubernetesBackendConfig`, `ContainerBackendConfig`, or
`LocalProcessBackendConfig` patterns.

### Step 4: Register in the client

Update `kubeflow/trainer/api/trainer_client.py` to accept the new config in
`__init__` and instantiate the backend.

### Step 5: Add import-linter contract (if new component)

If this is a new top-level component (not under `kubeflow/trainer/`), add
an isolation contract in `.importlinter`:

```ini
[importlinter:contract:<name>-isolated]
name = <Name> does not import other Kubeflow components
type = forbidden
source_modules =
    kubeflow.<name>
forbidden_modules =
    kubeflow.trainer
    kubeflow.spark
    kubeflow.optimizer
    kubeflow.hub
```

### Step 6: Write tests

Follow the parametrized `TestCase` pattern. Mock external APIs.

### Step 7: Validate

```bash
uv run ruff check kubeflow/<component>/backends/<name>/backend.py
uv run pytest -q kubeflow/<component>/backends/<name>/backend_test.py
make lint-imports
```

## Checklist

- [ ] Subclasses `RuntimeBackend` and implements all abstract methods
- [ ] Returns shared status vocabulary: `Created`, `Running`, `Complete`, `Failed`
- [ ] Unit tests mock external APIs — no network calls
- [ ] Backend config dataclass in `kubeflow/common/types.py` (if new backend)
- [ ] Registered in the client's `__init__` (if new backend)
- [ ] Import-linter contract added (if new component)
- [ ] All validation commands pass
