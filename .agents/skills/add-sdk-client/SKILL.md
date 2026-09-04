---
name: add-sdk-client
description: >-
  Scaffold a new SDK client module for the Kubeflow SDK. Use when asked to create
  a new component like trainer, spark, or optimizer — including client, types,
  backend, tests, and all wiring.
---

# Scaffold a New SDK Client Module

This skill walks through creating a complete SDK client module from scratch.
Use `kubeflow/spark/` as the canonical reference — it is the most recently
added component and follows all current conventions.

## Before you start

1. Read the reference client for the full structure:

```
kubeflow/spark/api/spark_client.py
```

2. Read the reference backend:

```
kubeflow/spark/backends/kubernetes/backend.py
```

3. Read the backend abstraction design doc:

```
docs/design/backend-abstraction.md
```

4. Decide on the component name (e.g., `kubeflow/<component>/`).

## Step 1: Create the directory structure

```
kubeflow/<component>/
├── __init__.py              # Public exports
├── api/
│   ├── __init__.py
│   ├── <component>_client.py    # Main client class
│   └── <component>_client_test.py
├── backends/
│   ├── __init__.py
│   ├── base.py              # Component-specific ABC (if needed)
│   └── kubernetes/
│       ├── __init__.py
│       ├── backend.py
│       ├── backend_test.py
│       ├── constants.py
│       └── utils.py
├── types/
│   ├── __init__.py
│   ├── types.py             # Dataclasses for component schemas
│   ├── types_test.py
│   ├── options.py           # Option callables (if applicable)
│   └── options_test.py
└── test/
    ├── __init__.py
    └── common.py            # Shared test fixtures
```

Every `__init__.py` should have the Apache 2.0 license header and re-export
public symbols.

## Step 2: Implement the client

In `api/<component>_client.py`:

```python
class <Component>Client:
    def __init__(self, backend_config: KubernetesBackendConfig | None = None):
        if backend_config is None:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config: {type(backend_config)}")
```

The client is a thin facade. All execution logic belongs in backends.

## Step 3: Define types

In `types/types.py`, create `@dataclass` classes for your component's schemas.
Use type hints on all fields, Google-style docstrings, and `str | None` unions
(not `Optional`).

## Step 4: Implement the backend

Follow the `add-backend` skill for backend implementation details. At minimum:
- Subclass from your component's base or `RuntimeBackend` equivalent
- Implement CRUD operations for your component's custom resource
- Mock K8s API in tests

## Step 5: Add backend config

In `kubeflow/common/types.py`, add a backend config dataclass if the existing
`KubernetesBackendConfig` is not sufficient.

## Step 6: Wire up exports

In `kubeflow/<component>/__init__.py`, export the client and key types:

```python
from kubeflow.<component>.api.<component>_client import <Component>Client
from kubeflow.<component>.types.types import <Type1>, <Type2>

__all__ = ["<Component>Client", "<Type1>", "<Type2>"]
```

## Step 7: Add import-linter contract

Add an isolation contract in `.importlinter` to enforce component boundaries:

```ini
[importlinter:contract:<component>-isolated]
name = <Component> does not import other Kubeflow components
type = forbidden
source_modules =
    kubeflow.<component>
forbidden_modules =
    kubeflow.trainer
    kubeflow.spark
    kubeflow.optimizer
    kubeflow.hub
```

## Step 8: Add optional dependency

If the component has optional Python dependencies, add them to `pyproject.toml`
under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
<component> = ["some-api-package>=1.0.0"]
```

## Step 9: Add component context rule

Create `.agents/rules/kubeflow-<component>.mdc` (via `.agents/rules/`):

```markdown
---
description: Kubeflow <Component> — <one-line summary>
globs: kubeflow/<component>/**
alwaysApply: false
---
```

Include: Role, Package map, Invariants, Common changes, Verify commands.
Follow existing `.mdc` files as templates.

Update the **Component context** table in `AGENTS.md`.

## Step 10: Update API surfaces

- Add the component to `openapi.yaml` under new paths/schemas.
- Add CRD entries to `.agents/api/surfaces.yaml` if using custom resources.

## Step 11: Validate

```bash
uv run ruff check kubeflow/<component>/
uv run pytest -q kubeflow/<component>/
make lint-imports
make verify-openapi
make verify
```

## Checklist

- [ ] Directory structure matches the template above
- [ ] Client is a thin facade; logic lives in backends
- [ ] Types use `@dataclass` with Google-style docstrings and type hints
- [ ] Backend implements all required abstract methods
- [ ] Every `__init__.py` has license header and re-exports public symbols
- [ ] Import-linter contract enforces component isolation
- [ ] Optional dependency in `pyproject.toml` (if applicable)
- [ ] `.agents/rules/kubeflow-<component>.mdc` created
- [ ] `AGENTS.md` Component context table updated
- [ ] `openapi.yaml` and `surfaces.yaml` updated
- [ ] All validation commands pass
