# Cross-Component Refactor Workflow

How to decompose a refactor that touches multiple SDK components (trainer,
spark, optimizer, hub) into parallel, conflict-free sub-tasks.

## When to use

- Renaming a shared type or constant used across components
- Changing a backend ABC method signature
- Updating a pattern enforced by import-linter across all components

## Component boundaries

Each component is isolated by import-linter contracts (`.importlinter`):

| Component | Can import from |
|-----------|----------------|
| `kubeflow.trainer` | `kubeflow.common` only |
| `kubeflow.spark` | `kubeflow.common` only |
| `kubeflow.hub` | `kubeflow.common` only |
| `kubeflow.optimizer` | `kubeflow.common` + `kubeflow.trainer.types` + `kubeflow.trainer.backends.kubernetes` + `kubeflow.trainer.constants` |

These boundaries mean **components can be edited independently** as long as
shared interfaces in `kubeflow.common` remain stable.

## Workflow

### Step 1: Identify the shared surface

Determine which files in `kubeflow/common/` are affected. Changes here
propagate to all components.

### Step 2: Change the shared layer first

Edit `kubeflow/common/` (types, utils, constants) and validate:

```bash
uv run ruff check kubeflow/common/
uv run pytest -q kubeflow/common/
```

### Step 3: Update components in parallel

Each component can be updated independently. If using multi-agent tools
(Cursor Task, Claude Code subagents), assign one sub-agent per component:

| Sub-agent | Scope | Validation |
|-----------|-------|------------|
| Agent 1 | `kubeflow/trainer/**` | `uv run pytest -q kubeflow/trainer/` |
| Agent 2 | `kubeflow/spark/**` | `uv run pytest -q kubeflow/spark/` |
| Agent 3 | `kubeflow/optimizer/**` | `uv run pytest -q kubeflow/optimizer/` |
| Agent 4 | `kubeflow/hub/**` | `uv run pytest -q kubeflow/hub/` |

Sub-agents should NOT edit files outside their assigned scope.

### Step 4: Merge validation

After all components are updated, run full validation:

```bash
make verify
make lint-imports
make test-python
```

`make lint-imports` is the critical gate — it confirms no component introduced
a forbidden cross-component import during the refactor.

### Step 5: Review

Review the combined diff. Ensure:
- Shared interface changes are backward-compatible (or all consumers updated)
- No component was missed
- Tests cover the changed paths in each component

## Pitfalls

- **Do not skip `make lint-imports`** after merging parallel work — it catches
  cross-component import violations that per-component tests miss.
- **`kubeflow.optimizer`** has a special allowance to import trainer types and
  K8s backend — check the `optimizer-trainer-limited` contract if modifying
  that boundary.
- **Update `openapi.yaml`** if the refactor changes public API signatures.
