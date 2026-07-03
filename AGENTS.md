# AI Agents Guide for Kubeflow SDK

The Kubeflow SDK provides unified Pythonic APIs for Kubeflow workloads at any scale. This file is the entry point for AI agents — keep it lean and follow linked documents for detail.

## Start here

| Read this | When |
|-----------|------|
| [references/agent-behaviors.md](references/agent-behaviors.md) | Before any code change — agent constraints, scope limits, and context-awareness rules |
| [references/core-principles.md](references/core-principles.md) | Writing or reviewing Python — public APIs, style, tests, security, or docstrings |
| [references/development-workflow.md](references/development-workflow.md) | Full change checklist — pre-edit steps, validation sequence, and commit/PR detail beyond AGENTS.md |
| [references/common-changes.md](references/common-changes.md) | Implementing a common change — backend options, backends, types, tests, or SDK clients |
| [docs/design/backend-abstraction.md](docs/design/backend-abstraction.md) | Understanding execution backends — Kubernetes, Container, LocalProcess |
| [docs/design/train-job-lifecycle.md](docs/design/train-job-lifecycle.md) | How `TrainerClient.train()` becomes a TrainJob CR on Kubernetes |
| [.agents/api/surfaces.yaml](.agents/api/surfaces.yaml) | Machine-readable surface index — SDK OpenAPI, CRD schemas, external APIs |

For code changes, read `agent-behaviors.md` first, then `core-principles.md` when editing Python,
`common-changes.md` when following an existing pattern, and `development-workflow.md` before
proposing a commit or PR. Read design docs when changing backend architecture or TrainJob creation.

## Repository Map

| Path | Use for |
|------|---------|
| `.github/` | CI/CD workflows |
| `CHANGELOG/` | Release changelogs |
| `docs/` | Project documentation |
| `docs/design/` | Architecture intent — backends, TrainJob lifecycle |
| `examples/` | Usage examples |
| `hack/` | CI/CD and installation scripts |
| `proposals/` | Kubeflow Enhancement Proposals (KEPs) |
| `test/` | Top-level end-to-end tests |
| `kubeflow/common/` | Shared utilities and types |
| `kubeflow/trainer/api/` | `TrainerClient` — main trainer entry point |
| `kubeflow/trainer/backends/` | Kubernetes, Container, and LocalProcess execution backends |
| `kubeflow/trainer/options/` | Training job options (e.g. labels, pod overrides) |
| `kubeflow/trainer/types/` | Trainer schemas (`TrainJob`, `CustomTrainer`, …) |
| `kubeflow/trainer/test/` | Shared test fixtures (`common.py`) |
| `kubeflow/spark/api/` | `SparkClient` |
| `kubeflow/optimizer/api/` | `OptimizerClient` |
| `kubeflow/hub/api/` | `ModelRegistryClient` |

## Component context

Path-scoped rules in `.cursor/rules/` provide module detail when editing a component.

| Component | Rule | Scope |
|-----------|------|-------|
| Trainer | `.cursor/rules/kubeflow-trainer.mdc` | `kubeflow/trainer/**` |
| Trainer K8s backend | `.cursor/rules/kubeflow-trainer-kubernetes.mdc` | `kubeflow/trainer/backends/kubernetes/**` |
| Spark | `.cursor/rules/kubeflow-spark.mdc` | `kubeflow/spark/**` |
| Optimizer | `.cursor/rules/kubeflow-optimizer.mdc` | `kubeflow/optimizer/**` |

## Environment & Tooling

- **Package manager**: `uv` (creates `.venv` automatically via targets)
- **Lint/format**: `ruff` (isort integrated)
- **Tests**: `pytest` with coverage
- **Build**: Hatchling (optional `uv build`)
- **Pre-commit**: Config provided and enforced in CI

## Commands

<!-- BEGIN: AGENT_COMMANDS -->

**Setup**:

```bash
make install-dev              # Install uv, create .venv, sync deps
```

**Verify (CI parity)**:

```bash
make verify                   # Runs ruff check --show-fixes and ruff format --check
make verify-openapi           # Validate openapi.yaml (OpenAPI 3.x schema)
```

**Single-file verification** (fast feedback during edits; no full build; target under 5 seconds per file):

```bash
uv run ruff check path/to/file.py
ruff check path/to/file.py
uv run ty check path/to/file.py
mypy path/to/file.py
uv run pytest -q path/to/file_test.py
```

Prefix with `uv run` when using the project virtualenv. Primary type checker in this repo is `ty`; `mypy` is supported for single-file type-check workflows.

Prefer these over full-repo commands while iterating on a single module.

**Testing**:

```bash
make test-python              # All unit tests + coverage (HTML by default)
make test-python report=xml   # XML coverage report
uv run pytest -q kubeflow/trainer/backends/kubernetes/backend_test.py
uv run pytest -q kubeflow/trainer/backends/kubernetes/backend_test.py::test_name -k "pattern"
uv run coverage run -m pytest <path> && uv run coverage report          # Ad-hoc coverage
```

**Local lint/format**:

```bash
uv run ruff check --fix .                    # Fix lint issues (all files)
uv run ruff format kubeflow                  # Format kubeflow package
```

**Type checking**:

```bash
uv run ty check kubeflow                     # Run type checker (project default)
```

See **Single-file verification** above for per-file `ruff check` and `mypy` commands.

**Pre-commit**:

```bash
uv run pre-commit install                    # Install pre-commit and commit-msg hooks
uv run pre-commit run --all-files           # Run all hooks
```

<!-- END: AGENT_COMMANDS -->

## Key Conventions

- Preserve public API signatures; use keyword-only args for new params
- Type hints on all functions; line length 100; first-party import is `kubeflow`
- Unit tests in `*_test.py`; no network calls in unit tests

## Pattern References

Copy-modify from these examples; see [references/common-changes.md](references/common-changes.md) for file lists.

- New Kubernetes training option: follow the pattern in `kubeflow/trainer/options/kubernetes.py`
- Extend an execution backend: follow the pattern in `kubeflow/trainer/backends/kubernetes/backend.py`
- Add trainer types: see `kubeflow/trainer/types/types.py` for the schema pattern
- Add backend tests: follow the pattern in `kubeflow/trainer/backends/kubernetes/backend_test.py`
- New SDK client: see `kubeflow/spark/api/spark_client.py` for reference

## Pull Requests

- Use Conventional Commits for PR titles; see [CONTRIBUTING.md](CONTRIBUTING.md#pull-request-title-conventions)
- Before proposing: run `make verify` and targeted tests for changed code
- Keep diffs minimal and scoped to the task; include rationale ("why") in the PR description
- Do not commit secrets or modify git config

Details: [references/development-workflow.md](references/development-workflow.md)
