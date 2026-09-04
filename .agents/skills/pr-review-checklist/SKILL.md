---
name: pr-review-checklist
description: >-
  Pre-submission quality checklist for Kubeflow SDK pull requests. Use when
  preparing a commit, PR, or reviewing changes before push.
---

# PR Review Checklist

Run through this checklist before committing or proposing a pull request.
Each section has a validation command — run them all and fix failures.

## 1. Lint and format

```bash
make verify
```

This runs `uv lock --check`, `ruff check`, `ruff format --check`, and
`ty check kubeflow/hub kubeflow/common` in one step.

To auto-fix lint issues:

```bash
uv run ruff check --fix .
uv run ruff format kubeflow
```

## 2. Import boundaries

```bash
make lint-imports
```

Confirms cross-component imports respect the contracts in `.importlinter`.
If you added a new component, add an isolation contract first.

## 3. OpenAPI spec

```bash
make verify-openapi
```

Only needed if you modified `openapi.yaml`. Validates against OpenAPI 3.x schema.

## 4. Tests

Run targeted tests for the files you changed:

```bash
uv run pytest -q kubeflow/<component>/path/to/<file>_test.py
```

For full test suite with coverage:

```bash
make test-python
```

Key rules:
- Unit tests in `*_test.py` files co-located with the module under test.
- No network calls in unit tests — mock external APIs.
- Follow the parametrized `TestCase` pattern from
  `kubeflow/trainer/backends/kubernetes/backend_test.py`.

## 5. Diff review

Before committing, review your diff against these standards:

- **Type hints** on all functions and return types.
- **Google-style docstrings** on all public functions with `Args`, `Returns`,
  `Raises` sections.
- **No bare `except:`** — use specific exception types.
- **No `eval()`, `exec()`, or `pickle`** on user-controlled input.
- **No secrets** in code, logs, or examples.
- **Descriptive variable names** — no single-letter names outside comprehensions.
- **Line length 100**, double quotes, spaces indent.

See `references/core-principles.md` for the full coding standards.

## 6. Commit message

Use Conventional Commits format:

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `chore`, `revert`

Scopes (from `.github/workflows/check-pr-title.yaml`):
`trainer`, `spark`, `optimizer`, `hub`, `ci`, `deps`, `docs`

Examples:
- `fix(trainer): validate polling_interval is strictly less than timeout`
- `feat(spark): add executor memory configuration option`
- `chore(docs): update backend abstraction design doc`

## 7. PR description

Include in the PR body:

1. **What this PR does / why we need it** — rationale, not just "what"
2. **Which issue(s) this PR fixes** — `Fixes #<number>` format
3. **Test plan** — commands to verify the change
4. **Checklist** — `[x] Docs included if any changes are user facing`

## Quick validation sequence

Copy-paste this block to run all checks in order:

```bash
make verify && make lint-imports && make verify-openapi && make test-python
```

Or for a faster iteration loop on a single file:

```bash
uv run ruff check path/to/file.py && \
uv run ty check path/to/file.py && \
uv run pytest -q path/to/file_test.py
```

## Checklist

- [ ] `make verify` passes (lint, format, type-check)
- [ ] `make lint-imports` passes (import boundaries)
- [ ] `make verify-openapi` passes (if `openapi.yaml` changed)
- [ ] Targeted tests pass for changed files
- [ ] No new linter warnings introduced
- [ ] Diff reviewed against coding standards
- [ ] Conventional Commit message format
- [ ] PR description includes rationale and test plan
