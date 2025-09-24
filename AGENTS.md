AGENTS: Quick Guide for kubeflow/sdk

- Setup: use `uv`. Create venv and sync deps with `make install-dev`.
- Lint/format (CI parity): `make verify` (runs `ruff check --show-fixes` and `ruff format --check`).
- One-off lint/format locally: `uv run ruff check --fix .` then `uv run ruff format kubeflow`.
- Pre-commit: `uv run pre-commit install`; run all hooks with `uv run pre-commit run --all-files`.
- Run all unit tests + coverage: `make test-python` (HTML by default; XML with `make test-python report=xml`).
- Run tests for one file: `uv run pytest -q kubeflow/trainer/utils/utils_test.py`.
- Run a single test: `uv run pytest -q kubeflow/trainer/utils/utils_test.py::test_name -k "pattern"`.
- Coverage for ad-hoc runs: `uv run coverage run -m pytest <path>` then `uv run coverage report`.
- Packaging: project uses Hatchling; optional build with `uv build`.

Code style (ruff manages lint + format)
- Line length 100; target Python 3.9; double quotes; spaces indent; docstring code wrapped at 100.
- Imports: isort via ruff; first-party is `kubeflow`; combine `as` imports; force sort within sections; prefer absolute imports.
- Naming: pep8-naming enforced; functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`; prefix private with `_`.
- Types: annotate public APIs and tests; avoid `Any`; include return types; prefer `TypedDict`, `Literal`, `Enum`; use Pydantic v2 models in `kubeflow.trainer.types` for data schemas.
- Errors: raise specific exceptions; avoid bare `except`; use `raise ... from err` for chaining; validate inputs early (Pydantic when applicable).
- Tests: place under `kubeflow/trainer/**` as `*_test.py`; use pytest style and fixtures (see `kubeflow/trainer/test/common.py`); avoid external I/O in unit tests.
- CI: PR titles must follow Conventional Commits (types: chore, fix, feat, revert; scopes: ci, docs, examples, scripts, test, trainer). CI runs `make verify` and tests on 3.9/3.11.
- Help: `make help` lists available targets.
