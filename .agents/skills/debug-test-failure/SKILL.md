---
name: debug-test-failure
description: >-
  Diagnose and fix a failing test in the Kubeflow SDK. Use when a test fails
  locally or in CI and you need a structured debugging approach.
---

# Debug a Test Failure

This is a diagnostic skill — it does not generate code directly but guides
you through a structured approach to find and fix the root cause of a test
failure.

## Step 1: Reproduce the failure

Run the specific failing test in isolation:

```bash
uv run pytest -q path/to/file_test.py::test_name -v
```

If the test name is unknown, run the full test file:

```bash
uv run pytest -q path/to/file_test.py -v
```

Note the exact error message, traceback, and which assertion failed.

## Step 2: Classify the failure

| Symptom | Likely cause | Where to look |
|---------|-------------|---------------|
| `ImportError` / `ModuleNotFoundError` | Missing dep or wrong import path | `pyproject.toml`, `__init__.py` exports |
| `AttributeError` on a mock | Mock spec is stale after refactor | Test file mock setup, `spec=` argument |
| `AssertionError` on expected output | Code logic changed without test update | The module under test, recent git diff |
| `TypeError` on function call | Signature changed (new/removed param) | Function definition, callers |
| `ValueError` from backend type check | Wrong mock class used in test | Mock `spec=` and `__class__` assignment |
| `KeyError` in job_spec dict | Option writing to wrong key path | The option's `__call__` method |
| Flaky / intermittent failure | Time-dependent, ordering, or randomness | Fixtures, `uuid`, `time.sleep` |

## Step 3: Read the test

Read the failing test function and understand:

1. **What it sets up** — fixtures, mocks, test data
2. **What it calls** — the function/method under test
3. **What it asserts** — expected output, side effects, exceptions

For parametrized tests, identify which `TestCase` is failing by checking
the `name` field in the test output.

## Step 4: Read the code under test

Read the function/method that the test exercises. Compare:

- Does the function signature match what the test passes?
- Does the function return what the test expects?
- Were there recent changes to this function? (`git log -5 path/to/file.py`)

## Step 5: Check common pitfalls

### Mock-related

- `Mock(spec=KubernetesBackend)` does not make `isinstance()` return `True`.
  You need `backend.__class__ = KubernetesBackend` for type guards.
- If using `MagicMock`, ensure `type(mock).__name__` returns the right class name.
- See `kubeflow/trainer/options/kubernetes_test.py` for the canonical mock pattern.

### Fixture-related

- Shared fixtures are in `kubeflow/trainer/test/common.py`:
  - `TestCase` dataclass with `name`, `expected_status`, `config`, `expected_output`, `expected_error`
  - Status constants: `SUCCESS`, `FAILED`
- Tests should NOT share mutable state between parametrized cases.

### Import-related

- If a test imports from another component (e.g., trainer test imports spark),
  it violates import boundaries. Run `make lint-imports` to confirm.
- Use `from unittest.mock import Mock` for mocking, not real imports of
  classes you only need for `spec=`.

## Step 6: Fix and verify

After identifying the root cause:

1. Fix the minimal code needed (prefer fixing the code if the test expectation
   is correct; fix the test if the code change was intentional).
2. Run the single test:

```bash
uv run pytest -q path/to/file_test.py::test_name -v
```

3. Run the full test file to check for side effects:

```bash
uv run pytest -q path/to/file_test.py
```

4. Run lint to ensure no style issues:

```bash
uv run ruff check path/to/file.py path/to/file_test.py
```

## Step 7: Check for broader impact

If the fix changed a shared function or type:

```bash
make test-python           # Full test suite
make lint-imports          # Import boundaries
```

## When to escalate

- **Flaky test with no clear cause** — flag for human review; do not add
  `@pytest.mark.skip` without explanation.
- **Test requires network/cluster access** — it belongs in E2E tests under
  `test/e2e/`, not unit tests.
- **Upstream API model change** — use the `update-api-dep` skill to
  coordinate the multi-file update.
