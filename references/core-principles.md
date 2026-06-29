# Core Development Principles

## 1. Maintain Stable Public Interfaces ⚠️ CRITICAL

**Always attempt to preserve function signatures, argument positions, and names for exported/public methods.**

❌ **Bad - Breaking Change:**

```python
def train_model(id, verbose=False):  # Changed from `model_id`
    pass
```

✅ **Good - Stable Interface:**

```python
def train_model(model_id: str, verbose: bool = False) -> TrainingResult:
    """Train model with optional verbose output."""
    pass
```

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings

## 2. Code Quality Standards

**All Python code MUST include type hints and return types.**

❌ **Bad:**

```python
def p(u, d):
    return [x for x in u if x not in d]
```

✅ **Good:**

```python
def filter_completed_jobs(jobs: list[str], completed: set[str]) -> list[str]:
    """Filter out jobs that are already completed.

    Args:
        jobs: List of job identifiers to filter.
        completed: Set of completed job identifiers.

    Returns:
        List of jobs that are not yet completed.
    """
    return [job for job in jobs if job not in completed]
```

**Style Requirements:**

- Line length 100, Python 3.10 target, double quotes, spaces indent
- Imports: isort via ruff; first-party is `kubeflow`; prefer absolute imports
- Naming: pep8-naming; functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`; prefix private with `_`
- Use descriptive, self-explanatory variable names. Avoid overly short or cryptic identifiers
- Break up complex functions (>20 lines) into smaller, focused functions where it makes sense
- Follow existing patterns in the codebase you're modifying

## 3. Testing Requirements

**Every new feature or bugfix MUST be covered by unit tests.**

**Test Organization:**

- Unit tests: `kubeflow/trainer/**/*_test.py` (no network calls allowed)
- Use `pytest` as the testing framework
- See `kubeflow/trainer/test/common.py` for fixtures and patterns
- Unit test structure must be consistent between each other (see `kubeflow/trainer/backends/kubernetes/backend_test.py` for reference)

**Test Structure Pattern** (following `backend_test.py`):

- Use `TestCase` dataclass for parametrized tests
- Include `name`, `expected_status`, `config`, `expected_output/error` fields
- Print test execution status for debugging
- Handle both success and exception cases in the same test function
- Use `pytest.mark.parametrize` with `TestCase` dataclass for multiple test scenarios:

```python
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": "job-1"},
            expected_output=["job-1"],
        ),
        TestCase(
            name="empty jobs list",
            expected_status=SUCCESS,
            config={"name": "empty"},
            expected_output=[],
        ),
    ],
)
def test_filter_jobs_parametrized(test_case):
    """Test job filtering with multiple scenarios."""
    result = filter_jobs(**test_case.config)
    assert result == test_case.expected_output
```

## 4. Security and Risk Assessment

**Security Checklist:**

- [ ] No `eval()`, `exec()`, or `pickle` on user-controlled input
- [ ] Proper exception handling (no bare `except:`) and use descriptive error messages
- [ ] Remove unreachable/commented code before committing
- [ ] Ensure proper resource cleanup (file handles, connections)
- [ ] No secrets in code, logs, or examples

❌ **Bad:**

```python
def load_config(path):
    with open(path) as f:
        return eval(f.read())  # ⚠️ Never eval user input
```

✅ **Good:**

```python
import yaml

def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
```

## 5. Documentation Standards

**Use Google-style docstrings with Args section for all public functions.**

❌ **Insufficient Documentation:**

```python
def submit_job(name, config):
    """Submit a job."""
```

✅ **Complete Documentation:**

```python
def submit_job(name: str, config: dict, *, priority: str = "normal") -> str:
    """Submit a training job with specified configuration.

    Args:
        name: The job name identifier.
        config: Job configuration dictionary.
        priority: Job priority level ('low', 'normal', 'high').

    Returns:
        Job ID string for tracking the submitted job.

    Raises:
        InvalidConfigError: If the configuration is invalid.
        ResourceUnavailableError: If required resources are not available.
    """
```

**Documentation Guidelines:**

- Types go in function signatures, NOT in docstrings
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Use Pydantic v2 models in `kubeflow.trainer.types` for schemas