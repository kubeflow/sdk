AGENTS: Guide for kubeflow/sdk

## Who This Is For

- **AI agents**: Automate repository tasks with minimal context
- **Contributors**: Humans using AI assistants or working directly
- **Maintainers**: Ensure assistants follow project conventions and CI rules

## What This Document Provides

- Environment setup and canonical commands for format, lint, and tests
- Repository map and conventions to keep changes consistent
- Guardrails for PRs, CI, and releases
- Quick references for common tasks and troubleshooting

## Project Overview

**Purpose**: Kubeflow SDK provides a unified Python SDK for AI practitioners to interact with multiple Kubeflow projects via consistent APIs, focusing on user workflows over infrastructure details.

**Problem It Solves**: Reduces Kubernetes and multi-project complexity, offering simple, local-first Python interfaces for training, tuning, pipelines (planned), and model lifecycle management.

**Key Benefits**:
- Unified experience across Kubeflow projects
- Simplified AI workflows with minimal infrastructure knowledge
- Local development support (install via `pip`) with optional cluster backends

**Today's Scope**:
- **Available**: Kubeflow Trainer (train/fine-tune with different backends)
- **Planned**: Katib (HPO), Pipelines (workflows), Model Registry
- See README "Supported Kubeflow Projects" for current status

## Repository Map

```
kubeflow/trainer/           # Trainer component
├── backends/kubernetes/    # K8s backend implementation + tests
├── backends/localprocess/  # Local process backend
├── api/                   # Client API, TrainerClient
├── types/                 # Pydantic v2 data models
└── utils/                 # Shared helpers + tests
docs/                      # Diagrams and proposals
scripts/                   # Project scripts (e.g., changelog)
Root files: AGENTS.md, README.md, pyproject.toml, Makefile, CI workflows
```

## Environment & Tooling

- **Package manager**: `uv` (creates `.venv` automatically via targets)
- **Lint/format**: `ruff` (isort integrated)
- **Tests**: `pytest` with coverage
- **Build**: Hatchling (optional `uv build`)
- **Pre-commit**: Config provided and enforced in CI

## Quick Start

**Setup**:
```bash
make install-dev              # Install uv, create .venv, sync deps
```

**Verify (CI parity)**:
```bash
make verify                   # Runs ruff check --show-fixes and ruff format --check
```

**Testing**:
```bash
make test-python              # All unit tests + coverage (HTML by default)
make test-python report=xml   # XML coverage report
uv run pytest -q kubeflow/trainer/utils/utils_test.py                    # One file
uv run pytest -q kubeflow/trainer/utils/utils_test.py::test_name -k "pattern"  # One test
uv run coverage run -m pytest <path> && uv run coverage report          # Ad-hoc coverage
```

**Local lint/format**:
```bash
uv run ruff check --fix .     # Fix lint issues
uv run ruff format kubeflow   # Format code
```

**Type checking**:
```bash
uv run mypy kubeflow          # Run type checker
```

**Pre-commit**:
```bash
uv run pre-commit install                    # Install hooks
uv run pre-commit run --all-files           # Run all hooks
```

## Core Development Principles

### 1. Maintain Stable Public Interfaces ⚠️ CRITICAL

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

### 2. Code Quality Standards

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
- Line length 100, Python 3.9 target, double quotes, spaces indent
- Imports: isort via ruff; first-party is `kubeflow`; prefer absolute imports
- Naming: pep8-naming; functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`; prefix private with `_`
- Use descriptive, self-explanatory variable names. Avoid overly short or cryptic identifiers
- Break up complex functions (>20 lines) into smaller, focused functions where it makes sense
- Follow existing patterns in the codebase you're modifying

### 3. Testing Requirements

**Every new feature or bugfix MUST be covered by unit tests.**

**Test Organization:**
- Unit tests: `kubeflow/trainer/**/*_test.py` (no network calls allowed)
- Use `pytest` as the testing framework
- See `kubeflow/trainer/test/common.py` for fixtures and patterns

**Test Quality Checklist:**
- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)

```python
def test_filter_completed_jobs():
    """Test filtering completed jobs from a list."""
    jobs = ["job-1", "job-2", "job-3"]
    completed = {"job-1", "job-2"}
    
    result = filter_completed_jobs(jobs, completed)
    
    assert result == ["job-3"]
    assert len(result) == 1
```

### 4. Security and Risk Assessment

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

### 5. Documentation Standards

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

### 6. Architectural Improvements

**When you encounter code that could be improved, suggest better designs:**

❌ **Poor Design:**
```python
def process_training(data, k8s_client, storage, logger):
    # Function doing too many things
    validated = validate_data(data)
    job = k8s_client.create_job(validated)
    storage.save_metadata(job)
    logger.info(f"Created job {job.name}")
    return job
```

✅ **Better Design:**
```python
@dataclass
class TrainingJobResult:
    """Result of training job submission."""
    job_id: str
    status: str
    created_at: datetime
    
class TrainingJobManager:
    """Handles training job lifecycle operations."""
    
    def __init__(self, k8s_client: KubernetesClient, storage: Storage):
        self.k8s = k8s_client
        self.storage = storage
        
    def submit_job(self, config: TrainingConfig) -> TrainingJobResult:
        """Submit and track a new training job."""
        validated_config = self._validate_config(config)
        job = self._create_k8s_job(validated_config)
        self._save_job_metadata(job)
        return TrainingJobResult(
            job_id=job.name,
            status=job.status,
            created_at=job.created_at
        )
```

## Component: Trainer

**Client entrypoints**: `kubeflow.trainer.api.TrainerClient` and trainer definitions such as `CustomTrainer`

**Backends**:
- `localprocess`: local execution for fast iteration
- `kubernetes`: K8s-backed jobs, see `backends/kubernetes`

**Typical flow**:
1. Get runtime, define trainer, submit with `TrainerClient().train(...)`
2. `wait_for_job_status(...)` then fetch logs with `get_job_logs(...)`
3. For full example, see README "Run your first PyTorch distributed job"

**Integration patterns**:
- Follow existing patterns in `kubeflow.trainer.backends` for new backends
- Use `kubeflow.trainer.types` for data models and type definitions
- Implement proper error handling and resource cleanup
- Include comprehensive tests for backend implementations

## CI & PRs

**PR Requirements**:
- Title must follow Conventional Commits:
  - Types: `chore`, `fix`, `feat`, `revert`
  - Scopes: `ci`, `docs`, `examples`, `scripts`, `test`, `trainer`
- CI runs `make verify` and tests on Python 3.9/3.11
- Keep changes focused and minimal; align with existing style

## Releasing

**Version management**:
```bash
make release VERSION=X.Y.Z   # Updates kubeflow/__init__.py and generates changelog
```
- Do not commit secrets; verify coverage and lint pass before tagging

## Troubleshooting

- **`uv` not found**: run `make uv` or re-run `make install-dev`
- **Ruff not installed**: `make install-dev` ensures tools; or `uv tool install ruff`
- **Virtualenv issues**: remove `.venv` and re-run `make install-dev`
- **Tests failing locally but not in CI**: run `make verify` to match CI formatting and lint rules

## Quick Reference Checklist

Before submitting code changes:

- [ ] **Breaking Changes**: Verified no public API changes without deprecation
- [ ] **Type Hints**: All functions have complete type annotations and return types
- [ ] **Tests**: New functionality is fully tested with unit tests
- [ ] **Security**: No dangerous patterns (eval, bare except, resource leaks, etc.)
- [ ] **Documentation**: Google-style docstrings for public functions
- [ ] **Code Quality**: `make verify` passes (lint and format)
- [ ] **Architecture**: Suggested improvements where applicable
- [ ] **Commit Message**: Follows Conventional Commits format

## Guidance for AI Agents

**Preferred commands**: use `uv run ...` to ensure tool consistency and `.venv` usage

**Development workflow**:
1. Read existing code patterns before making changes
2. Follow the Core Development Principles above
3. Run validation commands before proposing changes
4. Use descriptive commit messages and PR descriptions

**Validation before proposing changes**:
- Lint/format: `make verify`
- Tests: `make test-python` or targeted `pytest` invocations
- Type checking: `uv run mypy kubeflow` (if available)

**Commit/PR hygiene**:
- Follow Conventional Commits in titles and messages
- Include rationale ("why") in commit messages/PR descriptions
- Do not push secrets or change git config
- Scope discipline: only modify files relevant to the task; keep diffs minimal

## Security & Privacy

- No secrets in code, logs, or examples
- Avoid external network calls in tests; prefer local fixtures/mocks
- Validate inputs and raise specific exceptions

## Community & Support

- **Slack**: `#kubeflow-ml-experience`
- **Meetings**: "Kubeflow SDK and ML Experience" (bi-weekly)
- **Issues/Discussions**: https://github.com/kubeflow/sdk
- **Contributing**: see CONTRIBUTING.md

## References

- **README**: high-level overview and example usage
- **Makefile**: authoritative commands and targets (`make help`)
- **Help**: `make help` lists available targets
