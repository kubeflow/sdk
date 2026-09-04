---
name: add-k8s-option
description: >-
  Add a new Kubernetes training option to the Kubeflow Trainer SDK. Use when
  asked to add a K8s option, training option, or job spec modifier.
---

# Add a Kubernetes Training Option

This skill walks through adding a new callable option that mutates the TrainJob
spec during `KubernetesBackend.train()`. Every option follows the same pattern:
a `@dataclass` with a `__call__(self, job_spec, trainer, backend)` method.

## Before you start

1. Read the canonical option file to understand the pattern:

```
kubeflow/trainer/options/kubernetes.py
```

2. Read the existing test file for the parametrized test style:

```
kubeflow/trainer/options/kubernetes_test.py
```

3. Confirm which `job_spec` key your option should write to. Common keys:
   - `metadata` — labels, annotations, name
   - `spec.trainer` — command, args overrides
   - `spec.runtimePatches` — structured patches keyed by manager

## Step 1: Define the dataclass

Add your option class in `kubeflow/trainer/options/kubernetes.py`.

Follow this structure (using `Labels` as the simplest reference):

```python
@dataclass
class YourOption:
    """One-line summary of what this option does.

    Supported backends:
        - Kubernetes

    Args:
        your_field: Description of the field.
    """

    your_field: <type>

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: CustomTrainer | BuiltinTrainer | None,
        backend: RuntimeBackend,
    ) -> None:
        """Apply the option to the job specification.

        Args:
            job_spec: Job specification dictionary to modify.
            trainer: Optional trainer instance for context.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support this option.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"YourOption is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        # Mutate job_spec here
```

Key rules:
- Import `KubernetesBackend` inside `__call__` (avoids circular imports).
- Always guard with `isinstance(backend, KubernetesBackend)`.
- Use `job_spec.setdefault(...)` to safely nest into the spec dict.
- If the option restricts trainer type (e.g. `CustomTrainerContainer` only),
  add a second `isinstance` check on `trainer`.

## Step 2: Export the option

Add the new class to `kubeflow/trainer/options/__init__.py`:

1. Add it to the import block from `.kubernetes`.
2. Add it to the `__all__` list under `# Kubernetes options`.

## Step 3: Write tests

Add tests in `kubeflow/trainer/options/kubernetes_test.py`.

Follow the existing pattern:

```python
class TestYourOption:
    def test_applies_to_kubernetes_backend(self, mock_kubernetes_backend):
        option = YourOption(your_field=<value>)
        job_spec: dict = {}
        option(job_spec, None, mock_kubernetes_backend)
        assert job_spec[<expected_key>] == <expected_value>

    def test_rejects_non_kubernetes_backend(self, mock_localprocess_backend):
        option = YourOption(your_field=<value>)
        with pytest.raises(ValueError, match="not compatible"):
            option({}, None, mock_localprocess_backend)
```

Use the existing `mock_kubernetes_backend` and `mock_localprocess_backend`
fixtures defined at the top of the test file.

## Step 4: Update OpenAPI catalog

If your option introduces a new schema, add it under
`components.schemas` in `openapi.yaml`. This is a logical SDK catalog
(not REST), so model it as a schema the agent can discover.

## Step 5: Validate

Run these commands and fix any failures before proposing changes:

```bash
uv run ruff check kubeflow/trainer/options/kubernetes.py
uv run ruff check kubeflow/trainer/options/kubernetes_test.py
uv run pytest -q kubeflow/trainer/options/kubernetes_test.py
make lint-imports
make verify-openapi
```

## Checklist

- [ ] `@dataclass` with Google-style docstring listing `Args`
- [ ] `__call__` signature: `(self, job_spec, trainer, backend) -> None`
- [ ] Backend type guard with `isinstance(backend, KubernetesBackend)`
- [ ] Import of `KubernetesBackend` inside `__call__` (not at module level)
- [ ] Exported in `kubeflow/trainer/options/__init__.py`
- [ ] Tests: success path + rejection on wrong backend
- [ ] `openapi.yaml` updated if new schema introduced
- [ ] All validation commands pass
