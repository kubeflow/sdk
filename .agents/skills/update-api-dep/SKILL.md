---
name: update-api-dep
description: >-
  Bump an upstream API dependency (trainer, katib, spark, kubernetes). Use when
  updating kubeflow-trainer-api, kubeflow-katib-api, kubeflow-spark-api, or
  kubernetes client versions, or when vendoring updated CRDs.
---

# Bump an Upstream API Dependency

This skill coordinates the multi-file changes required when an upstream API
dependency version changes. Getting one file out of sync causes subtle runtime
failures, so follow every step.

## Before you start

1. Identify which dependency is being bumped and the target version.
2. Check release notes for breaking changes in the upstream project.
3. Read the current dependency pins:

```
pyproject.toml          # Python package versions
.agents/api/surfaces.yaml  # CRD versions and upstream pins
```

## Step 1: Update `pyproject.toml`

Edit the version constraint in `[project.dependencies]` or
`[project.optional-dependencies]`:

```toml
# Core dependencies
"kubeflow-trainer-api>=X.Y.Z",
"kubeflow-katib-api>=X.Y.Z",

# Optional (spark)
"kubeflow-spark-api>=X.Y.Z",
"pyspark-connect==X.Y.Z",
```

For the `kubernetes` client, note the existing pin comment explaining any
exclusions (e.g., `<36.0.0` for regressions).

After editing, sync the lock file:

```bash
uv lock
```

## Step 2: Update `.agents/api/surfaces.yaml`

Update the `upstream_pins` section to match the new version:

```yaml
upstream_pins:
  trainer:
    tag: vX.Y.Z
    matches_pyproject: kubeflow-trainer-api>=X.Y.Z
  katib:
    tag: vX.Y.Z
    matches_pyproject: kubeflow-katib-api>=X.Y.Z
```

Update `crds` entries if the CRD `upstream_url` points to a version-tagged path.

## Step 3: Vendor updated CRDs (if applicable)

For vendored CRDs (currently SparkConnect), download the new version:

```bash
curl -sSL <upstream_url> -o hack/crds/<filename>.yaml
```

Check `surfaces.yaml` for which CRDs use `strategy: vendored` vs
`strategy: upstream`. Only vendored CRDs need local files updated.

## Step 4: Update `openapi.yaml` (if API surface changed)

If the upstream API added, removed, or renamed fields:

1. Update the affected schemas in `openapi.yaml` to match the new API.
2. Check Python dataclasses in `kubeflow/*/types/` for alignment.

Validate:

```bash
make verify-openapi
```

## Step 5: Run tests

```bash
uv sync                    # Install updated deps
make verify                # Lint, format, type-check
make lint-imports          # Import boundaries
make test-python           # Full test suite
```

Pay special attention to tests that mock API models — they may need updated
field names or new required fields.

## Step 6: Check for breaking changes

Common breakage patterns:

| Pattern | Where to fix |
|---------|-------------|
| Renamed model field | `types/types.py` + `openapi.yaml` + tests |
| New required field | `backends/*/utils.py` (spec building) |
| Removed field | Remove from SDK types and tests |
| Changed enum values | Constants in `constants.py` |
| Client API signature change | `backends/*/backend.py` |

## Step 7: Validate everything

```bash
make verify && make lint-imports && make verify-openapi && make test-python
```

## Checklist

- [ ] `pyproject.toml` version constraint updated
- [ ] `uv lock` run (lock file regenerated)
- [ ] `.agents/api/surfaces.yaml` upstream pins updated
- [ ] CRD URLs in `surfaces.yaml` point to new version tag
- [ ] Vendored CRDs re-downloaded (if `strategy: vendored`)
- [ ] `openapi.yaml` schemas aligned with new API (if fields changed)
- [ ] Python types in `kubeflow/*/types/` aligned
- [ ] All tests pass with new dependency version
- [ ] All validation commands pass
