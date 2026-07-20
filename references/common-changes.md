# Common Change Patterns

Copy structure from these references. See [core-principles.md](core-principles.md) for code quality and test style. For step-by-step procedural guidance, use the linked skills.

### Adding a Kubernetes training option

- **Skill:** `.cursor/skills/add-k8s-option/` (full procedural walkthrough)
- **Reference:** `kubeflow/trainer/options/kubernetes.py` (e.g. `Labels`, `RuntimePatch`)
- **Also see:** `kubeflow/trainer/options/common.py`, `kubeflow/trainer/options/kubernetes_test.py`
- **Files:** option module under `kubeflow/trainer/options/`, `kubeflow/trainer/options/__init__.py` exports, co-located `*_test.py`; options must validate backend type in `__call__`

### Extending an execution backend

- **Skill:** `.cursor/skills/add-backend/` (full procedural walkthrough)
- **Reference:** `kubeflow/trainer/backends/kubernetes/backend.py` (`KubernetesBackend.train`, `_get_trainjob_spec`)
- **Also see:** `kubeflow/trainer/backends/base.py` (`RuntimeBackend` ABC), `kubeflow/trainer/backends/kubernetes/utils.py`
- **Files:** backend module under `kubeflow/trainer/backends/`, co-located `*_test.py`; register via `TrainerClient` backend config in `kubeflow/trainer/api/trainer_client.py`

### Bumping an upstream API dependency

- **Skill:** `.cursor/skills/update-api-dep/` (full procedural walkthrough)
- **Reference:** `pyproject.toml` (dependency pins), `.agents/api/surfaces.yaml` (upstream pins and CRD URLs)
- **Files:** `pyproject.toml`, `uv.lock`, `.agents/api/surfaces.yaml`, `openapi.yaml` (if fields changed), vendored CRDs under `hack/crds/` (if `strategy: vendored`), Python types in `kubeflow/*/types/`

### Creating a KEP proposal

- **Skill:** `.cursor/skills/add-kep-proposal/` (full procedural walkthrough)
- **Reference:** `proposals/107-spark-client/` (`kep.yaml` + `README.md`)
- **Files:** new directory under `proposals/<number>-<short-name>/` with `kep.yaml` and `README.md`

### Adding trainer types or schemas

- **Reference:** `kubeflow/trainer/types/types.py` (`CustomTrainer`, `TrainJob`, `Runtime`, …)
- **Also see:** `kubeflow/trainer/types/types_test.py`
- **Files:** `kubeflow/trainer/types/types.py`, exports in `kubeflow/trainer/types/__init__.py` and `kubeflow/trainer/__init__.py` if public API

### Adding a parametrized backend test

- **Reference:** `kubeflow/trainer/backends/kubernetes/backend_test.py` (`TestCase` pattern)
- **Also see:** `kubeflow/trainer/test/common.py` (`TestCase`, `SUCCESS`/`FAILED` fixtures)
- **Files:** co-located `*_test.py` next to the module under test; no network calls in unit tests

### Adding a new SDK client module

- **Skill:** `.cursor/skills/add-sdk-client/` (full procedural walkthrough)
- **Reference:** `kubeflow/spark/api/spark_client.py` or `kubeflow/optimizer/api/optimizer_client.py`
- **Also see:** matching `*_test.py`, `kubeflow/<component>/types/`, `kubeflow/<component>/backends/kubernetes/` if cluster-backed
- **Files:** `api/` client, `types/`, backend under `backends/`, exports in package `__init__.py`
