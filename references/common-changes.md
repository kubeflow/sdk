# Common Change Patterns

Copy structure from these references. See [core-principles.md](core-principles.md) for code quality and test style.

### Adding a Kubernetes training option

- **Reference:** `kubeflow/trainer/options/kubernetes.py` (e.g. `Labels`, `PodTemplateOverrides`)
- **Also see:** `kubeflow/trainer/options/common.py`, `kubeflow/trainer/options/kubernetes_test.py`
- **Files:** option module under `kubeflow/trainer/options/`, `kubeflow/trainer/options/__init__.py` exports, co-located `*_test.py`; options must validate backend type in `__call__`

### Extending an execution backend

- **Reference:** `kubeflow/trainer/backends/kubernetes/backend.py` (`KubernetesBackend.train`, `_get_trainjob_spec`)
- **Also see:** `kubeflow/trainer/backends/base.py` (`RuntimeBackend` ABC), `kubeflow/trainer/backends/kubernetes/utils.py`
- **Files:** backend module under `kubeflow/trainer/backends/`, co-located `*_test.py`; register via `TrainerClient` backend config in `kubeflow/trainer/api/trainer_client.py`

### Adding trainer types or schemas

- **Reference:** `kubeflow/trainer/types/types.py` (`CustomTrainer`, `TrainJob`, `Runtime`, …)
- **Also see:** `kubeflow/trainer/types/types_test.py`
- **Files:** `kubeflow/trainer/types/types.py`, exports in `kubeflow/trainer/types/__init__.py` and `kubeflow/trainer/__init__.py` if public API

### Adding a parametrized backend test

- **Reference:** `kubeflow/trainer/backends/kubernetes/backend_test.py` (`TestCase` pattern)
- **Also see:** `kubeflow/trainer/test/common.py` (`TestCase`, `SUCCESS`/`FAILED` fixtures)
- **Files:** co-located `*_test.py` next to the module under test; no network calls in unit tests

### Adding a new SDK client module

- **Reference:** `kubeflow/spark/api/spark_client.py` or `kubeflow/optimizer/api/optimizer_client.py`
- **Also see:** matching `*_test.py`, `kubeflow/<component>/types/`, `kubeflow/<component>/backends/kubernetes/` if cluster-backed
- **Files:** `api/` client, `types/`, backend under `backends/`, exports in package `__init__.py`
