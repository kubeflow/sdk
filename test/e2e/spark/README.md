# SparkClient E2E Tests

End-to-end tests that validate Spark examples execute correctly with Kubernetes cluster and Spark Operator.

## Test Files

### **test_spark_examples.py**

Validates that Spark example scripts execute successfully. Runs as subprocesses
against the ambient kubeconfig by default, or as in-cluster Kubernetes Jobs when
`SPARK_E2E_RUN_IN_CLUSTER=1` and `SPARK_E2E_RUNNER_IMAGE` are set (see `run_in_cluster.py`
and `hack/Dockerfile.spark-e2e-runner`):

- `test_spark_connect_crd_smoke` - Smoke test that the SparkConnect CRD is accepted by the API server
- `test_spark_connect_simple_example` - Validates spark_connect_simple.py runs without errors
- `test_spark_advanced_options_example` - Validates spark_advanced_options.py runs without errors
- `test_connect_existing_session_example` - Validates connect_existing_session.py (in-cluster only)
- `test_batch_job_lifecycle_example` - Validates batch_job_lifecycle.py runs without errors
- `test_batch_failed_job_example` - Validates batch_failed_job.py handles failed Spark jobs
- `test_batch_func_job_lifecycle_example` - Validates batch_func_job_lifecycle.py runs without errors
- `test_batch_job_options_example` - Validates batch_job_options.py runs without errors

A background cluster watcher (`cluster_watcher.py`) polls `SparkConnect`,
`SparkApplication`, pods, and events for diagnostics on failure.

## Prerequisites

Cluster lifecycle (Kind cluster create/delete, image loading, Spark Operator
deploy) is not owned by the SDK. It's delegated to the
[kubeflow/spark-operator](https://github.com/kubeflow/spark-operator) repo's
own Makefile, the same way `test-e2e.yaml` delegates Trainer cluster setup to
the `kubeflow/trainer` repo's Makefile. This keeps the SDK from hand-rolling
setup for a control plane it doesn't ship, and keeps client-side and
server-side testing reproducible against the same upstream tooling.

1. Check out `kubeflow/spark-operator` alongside this repo and deploy it to a
   Kind cluster using its own Makefile:
   ```bash
   git clone https://github.com/kubeflow/spark-operator ../spark-operator
   cd ../spark-operator
   make deploy KIND_CLUSTER_NAME=spark-test KIND_K8S_VERSION=v1.32.11
   export KUBECONFIG="$PWD/.kube/config"
   cd -
   ```

2. Create and label the test namespace (the chart's `charts/spark-operator-chart/ci/ci-values.yaml`
   watches the `default` namespace plus any namespace labeled `spark=enabled`):
   ```bash
   kubectl create namespace spark-test
   kubectl label namespace spark-test spark=enabled
   ```

3. Spark Operator running in the cluster

## Running Tests

### All E2E Tests
```bash
uv run pytest test/e2e/spark/ -v
```

### Specific Test
```bash
uv run pytest test/e2e/spark/test_spark_examples.py::TestSparkExamples::test_spark_connect_simple_example -v
```

### Quick Validation (No pytest)
```bash
python3 examples/spark/spark_connect_simple.py
```

## Test Configuration

Tests use the following configuration:

- **Cluster name**: `spark-test` (passed as `KIND_CLUSTER_NAME` to the spark-operator Makefile)
- **Namespace**: `spark-test` (via `SPARK_TEST_NAMESPACE` env var)

These are set automatically by the GitHub Actions workflow.

## Troubleshooting

### Tests fail with "Example not found"

**Cause:** Example scripts missing in `examples/spark/` directory

**Solution:** Verify example files exist:
```bash
ls -la examples/spark/
```

### Tests timeout or hang

**Cause:** Spark Operator not installed, cluster not ready, or session/port-forward/connect stuck.

**Solution:** Run with debug logging to see where it stops:
```bash
SPARK_E2E_DEBUG=1 uv run pytest test/e2e/spark/test_spark_examples.py -v --tb=short -s
```
`-s` shows stderr from the example subprocess (session wait, port-forward URL, connect URL). Logs include: "Waiting for session...", "Session ready...", "Port-forward svc/...", "Connecting SparkSession to sc://...".

Verify cluster setup:
```bash
kubectl get pods -n spark-operator
kubectl get deployment spark-operator-controller -n spark-operator
```

## CI/CD Integration

E2E tests are integrated into GitHub Actions and run automatically on pull requests.

### Workflow: Spark Examples E2E Test

**File:** `.github/workflows/test-spark-examples.yaml`

**Triggers:**
- Changes to `examples/spark/**`
- Changes to `kubeflow/spark/**`
- Changes to example test file
- Manual workflow dispatch

**Matrix:**
- Kubernetes versions: 1.32.11, 1.33.7, 1.34.3, 1.35.0
- Python version: 3.11

**Tests:**
- Validates Spark examples execute successfully
- Checks out `kubeflow/spark-operator` and deploys it to a Kind cluster via its own Makefile
- Runs example validation tests
- Collects logs on failure

**Duration:** ~5-10 minutes per K8s version

### Viewing CI Results

```bash
# View recent workflow runs
gh run list --workflow=test-spark-examples.yaml --repo kubeflow/sdk

# View logs for specific run
gh run view <run-id> --log --repo kubeflow/sdk
```

### Local Validation

Run the same tests locally before submitting PR:

```bash
# Setup test cluster (see Prerequisites above)
git clone https://github.com/kubeflow/spark-operator ../spark-operator
(cd ../spark-operator && make deploy KIND_CLUSTER_NAME=spark-test KIND_K8S_VERSION=v1.32.11)
export KUBECONFIG="$PWD/../spark-operator/.kube/config"
kubectl create namespace spark-test
kubectl label namespace spark-test spark=enabled

# Run example validation tests
python -m pytest test/e2e/spark/test_spark_examples.py -v

# Cleanup
(cd ../spark-operator && make kind-delete-cluster KIND_CLUSTER_NAME=spark-test)
```
