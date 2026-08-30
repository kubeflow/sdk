# SparkClient E2E Tests

End-to-end tests that validate Spark examples execute correctly with Kubernetes cluster and Spark Operator.

## Test Files

### **test_spark_examples.py** (7 tests)

Validates that Spark example scripts execute successfully:

**Interactive Sessions**
- `test_spark_connect_simple_example` - Validates spark_connect_simple.py runs without errors
- `test_spark_advanced_options_example` - Validates spark_advanced_options.py runs without errors
- `test_connect_existing_session_example` - Validates connect_existing_session.py (SKIPPED unless `SPARK_E2E_RUN_IN_CLUSTER=1`; requires in-cluster execution)

**Batch Jobs**
- `test_batch_job_lifecycle_example` - Validates batch_job_lifecycle.py: submits a `FileJob` (spark_job.py as the remote `file_source`), waits for `COMPLETED`, then exercises `get_job`, `list_jobs` (including a status filter), `get_job_logs`, and `delete_job`
- `test_batch_func_job_lifecycle_example` - Validates batch_func_job_lifecycle.py: submits a `FuncJob` (a Python function run as the Spark app), waits for completion, then exercises the same lifecycle APIs as above
- `test_batch_failed_job_example` - Validates batch_failed_job.py: submits a `FileJob` expected to fail, waits for `FAILED` status, and verifies `get_job`, `get_job_logs`, and `delete_job` still work against a failed job
- `test_batch_job_options_example` - Validates batch_job_options.py: submits a `FileJob` with `Name`, `Labels`, `Annotations`, `NodeSelector`, and `Toleration` options, then verifies those options were applied to the underlying `SparkApplication` CR before deleting the job

`spark_job.py` is not a standalone example — it's the simple Spark application used as the remote `file_source` for the batch job examples (`batch_job_lifecycle.py`, `batch_job_options.py`).

## Prerequisites

1. Kind cluster with Spark Operator installed:
   ```bash
   ./hack/e2e-setup-cluster.sh
   ```

2. Kubectl context set to the Kind cluster:
   ```bash
   kubectl config use-context kind-spark-test
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
# Interactive sessions
python3 examples/spark/spark_connect_simple.py
python3 examples/spark/spark_advanced_options.py
python3 examples/spark/demo_existing_sparkconnect.py  # requires manual port-forward
python3 examples/spark/connect_existing_session.py     # requires an existing Spark Connect session
python3 examples/spark/test_connect_url.py

# Batch job lifecycle
python3 examples/spark/batch_job_lifecycle.py
python3 examples/spark/batch_func_job_lifecycle.py
python3 examples/spark/batch_failed_job.py
python3 examples/spark/batch_job_options.py
```

## Test Configuration

Tests use the following configuration:

- **Cluster name**: `spark-test` (via `SPARK_TEST_CLUSTER` env var)
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
- Kubernetes versions: 1.30.0, 1.31.0, 1.32.3
- Python version: 3.11

**Tests:**
- Validates Spark interactive session examples execute successfully
- Validates Spark batch job examples execute successfully
- Creates Kind cluster with Spark Operator
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
# Setup test cluster
bash hack/e2e-setup-cluster.sh

# Run example validation tests
python -m pytest test/e2e/spark/test_spark_examples.py -v

# Cleanup
bash hack/e2e-setup-cluster.sh --delete
```
