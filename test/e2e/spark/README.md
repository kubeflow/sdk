# SparkClient E2E Tests

End-to-end tests that validate Spark examples execute correctly with Kubernetes cluster and Spark Operator.

## Test Files

### **test_spark_examples.py** (3 tests)

Validates that Spark example scripts execute successfully:

- `test_spark_connect_simple_example` - Validates spark_connect_simple.py runs without errors
- `test_spark_advanced_options_example` - Validates spark_advanced_options.py runs without errors
- `test_demo_existing_sparkconnect_example` - Validates demo_existing_sparkconnect.py structure (SKIPPED - requires manual port-forward)

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
python3 examples/spark/spark_connect_simple.py
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
- Validates Spark examples execute successfully
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
