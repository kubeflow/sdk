# Spark Examples

This directory contains examples for using the Kubeflow Spark SDK.

The Spark SDK supports two ways to run Spark, matching the [Spark SDK documentation](https://sdk.kubeflow.org/en/latest/spark/index.html):

- **Interactive Sessions** - Connect to Spark from a notebook or script using Spark Connect.
- **Batch Jobs** - Submit Spark applications as managed Kubernetes workloads.

For the full documentation, see [Interactive Sessions](https://sdk.kubeflow.org/en/latest/spark/sessions.html), [Batch Jobs](https://sdk.kubeflow.org/en/latest/spark/batch-jobs.html), and [Job Lifecycle](https://sdk.kubeflow.org/en/latest/spark/lifecycle.html).

## Examples

### Interactive Sessions

- **spark_connect_simple.ipynb** - Basic SparkClient usage with simple API
- **spark_advanced_options.ipynb** - Advanced configuration with Driver/Executor objects
- **demo_existing_sparkconnect.ipynb** - Connect to existing SparkConnect cluster
- **connect_existing_session.ipynb** - Connect to an existing Spark Connect session through `base_url`
- **test_connect_url.py** - Test URL-based connection to Spark Connect

### Batch Jobs

- **batch_job_lifecycle.ipynb** - Submit a `FileJob` and exercise the batch job lifecycle (`get_job`, `list_jobs`, `get_job_logs`, `delete_job`)
- **batch_func_job_lifecycle.ipynb** - Submit a `FuncJob` and exercise the batch job lifecycle
- **batch_failed_job.ipynb** - Submit a `FileJob` expected to fail and inspect the `FAILED` state
- **batch_job_options.ipynb** - Submit a batch job with Kubernetes options (labels, annotations, node selector, tolerations, custom name)
  See the [Options Reference](https://sdk.kubeflow.org/en/latest/spark/options.html) for details on Kubernetes options.
- **spark_job.py** - A simple Spark application used as the remote `file_source` for the batch job examples

## Prerequisites

Install spark dependencies:
```bash
uv pip install kubeflow[spark]
```

The Spark examples run against a Kubernetes cluster with the Spark Operator installed. Batch job submission requires a `spark-operator-spark` ServiceAccount in the target namespace with the required SparkApplication RBAC permissions. See the [Spark SDK docs](https://sdk.kubeflow.org/en/latest/spark/index.html) for prerequisites.


## Running Examples

```bash
# Run from repository root

# Launch Jupyter to explore and run the interactive notebooks:
jupyter lab examples/spark/

# Or execute headlessly with papermill:
uv run papermill examples/spark/spark_connect_simple.ipynb /dev/null
```

## Interactive Session APIs

Interactive session examples use `SparkClient` and `connect()`:

- `SparkClient()` - Create a client for the Kubernetes cluster
- `connect()` - Create a new Spark Connect session, or connect to an existing server when `base_url` is provided
- `list_sessions()` / `get_session()` - Inspect Spark Connect sessions
- `get_session_logs()` - Read session logs
- `delete_session()` - Delete a Spark Connect session when you are done with it

See the [Spark SDK API reference](https://sdk.kubeflow.org/en/latest/spark/api.html) for the current API surface.

## Batch Job APIs

Batch job examples use `submit_job()` with either `FileJob` or `FuncJob`, then manage the job with the lifecycle APIs:

- `submit_job()` - Submit a Spark application as a managed Kubernetes workload
- `get_job()` / `list_jobs()` - Inspect batch jobs
- `wait_for_job_status()` - Wait for a job to reach a target `SparkJobStatus`
- `get_job_logs()` - Read driver pod logs
- `delete_job()` - Delete a batch job

See the [Spark SDK API reference](https://sdk.kubeflow.org/en/latest/spark/api.html) for the current API surface.
