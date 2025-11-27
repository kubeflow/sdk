# Container Backend for Optimizer

Local hyperparameter optimization backend that runs trials in Docker/Podman containers without requiring a Kubernetes cluster.

## Architecture

```
ContainerBackend → Optuna (TPE) + TrainerBackend (Container) → Docker/Podman
```

**Key Components:**
- `backend.py` - Core backend implementation
- `storage.py` - Local JSON file persistence
- `types.py` - Configuration and type definitions

## Files

### `backend.py` (~500 lines)
Main backend implementation implementing the `OptimizerBackend` interface.

**Key Methods:**
- `optimize()` - Creates optimization job, initializes Optuna study, runs trial loop
- `_run_trial()` - Executes single trial: sample HPs → run container → extract metrics
- `_extract_metrics_from_trial()` - Parses logs for `metric_name: value` patterns
- `get_best_results()` - Returns best hyperparameters and metrics
- `get_job_logs()` - Aggregates logs from all trial containers
- `delete_job()` - Cleanup job state and containers

**Integration Points:**
- **Optuna**: Uses TPE sampler for adaptive hyperparameter sampling
- **TrainerBackend**: Delegates trial execution to `TrainerClient` with `ContainerBackend`
- **ThreadPoolExecutor**: Manages parallel trial execution
- **Storage**: Persists job/trial state to JSON files

**Error Handling:**
- Validates search space parameters against Optuna types
- Handles container failures gracefully
- Continues optimization on trial failures (logs errors)

### `storage.py` (~200 lines)
Local file system storage for optimization state.

**Key Classes:**
- `ExperimentStorage` - Manages optimization job metadata and state
- `TrialStorage` - Handles individual trial results

**File Structure:**
```
{storage_path}/{job_name}/
├── experiment.json    # Job spec, status, best trial
├── optuna.db         # Optuna SQLite study
└── trials/
    └── {trial_name}.json  # Trial parameters, metrics
```

**Key Methods:**
- `create_experiment()` - Initializes job directory and metadata
- `get_experiment()` - Loads job state from JSON
- `update_experiment_status()` - Updates trials and best trial
- `save_trial()` / `get_trial()` - Trial CRUD operations

### `types.py` (~100 lines)
Configuration and data types for container backend.

**Key Classes:**
- `ContainerBackendConfig` - Backend configuration
  - `storage_path` - Where to store job state (default: `~/.kubeflow/optimizer`)
  - `max_parallel_trials` - Concurrent trial limit (default: 1)
  - `pull_policy` - Container image pull policy
  - `container_runtime` - Docker/Podman (auto-detected)

**Validation:**
- Ensures valid pull policies (`Always`, `IfNotPresent`, `Never`)
- Validates `max_parallel_trials` > 0
- Expands `~` in paths

## Implementation Details

### Optimization Flow
1. **Job Creation** - Create experiment storage + Optuna study (SQLite)
2. **Trial Loop** (up to `num_trials`):
   - Optuna samples hyperparameters using TPE
   - Create trial storage with parameter assignments
   - Launch container via TrainerBackend with hyperparameters
   - Wait for completion or failure
   - Extract metrics from container logs
   - Report results to Optuna
3. **Completion** - Update best trial, persist final state

### Metric Extraction
Parses container logs for patterns:
```python
print(f"accuracy: 0.95")  # Extracted as accuracy=0.95
print(f"loss: 0.05")      # Extracted as loss=0.05
```

Uses regex: `r"(\w+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"` to find metric values.

### Parallel Execution
- Uses `ThreadPoolExecutor` with `max_parallel_trials` workers
- Submits trials as futures, waits for completion
- Continues on individual trial failures

### State Persistence
- **Optuna DB**: SQLite database stores study history for resumption
- **JSON Files**: Store Kubeflow-compatible experiment/trial metadata
- **Atomic Updates**: Uses file locks for concurrent access safety

## Design Decisions

### Why Optuna?
- Mature adaptive sampling algorithms (TPE, CMA-ES, etc.)
- Handles parameter types (uniform, loguniform, categorical)
- Built-in pruning and resumption support
- SQLite persistence for interrupted jobs

### Why TrainerBackend Integration?
- Reuses existing container orchestration logic
- Consistent with K8s backend architecture
- Leverages runtime management (image pull, cleanup, etc.)
- Unified training job interface

### Why Local JSON Files?
- Simple, human-readable state inspection
- No external dependencies (vs K8s CRDs)
- Easy debugging and troubleshooting
- Compatible with existing SDK patterns

## Dependencies

```python
optuna>=3.0.0          # Hyperparameter optimization
docker>=6.0.0          # Container runtime (optional, auto-detected)
```

## Testing

See `tests/optimizer/backends/container/`:
- `test_config.py` - Configuration validation tests
- `test_backend.py` - Backend unit tests (mocked dependencies)
- `test_integration.py` - End-to-end tests with real containers

## Usage Example

```python
from kubeflow.optimizer import OptimizerClient
from kubeflow.optimizer.backends.container import ContainerBackend, ContainerBackendConfig

backend = ContainerBackend(ContainerBackendConfig(
    storage_path="~/.kubeflow/optimizer/my-job",
    max_parallel_trials=3,
))

client = OptimizerClient(backend=backend)

job_name = client.optimize(
    trial_template=template,
    search_space={"lr": Search.uniform(0.001, 0.1)},
    objectives=[Objective(metric="accuracy", direction="maximize")],
    trial_config=TrialConfig(num_trials=10, parallel_trials=3),
)

results = client.get_best_results(job_name)
```

## Limitations

- **Scalability**: Limited by local machine resources (CPU, RAM, disk)
- **Parallel Trials**: Typically 1-10 concurrent trials (vs unlimited on K8s)
- **Metric Extraction**: Simple regex parsing (no structured logging yet)
- **Resume**: Basic support via Optuna DB (no distributed coordination)
