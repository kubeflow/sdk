# Local Hyperparameter Optimization - Container Backend

## Overview

Enables local hyperparameter optimization without requiring a Kubernetes cluster:
- Runs trials in isolated Docker/Podman containers
- Uses Optuna TPE for adaptive sampling
- Stores state locally in JSON files
- Supports parallel trial execution

## Architecture

```
OptimizerClient → ContainerBackend → Optuna (TPE) + TrainerBackend → Docker Containers
```


## Key Implementation Details

### 1. Backend Configuration
```python
ContainerBackendConfig(
    storage_path="~/.kubeflow/optimizer/simple-example",
    max_parallel_trials=1,  # Concurrent trial limit
    pull_policy="IfNotPresent",
)
```

### 2. Training Function Requirements
```python
def train_fn(learning_rate: float, batch_size: int):
    # Parameters must match search space names
    print(f"accuracy: {accuracy:.4f}")  # Metric extraction format
    return accuracy
```

### 3. Search Space Definition
```python
search_space = {
    "learning_rate": Search.uniform(min=0.001, max=0.1),
    "batch_size": Search.choice([32, 64]),
}
```

### 4. Optimization Flow
1. **Job Creation** - Creates metadata + Optuna study
2. **Trial Loop** - Optuna samples → Container runs → Metrics extracted
3. **Adaptive Learning** - TPE learns from results for next trial

### 5. File Structure
```
~/.kubeflow/optimizer/simple-example/
├── <job_name>/
│   ├── experiment.json          # Job metadata
│   ├── optuna.db                # Optuna SQLite DB
│   └── trials/
│       └── <job>-trial-*.json   # Trial results
```



## Prerequisites

```bash
cd sdk-narayanasabari
pip install -e '.[docker]' optuna
```

Requires Docker/Podman running: `docker ps`

## Running

```bash
python examples/optimizer/simple-local-example.py
```

First run downloads `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` (~5GB, one-time).



## Key APIs

### Backend Configuration
```python
ContainerBackendConfig(
    pull_policy="IfNotPresent",              # Image pull policy
    storage_path="~/.kubeflow/optimizer",    # State storage
    max_parallel_trials=1,                   # Concurrency limit
)
```

### Search Space
```python
Search.uniform(min, max)          # Continuous uniform
Search.loguniform(min, max)       # Log-uniform
Search.choice([...])              # Categorical
```

### Optimization
```python
optimizer.optimize(
    trial_template=TrainJobTemplate(...),
    search_space={...},
    objectives=[Objective(metric="accuracy", direction="maximize")],
    trial_config=TrialConfig(num_trials=3, parallel_trials=1),
)
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Docker not running | `docker ps` should succeed |
| Import errors | `pip install -e '.[docker]' optuna` |
| SQLite errors | Normal on resume - loads existing study |
| No metrics extracted | Print as `metric_name: value` |

## Backend Comparison

| Feature | Container | Kubernetes |
|---------|-----------|------------|
| Setup | Simple (Docker) | Complex (K8s cluster) |
| Cost | Free | Cloud costs |
| Scalability | Local resources | Cluster capacity |
| Use Case | Dev/testing | Production |
