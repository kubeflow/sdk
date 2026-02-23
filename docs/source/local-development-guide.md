# Local Development Guide

## Introduction

Local development in the Kubeflow SDK allows data scientists and ML engineers to prototype, debug, and validate training logic on their local workstations before scaling to production Kubernetes clusters. The SDK is built on the philosophy of "same training code, different backend," ensuring that the core training function remains unchanged regardless of whether it is running in a local process, a container, or a remote cluster.

By providing a consistent API across different environments, the Kubeflow SDK enables a seamless transition from a laptop to a Kubernetes cluster. This staged approach reduces the risks associated with environment mismatches and improves the overall reproducibility of machine learning workflows.

## Backend Overview

### Local Process Backend

The Local Process Backend runs training jobs as standard Python processes on the host machine. It is the most lightweight option and does not require a container runtime.

*   **Setup Requirements**: A local Python environment with the required dependencies installed.
*   **When to Use**: Quick prototyping, debugging simple training scripts, or when container runtimes are unavailable.
*   **Limitations**: Minimal isolation; depends on the host's Python environment and installed packages; no resource constraints (CPU/GPU) enforcement.

**Minimal Example**:

```python
from kubeflow.trainer import TrainerClient, LocalProcessBackendConfig

# Initialize client with Local Process Backend
client = TrainerClient(backend_config=LocalProcessBackendConfig())
```

### Container Backend (Docker)

The Container Backend uses Docker to run training jobs in isolated environments. It ensures that the training code runs in the same environment (OS, libraries, drivers) that will be used in production.

*   **Setup Requirements**: Docker Engine installed and running on the host machine.
*   **When to Use**: Validating environment dependencies, testing multi-node training locally, and ensuring consistency with production images.
*   **Limitations**: Requires Docker daemon; slightly higher startup overhead than local processes.

**Minimal Example**:

```python
from kubeflow.trainer import TrainerClient, ContainerBackendConfig

# Initialize client with Docker backend
client = TrainerClient(
    backend_config=ContainerBackendConfig(container_runtime="docker")
)
```

### Container Backend (Podman)

Similar to the Docker backend, the Podman backend provides containerized execution but utilizes Podman as the runtime. This is often preferred in environments where daemonless or rootless containers are required.

*   **Setup Requirements**: Podman installed and the Podman machine/socket started.
*   **When to Use**: Local development on macOS/Windows using Podman Machine, or in environments where Docker is not permitted.
*   **Limitations**: Requires Podman socket; initial setup of Podman machine may be required on non-Linux systems.

**Minimal Example**:

```python
from kubeflow.trainer import TrainerClient, ContainerBackendConfig

# Initialize client with Podman backend
client = TrainerClient(
    backend_config=ContainerBackendConfig(container_runtime="podman")
)
```

### Kubernetes Backend

The Kubernetes Backend submits training jobs to a remote or local Kubernetes cluster (e.g., Kind, Minikube, or a production GKE/EKS cluster). This is the final stage of development before full-scale production runs.

*   **Purpose**: Scaling to multi-node clusters, utilizing cluster-wide GPUs, and integrating with other Kubeflow components.
*   **Setup Requirements**: `kubectl` configured with access to a cluster; Kubeflow Trainer controller installed in the cluster.
*   **When to Use**: Final validation before production, large-scale training, and when local resources are insufficient.

**Minimal Example**:

```python
from kubeflow.trainer import TrainerClient, KubernetesBackendConfig

# Initialize client with Kubernetes backend
client = TrainerClient(
    backend_config=KubernetesBackendConfig(namespace="kubeflow")
)
```

## Complete Training Example

The following example demonstrates a realistic PyTorch training workflow. Notice that the `train_fn` and the job submission logic remain identical regardless of the chosen backend.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from kubeflow.trainer import TrainerClient, LocalProcessBackendConfig, CustomTrainer

# 1. Define the training function
def train_fn():
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(5):
        inputs = torch.randn(16, 10)
        labels = torch.randn(16, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 2. Configure the backend (e.g., Local Process)
backend_config = LocalProcessBackendConfig()

# 3. Initialize TrainerClient
client = TrainerClient(backend_config=backend_config)

# 4. Submit the job
job_name = client.train(
    trainer=CustomTrainer(func=train_fn),
    runtime="torch-distributed"
)

# 5. List jobs and wait for completion
print(f"Submitted job: {job_name}")
client.wait_for_job_status(name=job_name)

# 6. Stream logs
for log_line in client.get_job_logs(name=job_name, follow=True):
    print(log_line)

# 7. Clean up
client.delete_job(name=job_name)
```

## Backend Comparison Table

| Feature | Local Process | Container (Docker/Podman) | Kubernetes |
| :--- | :--- | :--- | :--- |
| **Setup Complexity** | Low | Medium | High |
| **Isolation Level** | Process-level | Container-level | Cluster-level |
| **Multi-node Support** | Single-node only | Multi-container (Simulated) | Native Multi-node |
| **GPU Support** | Via host drivers | Via NVIDIA Container Toolkit | Via K8s Device Plugin |
| **Startup Time** | Near-instant | Fast | Moderate (Pod scheduling) |
| **Production Suitability** | Development only | Staging / Validation | Production |
| **Recommended Use Case** | Rapid Prototyping | Environment Validation | Large-scale Training |

## Switching Between Backends

Switching between backends is achieved by changing only the `backend_config` passed to the `TrainerClient`. Your training function and job specifications remain unchanged.

```python
# To run as local processes
client = TrainerClient(backend_config=LocalProcessBackendConfig())

# To run in Docker containers
client = TrainerClient(backend_config=ContainerBackendConfig(container_runtime="docker"))

# To run in a Kubernetes cluster
client = TrainerClient(backend_config=KubernetesBackendConfig(namespace="kubeflow"))
```

This decoupling allows developers to maintain a single codebase for the entire ML lifecycle.

## Recommended Development Workflow

The Kubeflow SDK facilitates a staged progression to reduce development risks:

1.  **Prototype Locally**: Use `LocalProcessBackendConfig` for fast iterations on the training logic.
2.  **Validate with Containers**: Use `ContainerBackendConfig` to ensure all library dependencies, OS configurations, and hardware requirements (like CUDA) are correctly met.
3.  **Deploy to Kubernetes**: Use `KubernetesBackendConfig` for final distributed training on production-grade infrastructure.

This workflow ensures that environment-specific issues are identified early, improving the reliability of the final deployment.

## Debugging and Troubleshooting

### Docker daemon not running
*   **Issue**: `RuntimeError` stating it cannot connect to the Docker daemon.
*   **Solution**: Ensure Docker Desktop or the Docker engine is running. On macOS/Linux, verify with `docker ps`.

### Permission denied errors
*   **Issue**: Permission denied when accessing `/var/run/docker.sock`.
*   **Solution**: Run the command with appropriate permissions or add your user to the `docker` group.

### Podman machine not started
*   **Issue**: Connection failure with Podman runtime.
*   **Solution**: Run `podman machine start` and ensure the `CONTAINER_HOST` environment variable is set appropriately.

### Missing Python packages
*   **Issue**: `ImportError` during local process execution.
*   **Solution**: Ensure the local virtual environment has all dependencies installed. For container/K8s backends, ensure the packages are included in the runtime image or defined in `CustomTrainer`.

### Runtime not found
*   **Issue**: `ValueError` when specifying a runtime name.
*   **Solution**: Use `client.list_runtimes()` to see available runtimes on the current backend.

### GPU not detected
*   **Issue**: Training runs on CPU despite requesting GPU.
*   **Solution**: Verify NVIDIA drivers are installed on the host. For containers, ensure the NVIDIA Container Toolkit is configured.

### Job stuck in Running state
*   **Issue**: `wait_for_job_status` appears to hang or times out while waiting for the job to complete.
*   **Solution**: Check the logs using `client.get_job_logs()` and inspect the status of the underlying resources (containers or pods). If the job is still progressing, consider increasing the `timeout` value or adjusting the `polling_interval` parameter in `wait_for_job_status` to avoid premature `TimeoutError`s.

## Runtime Configuration

Runtimes define the environment and distribution strategy for training jobs.

*   **Listing Runtimes**: Use `client.list_runtimes()` to retrieve a list of supported runtimes for the current backend.
*   **Using a Specific Runtime**: Pass the runtime name to `client.train(runtime="torch-distributed")`.
*   **Custom Runtime Sources**: For container backends, custom runtimes can be configured via `ContainerBackendConfig(runtime_source=TrainingRuntimeSource(sources=[...]))`.
*   **Source Resolution Order**: The SDK looks for runtimes in user-provided sources first (GitHub URLs, local files), then falls back to built-in runtimes.

## Limitations and When to Transition to Kubernetes

While local backends are powerful for development, they have inherent constraints:

*   **Resource Constraints**: Local machines are limited by their local CPU, RAM, and GPU capacity.
*   **Network Isolation**: True multi-node network performance cannot be fully replicated on a single host.
*   **Management Overheads**: Managing long-running jobs on a personal workstation is impractical.

Transition to the Kubernetes backend when you require distributed training across multiple physical nodes, need to access specialized cluster resources (e.g., TPUs, specific GPU types), or want to leverage the fault-tolerance and scheduling capabilities of a Kubernetes cluster.
