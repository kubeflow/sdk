Local Development Guide
========================

Overview
--------

The Kubeflow SDK provides first-class support for local development, allowing you to prototype and validate training logic on your workstation before scaling to a Kubernetes cluster. The SDK utilizes a unified API where the same :class:`TrainerClient` and training function are used across all backends.

Supported Backends:

*   **Local Process**: Runs training as standard Python processes on your host.
*   **Container (Docker/Podman)**: Runs training inside isolated containers to ensure environment consistency.
*   **Kubernetes**: Submits training to a local or remote Kubernetes cluster.

Backend Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Backend
     - Best For
     - Requirements
   * - **Local Process**
     - Fast prototyping, debugging
     - Python 3.10+, Dependencies installed
   * - **Container**
     - Environment validation, distributed testing
     - Docker or Podman runtime service
   * - **Kubernetes**
     - Large-scale training, GPU clusters
     - Kubernetes cluster access, Kubeflow Trainer operator

Local Process Backend
---------------------

The Local Process backend executes your code directly in the host's Python environment. This is the fastest way to get started and is ideal for early-stage development.

**When to use:**

- Rapidly iterating on training logic.
- Debugging scripts using local IDE tools.
- Scenarios where container overhead is unnecessary.

**Example:**

.. code-block:: python

   from kubeflow.trainer import TrainerClient, LocalProcessBackendConfig, CustomTrainer

   def train_fn():
       print("Training locally...")

   # Initialize with Local Process backend
   client = TrainerClient(backend_config=LocalProcessBackendConfig())

   # Submit local job
   client.train(trainer=CustomTrainer(func=train_fn))

**Key Limitations:**

- Minimal environment isolation; relies on host-level library management.
- No resource (CPU/GPU) enforcement.
- Limited to single-node execution.

Container Backend (Docker/Podman)
---------------------------------

The Container backend provides isolated execution by running your code inside a specified runtime image. It attempts to connect to available Docker or Podman services using standard socket locations.

**When to use:**

- Verifying environment consistency with production images.
- Testing distributed training logic across multiple containers on a single host.

**Example:**

By default, the SDK attempts to detect an available runtime. To use a specific runtime, set the ``container_runtime`` parameter.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, ContainerBackendConfig

   # Explicitly request Docker
   config = ContainerBackendConfig(container_runtime="docker")
   client = TrainerClient(backend_config=config)

**Distributed Training Example:**

You can simulate multi-node training on your workstation by specifying ``num_nodes``. Each node runs in a dedicated container connected via a virtual bridge network.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, ContainerBackendConfig, CustomTrainer

   client = TrainerClient(backend_config=ContainerBackendConfig())

   client.train(
       trainer=CustomTrainer(func=train_fn, num_nodes=2),
       runtime="torch-distributed"
   )

Container Host Configuration
----------------------------

If you use a non-standard setup (e.g., Colima or a custom Podman machine), you can explicitly provide the ``container_host`` socket path.

**Common Socket Paths:**

*   **macOS (Colima)**: ``unix:///Users/<user>/.colima/default/docker.sock``
*   **macOS (Podman)**: ``unix:///var/run/user/<uid>/podman/podman.sock``
*   **Linux**: ``unix:///var/run/docker.sock``
*   **Windows**: ``npipe:////./pipe/docker_engine``

**Configuration Example:**

.. code-block:: python

   import os
   from kubeflow.trainer import TrainerClient, ContainerBackendConfig

   sock_path = f"unix://{os.environ['HOME']}/.colima/default/docker.sock"
   config = ContainerBackendConfig(container_host=sock_path)
   client = TrainerClient(backend_config=config)

.. note::
   If connection errors occur, verify that your container service is active and that your user account has sufficient permissions to access the socket file.

Kubernetes Backend
------------------

The Kubernetes backend submits training jobs to a cluster. This backend is intended for large-scale training and production workflows.

**When to use:**

- Scaling horizontally across multiple physical nodes.
- Accessing cluster-managed hardware (e.g., NVIDIA GPUs, TPUs).
- Final validation before full production deployment.

**Example:**

.. code-block:: python

   from kubeflow.trainer import TrainerClient, KubernetesBackendConfig

   # Initialize with Kubernetes backend in a specific namespace
   config = KubernetesBackendConfig(namespace="kubeflow-user")
   client = TrainerClient(backend_config=config)

Switching Between Backends
--------------------------

Switching backends requires changing only the configuration object passed to :class:`TrainerClient`. The training function and job specifications remain consistent across all environments.

.. code-block:: python

   # Local execution
   client = TrainerClient(backend_config=LocalProcessBackendConfig())

   # Containerized execution
   client = TrainerClient(backend_config=ContainerBackendConfig())

   # Cluster execution
   client = TrainerClient(backend_config=KubernetesBackendConfig())

Common Operations
-----------------

Once a job is submitted, management remains consistent regardless of the underlying backend:

.. code-block:: python

   # List all jobs
   jobs = client.list_jobs()

   # Stream logs in real-time
   for line in client.get_job_logs(name=job_name, follow=True):
       print(line)

   # Wait for completion with a 1-hour timeout
   client.wait_for_job_status(name=job_name, timeout=3600)

   # Delete the job and clean up resources
   client.delete_job(name=job_name)

Troubleshooting
---------------

LocalProcess Issues
   - **ModuleNotFoundError**: Verify that all dependencies are installed in your current Python environment.
   - **Environment Conflicts**: Consider using a virtual environment or switching to the Container backend.

Docker Daemon Errors
   - **Connection Failure**: Ensure Docker Desktop or the Docker daemon is fully operational.
   - **Permission Denied**: On Linux, ensure your user is added to the ``docker`` group.

Podman Permission Issues
   - **Socket Not Found**: Ensure the Podman API service is active (e.g., ``podman system service --time=0 &``).
   - **Machine Inactive**: On non-Linux systems, verify the Podman machine is started.
