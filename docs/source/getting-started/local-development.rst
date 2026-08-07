Local Development
==================

This guide explains how to run Kubeflow TrainJobs locally using various Kubeflow SDK backends, allowing for faster iteration before deploying to a Kubernetes cluster.

Overview
--------

The Kubeflow SDK provides three backends for running TrainJobs:

.. list-table:: Backend Comparison
   :header-rows: 1
   :widths: 20 35 45

   * - Backend
     - Best For
     - Requirements
   * - **Local Process**
     - Quick prototyping, single-node testing
     - Python 3.10+
   * - **Container**
     - Multi-process distributed training (single host), reproducibility
     - Docker or Podman installed
   * - **Kubernetes**
     - Production deployments
     - K8s cluster with Trainer operator

All backends use the same ``TrainerClient`` interface - only the configuration
changes. This means you can develop locally and deploy to production with
minimal code changes.

Local Process Backend
---------------------

The fastest option for quick testing. Runs training directly as Python processes.

**When to use:**

- Rapid prototyping and debugging
- Testing training logic without container overhead
- Environments without Docker/Podman

**Example:**

.. code-block:: python

   from kubeflow.trainer import TrainerClient, LocalProcessBackendConfig
   from kubeflow.trainer import CustomTrainer

   # Configure local process backend
   backend_config = LocalProcessBackendConfig()
   client = TrainerClient(backend_config=backend_config)

   # Define your training function
   def train_model():
       import torch
       print(f"Training on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
       # Your training logic here

   # Create trainer and run
   trainer = CustomTrainer(func=train_model)
   job_name = client.train(trainer=trainer)

   # View logs
   for log_line in client.get_job_logs(name=job_name, follow=True):
       print(log_line)

**Limitations:**

- Single-node only (no distributed training)
- No container isolation

Container Backend (Docker/Podman)
---------------------------------

Run training in isolated containers with multi-process distributed training
support on a single host.

**When to use:**

- Distributed training with multiple workers on one machine
- Reproducible containerized environments
- Testing production-like setups locally

**Example with Docker:**

.. code-block:: python

   from kubeflow.trainer import TrainerClient, ContainerBackendConfig
   from kubeflow.trainer import CustomTrainer

   # Configure Docker backend
   backend_config = ContainerBackendConfig(
       container_runtime="docker",  # or "podman"
   )
   client = TrainerClient(backend_config=backend_config)

   # Same trainer works - now with multi-process support!
   trainer = CustomTrainer(
       func=train_model,
       num_nodes=4,  # Distributed across 4 containers on this host
   )
   job_name = client.train(trainer=trainer)

.. _container-host-configuration:

Container Host Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the Container backend on **macOS**, you may need to configure the
``container_host`` parameter to point to your Docker or Podman socket. This is
because the default socket path differs across operating systems.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - OS
     - Default ``container_host``
   * - Linux
     - ``unix:///var/run/docker.sock`` (Docker) or ``unix:///run/user/<UID>/podman/podman.sock`` (Podman), where ``<UID>`` is the current user's ID (run ``id -u`` to find it)
   * - macOS
     - ``unix://$HOME/.docker/run/docker.sock`` (Docker Desktop) or check ``podman machine inspect`` for Podman
   * - Windows
     - ``npipe:////./pipe/docker_engine`` (Docker Desktop)

**Example for macOS:**

.. code-block:: python

   import os

   backend_config = ContainerBackendConfig(
       container_runtime="docker",
       # macOS Docker Desktop socket path
       container_host=f"unix://{os.environ['HOME']}/.docker/run/docker.sock",
   )
   client = TrainerClient(backend_config=backend_config)

.. note::

   If you encounter ``Cannot connect to Docker daemon`` errors on macOS,
   verify the socket path by running ``docker context inspect`` and check
   the ``Host`` value in the output.

Kubernetes Backend
-------------------

Deploy training jobs to a Kubernetes cluster running the Kubeflow Trainer
operator. This is the production backend and the only one that supports true
multi-node distributed training across separate physical machines.

**When to use:**

- Production deployments
- True multi-node distributed training across a cluster
- Workloads that need Kubernetes-native scheduling, GPU allocation, or autoscaling

**Example:**

.. code-block:: python

   from kubeflow.trainer import TrainerClient, KubernetesBackendConfig
   from kubeflow.trainer import CustomTrainer

   # Configure Kubernetes backend
   backend_config = KubernetesBackendConfig(
       namespace="kubeflow",          # Target namespace for TrainJobs
       config_file="~/.kube/config",  # Optional: path to kubeconfig (defaults to standard lookup)
       context="my-cluster",          # Optional: kubeconfig context to use
   )
   client = TrainerClient(backend_config=backend_config)

   # Same trainer code, now running on a real cluster
   trainer = CustomTrainer(
       func=train_model,
       num_nodes=4,  # Distributed across 4 pods on separate nodes
   )
   job_name = client.train(trainer=trainer)

**Limitations:**

- Requires a Kubernetes cluster with the Kubeflow Trainer operator installed
- Requires valid kubeconfig credentials with appropriate RBAC permissions

Switching Between Backends
--------------------------

A key benefit of the Kubeflow SDK is seamless backend switching. Your training code
stays the same - only the backend configuration changes:

.. code-block:: python

   # Development: Use local process for fast iteration
   from kubeflow.trainer import LocalProcessBackendConfig
   backend_config = LocalProcessBackendConfig()

   # Testing: Switch to Docker for distributed testing
   from kubeflow.trainer import ContainerBackendConfig
   backend_config = ContainerBackendConfig(container_runtime="docker")

   # Production: Deploy to Kubernetes
   from kubeflow.trainer import KubernetesBackendConfig
   backend_config = KubernetesBackendConfig(namespace="kubeflow")

   # Same client and trainer code works with all backends!
   client = TrainerClient(backend_config=backend_config)
   job_name = client.train(trainer=trainer)

Common Operations
-----------------

These operations work identically across all backends:

**List Jobs:**

.. code-block:: python

   jobs = client.list_jobs()
   for job in jobs:
       print(f"{job.name}: {job.status}")

**View Logs:**

.. code-block:: python

   # Follow logs in real-time
   for log_line in client.get_job_logs(name=job_name, follow=True):
       print(log_line)

**Wait for Completion:**

.. code-block:: python

   job = client.wait_for_job_status(
       name=job_name,
       timeout=3600,  # 1 hour timeout
   )

**Delete Jobs:**

.. code-block:: python

   client.delete_job(name=job_name)

Troubleshooting
---------------

**Local Process Backend:**

- ``ModuleNotFoundError``: Ensure dependencies are installed in current environment
- Training hangs: Check for infinite loops in your training function

**Container Backend:**

- ``Cannot connect to Docker daemon``: Start Docker/Podman service. On macOS,
  verify the socket path — see :ref:`container-host-configuration`.
- Image pull errors: Check network connectivity and image registry access
- Permission denied: For Podman, ensure rootless mode is configured

**Kubernetes Backend:**

- ``ConfigException``: Verify the ``config_file`` path and ``context`` are correct
- TrainJob stuck in ``Created``: Check that the Trainer operator is installed and pods are schedulable

Next Steps
----------

- :doc:`../train/custom-training` - Define your trainers
- :doc:`../train/distributed` - Scale across nodes
- `Kubeflow Trainer Docs <https://www.kubeflow.org/docs/components/trainer/>`_ - Full documentation
