End-to-End Workflow Tutorial
============================

This tutorial walks you through a complete AI/ML workflow using the three core
Kubeflow SDK clients together:

1. **Train** a model using ``TrainerClient``
2. **Optimize** hyperparameters using ``OptimizerClient``
3. **Register** the best model using ``ModelRegistryClient``

By the end, you'll understand how these components connect and how to build
reproducible, end-to-end ML pipelines with the Kubeflow SDK.

.. note::

   **Steps 1–2** (training) can run locally using Docker or Podman — no
   Kubernetes cluster required. **Steps 3–4** (optimization and model
   registration) require a Kubernetes cluster with Kubeflow Katib and
   Model Registry deployed. See the
   :doc:`installation guide <getting-started/installation>` for cluster setup.

----

Prerequisites
-------------

.. code-block:: bash

   # Install the Kubeflow SDK with all components
   pip install 'kubeflow[hub]'

For **local training** (Steps 1–2) without a Kubernetes cluster:

.. code-block:: bash

   # Docker backend
   pip install 'kubeflow[docker]'

   # Or Podman backend
   pip install 'kubeflow[podman]'

----

Step 1: Define a Training Function
-----------------------------------

Start by writing a standard PyTorch training function. The Kubeflow SDK serializes
this function and runs it inside containers — no Dockerfiles required.

.. code-block:: python

   def train_fn(learning_rate: str, num_epochs: str):
       """Train a simple model for demonstration."""
       import os
       import torch
       import torch.distributed as dist

       # Initialize distributed training
       dist.init_process_group(backend="gloo")
       rank = dist.get_rank()
       world_size = dist.get_world_size()

       # Create a simple model
       model = torch.nn.Linear(10, 1)
       optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))

       # Simulate training
       for epoch in range(int(num_epochs)):
           x = torch.randn(32, 10)
           y = torch.randn(32, 1)
           loss = torch.nn.functional.mse_loss(model(x), y)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

       # Report final loss from rank 0
       if rank == 0:
           final_loss = loss.item()
           print(f"loss={final_loss:.4f}")

----

Step 2: Train the Model (Local or Kubernetes)
----------------------------------------------

Use ``TrainerClient`` to run your training function. For local development, use
``ContainerBackendConfig`` to run in Docker/Podman containers without a Kubernetes
cluster.

.. code-block:: python

   from kubeflow.trainer import (
       TrainerClient,
       ContainerBackendConfig,
       CustomTrainer,
       TrainJobTemplate,
   )

   # Create a TrainJob template (reused for optimization later)
   template = TrainJobTemplate(
       runtime="torch-distributed",
       trainer=CustomTrainer(
           func=train_fn,
           func_args={"learning_rate": "0.01", "num_epochs": "5"},
           num_nodes=2,
           resources_per_node={"cpu": 2},
       ),
   )

   # Use local containers (no Kubernetes needed)
   client = TrainerClient(backend_config=ContainerBackendConfig())

   # Launch the training job
   job_id = client.train(**template)
   print(f"TrainJob created: {job_id}")

   # Wait for completion
   client.wait_for_job_status(job_id)
   print("Training complete!")

   # View the logs
   for line in client.get_job_logs(name=job_id):
       print(line, end="")

.. tip::

   For production use on Kubernetes, simply omit the ``backend_config`` argument::

      client = TrainerClient()

   The same code works on both local and Kubernetes backends.

----

Step 3: Optimize Hyperparameters (Kubernetes)
----------------------------------------------

.. important::

   ``OptimizerClient`` requires a Kubernetes cluster with
   `Kubeflow Katib <https://www.kubeflow.org/docs/components/katib/>`_
   deployed. This step cannot run locally.

Now use ``OptimizerClient`` to find the best hyperparameters. The same
``TrainJobTemplate`` is reused — the optimizer overrides ``func_args`` values
according to the search space.

.. code-block:: python

   from kubeflow.optimizer import OptimizerClient, Search, TrialConfig

   opt_client = OptimizerClient()

   # Run an optimization experiment
   optimization_id = opt_client.optimize(
       trial_template=template,
       trial_config=TrialConfig(num_trials=10, parallel_trials=2),
       search_space={
           "learning_rate": Search.loguniform(0.001, 0.1),
           "num_epochs": Search.choice([5, 10, 15]),
       },
   )
   print(f"OptimizationJob created: {optimization_id}")

   # Wait for the OptimizationJob to complete
   opt_client.wait_for_job_status(optimization_id)
   print("Optimization complete!")

The optimizer runs multiple training trials in parallel, each with different
hyperparameter combinations. It finds the best combination by comparing the
reported ``loss`` metric.

.. note::

   The ``print(f"loss={final_loss:.4f}")`` statement in the training function is
   how the optimizer collects metrics. Kubeflow Katib parses stdout to extract
   metric values.

After optimization completes, retrieve the best results:

.. code-block:: python

   # Get the best hyperparameters and metrics
   best = opt_client.get_best_results(optimization_id)
   if best:
       print(f"Best hyperparameters: {best.parameters}")
       for metric in best.metrics:
           print(f"  {metric.name}: {metric.latest}")

----

Step 4: Register the Best Model (Kubernetes)
----------------------------------------------

.. important::

   ``ModelRegistryClient`` requires a
   `Model Registry server <https://www.kubeflow.org/docs/components/model-registry/installation/>`_.

After training and optimization, register the best model in Model Registry for
versioning, sharing, and deployment.

.. code-block:: python

   from kubeflow.hub import ModelRegistryClient

   mr_client = ModelRegistryClient(
       "https://model-registry.kubeflow.svc.cluster.local",
       author="Your Name",
   )

   # Retrieve the best trial's hyperparameters
   best = opt_client.get_best_results(optimization_id)

   # Register the best model with its hyperparameters
   model = mr_client.register_model(
       name="my-optimized-model",
       uri="s3://my-bucket/models/best-checkpoint",
       version="v1.0.0",
       model_format_name="pytorch",
       model_format_version="2.0",
       version_description=(
           f"Best model from optimization job {optimization_id}. "
           f"Hyperparameters: {best.parameters}"
       ),
   )

   print(f"Model registered: {model.name}")

You can then query the registry to retrieve or list your models:

.. code-block:: python

   # Retrieve the model
   model = mr_client.get_model("my-optimized-model")
   print(f"Model: {model.name}")

   # List all model versions
   for version in mr_client.list_model_versions("my-optimized-model"):
       print(f"  Version: {version.name} — {version.description}")

----

Putting It All Together
-----------------------

Here's the complete workflow in a single script. This script requires a
**Kubernetes cluster** with Kubeflow Trainer, Katib, and Model Registry deployed.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer, TrainJobTemplate
   from kubeflow.optimizer import OptimizerClient, Search, TrialConfig
   from kubeflow.hub import ModelRegistryClient


   # --- Training Function ---
   def train_fn(learning_rate: str, num_epochs: str):
       import os
       import torch
       import torch.distributed as dist

       dist.init_process_group(backend="gloo")
       model = torch.nn.Linear(10, 1)
       optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))

       for epoch in range(int(num_epochs)):
           x = torch.randn(32, 10)
           y = torch.randn(32, 1)
           loss = torch.nn.functional.mse_loss(model(x), y)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

       if dist.get_rank() == 0:
           print(f"loss={loss.item():.4f}")


   # --- Step 1: Train ---
   template = TrainJobTemplate(
       runtime="torch-distributed",
       trainer=CustomTrainer(
           func=train_fn,
           func_args={"learning_rate": "0.01", "num_epochs": "5"},
           num_nodes=2,
           resources_per_node={"cpu": 2},
       ),
   )

   trainer_client = TrainerClient()
   job_id = trainer_client.train(**template)
   trainer_client.wait_for_job_status(job_id)
   print(f"Training complete: {job_id}")

   # --- Step 2: Optimize ---
   opt_client = OptimizerClient()
   optimization_id = opt_client.optimize(
       trial_template=template,
       trial_config=TrialConfig(num_trials=10, parallel_trials=2),
       search_space={
           "learning_rate": Search.loguniform(0.001, 0.1),
           "num_epochs": Search.choice([5, 10, 15]),
       },
   )
   opt_client.wait_for_job_status(optimization_id)
   print(f"Optimization complete: {optimization_id}")

   # --- Step 3: Register Best Model ---
   best = opt_client.get_best_results(optimization_id)
   mr_client = ModelRegistryClient(
       "https://model-registry.kubeflow.svc.cluster.local",
       author="Your Name",
   )
   model = mr_client.register_model(
       name="my-optimized-model",
       uri="s3://my-bucket/models/best-checkpoint",
       version="v1.0.0",
       model_format_name="pytorch",
       model_format_version="2.0",
       version_description=(
           f"Best model from optimization {optimization_id}. "
           f"Hyperparameters: {best.parameters}"
       ),
   )
   print(f"Model registered: {model.name}")

.. tip::

   For local-only training (no Kubernetes), use Steps 1–2 with
   ``ContainerBackendConfig()``::

      client = TrainerClient(backend_config=ContainerBackendConfig())

----

Next Steps
----------

- **Scale up**: Remove ``ContainerBackendConfig()`` to run on Kubernetes
- **Custom runtimes**: See :doc:`train/runtimes` for GPU and framework-specific runtimes
- **Advanced optimization**: See :doc:`optimize/search-space` for more search strategies
- **Model serving**: Deploy registered models with KServe or Seldon
- **More examples**: Browse :doc:`examples` for framework-specific notebooks
