End-to-End Workflow Tutorial
============================

This tutorial shows how the three core SDK clients can work together in one practical flow:

1. Train a model with :class:`~kubeflow.trainer.TrainerClient`
2. Tune hyperparameters with :class:`~kubeflow.optimizer.OptimizerClient`
3. Register the best model with :class:`~kubeflow.hub.ModelRegistryClient`

The goal is to help you understand the *handoff points* between services, not just each client in isolation.

Workflow Overview
-----------------

.. code-block:: text

   Training code (TrainerClient)
        ↓
   Hyperparameter search (OptimizerClient)
        ↓
   Best model artifact URI
        ↓
   Model registration + versioning (ModelRegistryClient)

Step 1: Define Reusable Training Logic
--------------------------------------

Start with a training function that can accept hyperparameters.

.. code-block:: python

   def train_model(learning_rate: float = 1e-3, batch_size: int = 32):
       import torch
       from torch import nn

       model = nn.Linear(10, 2)
       optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

       # Simplified example loop
       for _ in range(100):
           x = torch.randn(batch_size, 10)
           y = torch.randint(0, 2, (batch_size,))
           loss = nn.CrossEntropyLoss()(model(x), y)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()

       # Persist the trained model to shared storage.
       # In production, use your artifact storage path (S3/GCS/PVC/etc.).
       model_uri = "s3://ml-artifacts/my-model/latest"
       return {"accuracy": 0.91, "model_uri": model_uri}

Step 2: Run Baseline Training
-----------------------------

Use ``TrainerClient`` to validate that your training entrypoint runs correctly.

.. code-block:: python

   from kubeflow.trainer import TrainerClient
   from kubeflow.trainer.types import CustomTrainer

   trainer_client = TrainerClient()

   baseline_job = trainer_client.train(
       trainer=CustomTrainer(func=train_model),
   )

   trainer_client.wait_for_job_status(baseline_job)

Step 3: Optimize Hyperparameters
--------------------------------

Use ``OptimizerClient`` (Katib) to explore better parameter combinations.

.. code-block:: python

   from kubeflow.optimizer import OptimizerClient
   from kubeflow.optimizer.types import Search, Objective
   from kubeflow.trainer.types import TrainJobTemplate, CustomTrainer

   optimizer_client = OptimizerClient()

   search_space = {
       "learning_rate": Search.loguniform(1e-5, 1e-2),
       "batch_size": Search.choice([16, 32, 64]),
   }

   optimization_job = optimizer_client.optimize(
       trial_template=TrainJobTemplate(trainer=CustomTrainer(func=train_model)),
       search_space=search_space,
       objectives=[Objective(name="accuracy", type="maximize")],
   )

   best = optimizer_client.get_best_results(optimization_job)

Step 4: Register the Best Model
-------------------------------

After selecting the best trial, register the winning artifact for traceability and deployment.

.. code-block:: python

   from kubeflow.hub import ModelRegistryClient

   registry = ModelRegistryClient(
       base_url="https://registry.example.com",
       author="your-name",
   )

   # Replace with the artifact URI produced by your best trial.
   best_model_uri = "s3://ml-artifacts/my-model/best"

   registry.register_model(
       name="demo-classifier",
       version="1.0.0",
       uri=best_model_uri,
       model_format_name="pytorch",
       description="Best model selected from Katib hyperparameter tuning",
   )

Putting It Together
-------------------

A typical production loop looks like this:

- Re-run optimization when data distribution changes
- Register each promoted model version with metadata
- Deploy using the registered URI as the source of truth

This pattern gives you:

- **Reproducibility** (tracked training and tuning runs)
- **Comparability** (best-trial selection with explicit objective)
- **Governance** (versioned registry entries for downstream deployment)
