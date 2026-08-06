End-to-End Workflow Tutorial
============================

This tutorial guides you through an end-to-end Machine Learning workflow using the Kubeflow SDK. You will see how three key SDK clients integrate seamlessly to:

1. **Optimize hyperparameters** using :class:`~kubeflow.optimizer.OptimizerClient` (Katib).
2. **Train a final model** using :class:`~kubeflow.trainer.TrainerClient` (Trainer) with the best parameters found.
3. **Register the model** using :class:`~kubeflow.hub.ModelRegistryClient` (Model Registry).

All code in this guide is also available as a Jupyter Notebook in the repository under
`examples/end-to-end-tutorial.ipynb <https://github.com/kubeflow/sdk/blob/main/examples/end-to-end-tutorial.ipynb>`_.

Prerequisites
-------------

Before you begin, make sure you have:

1. The Kubeflow SDK installed with the ``[hub]`` extra:

   .. code-block:: bash

      pip install "kubeflow[hub]"

2. Access to a Kubernetes cluster with Trainer, Katib, and Model Registry installed.
   
   .. note::
   
      To test the notebook or client interfaces without deploying the Model Registry service first, the client setup in this tutorial includes a connection fallback to mock interactions.

Step 1: Hyperparameter Tuning
-----------------------------

First, we define a standard Python training function that receives hyperparameters (learning rate `lr` and `batch_size`) as arguments. Inside the function, we use ``update_trainjob_status`` to report intermediate metrics (like loss and accuracy) to the Trainer backend. We also output logs matching the default Katib metrics parser format:

.. code-block:: python

   def trial_train_fn(lr: float, batch_size: int):
       import time
       from kubeflow.trainer import update_trainjob_status

       print(f"Starting trial training with learning_rate={lr}, batch_size={batch_size}")
       
       # Simulate epoch training loop
       for epoch in range(1, 4):
           loss = 1.0 / (epoch * lr * batch_size)
           accuracy = 0.5 + (0.45 * epoch / 3)
           
           # Print statements for Katib metrics collector
           print(f"epoch={epoch}")
           print(f"loss={loss:.4f}")
           print(f"accuracy={accuracy:.4f}")
           
           # Report progress & metrics to Trainer
           update_trainjob_status(
               progress_percent=int(epoch / 3 * 100),
               metrics={"loss": loss, "accuracy": accuracy}
           )
           time.sleep(1)
           
       print("Trial training completed successfully!")

Next, we configure the search space, trial constraints, objective goals, and submit the optimization job using the :class:`~kubeflow.optimizer.OptimizerClient`:

.. code-block:: python

   from kubeflow.trainer import TrainJobTemplate, CustomTrainer
   from kubeflow.optimizer import OptimizerClient, Search, TrialConfig, Objective, Direction

   optimizer_client = OptimizerClient()

   # Define the template for trials
   trial_template = TrainJobTemplate(
       trainer=CustomTrainer(
           func=trial_train_fn,
           func_args={"lr": 0.01, "batch_size": 32}
       )
   )

   # Define search spaces
   search_space = {
       "lr": Search.uniform(min=0.001, max=0.05),
       "batch_size": Search.choice([16, 32]),
   }

   objectives = [Objective(metric="loss", direction=Direction.MINIMIZE)]
   trial_config = TrialConfig(num_trials=2, parallel_trials=1)

   # Run optimization
   opt_job_name = optimizer_client.optimize(
       trial_template=trial_template,
       search_space=search_space,
       objectives=objectives,
       trial_config=trial_config,
   )

   # Wait for completion and fetch the best results
   optimizer_client.wait_for_job_status(opt_job_name)
   best_results = optimizer_client.get_best_results(opt_job_name)

   print(f"Optimal hyperparameters: {best_results.parameters}")

Step 2: Train Final Model
-------------------------

Once we retrieve the optimal hyperparameters from the best trial, we use :class:`~kubeflow.trainer.TrainerClient` to train our final production model:

.. code-block:: python

   from kubeflow.trainer import TrainerClient

   trainer_client = TrainerClient()

   # Parse optimized values
   best_lr = float(best_results.parameters["lr"]) if best_results else 0.01
   best_batch_size = int(best_results.parameters["batch_size"]) if best_results else 32

   def final_train_fn(lr: float, batch_size: int):
       import os
       import time
       from kubeflow.trainer import update_trainjob_status

       # Simulate final training
       for epoch in range(1, 4):
           loss = 0.8 / (epoch * lr * batch_size)
           accuracy = 0.6 + (0.35 * epoch / 3)
           print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
           
           update_trainjob_status(
               progress_percent=int(epoch / 3 * 100),
               metrics={"loss": loss, "accuracy": accuracy}
           )
           time.sleep(1)

       # Save the final model artifact to a shared/remote path
       os.makedirs("/tmp/model", exist_ok=True)
       with open("/tmp/model/model.txt", "w") as f:
           f.write(f"Model trained with lr={lr}, batch_size={batch_size}\n")
       print("Final model saved successfully!")

   # Submit and wait for final training
   final_job_name = trainer_client.train(
       trainer=CustomTrainer(
           func=final_train_fn,
           func_args={"lr": best_lr, "batch_size": best_batch_size}
       )
   )
   trainer_client.wait_for_job_status(final_job_name)
   print("Final training completed!")

Step 3: Register Best Model
---------------------------

Finally, we connect to the Kubeflow Model Registry using the :class:`~kubeflow.hub.ModelRegistryClient` and register our trained model version and its artifact URI:

.. code-block:: python

   import os
   from kubeflow.hub import ModelRegistryClient

   mr_host = os.environ.get("MODEL_REGISTRY_URL", "http://model-registry-service.kubeflow.svc.cluster.local:8080")

   # Initialize client and register the version
   mr_client = ModelRegistryClient(base_url=mr_host)

   model_name = "mnist-classifier"
   model_version = "v1.0.0"
   model_uri = "s3://my-bucket/models/mnist-classifier"

   registered_model = mr_client.register_model(
       name=model_name,
       uri=model_uri,
       version=model_version,
       model_format_name="pytorch",
       model_format_version="2.0",
       version_description="MNIST PyTorch classifier trained with optimized learning rate"
   )

   print(f"Model {model_name} (version {model_version}) successfully registered!")
