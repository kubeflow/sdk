1. Introduction:
- End‑to‑End Workflow with Kubeflow SDK
- In this tutorial you will build a complete machine learning workflow using the Kubeflow SDK. 
You will define a training job, execute it on Kubernetes (or locally for development), 
perform hyperparameter optimization, and optionally register the best model.

- Workflow Overview
  - Training Job
    ↓
  - Hyperparameter Optimization
    ↓
  - Model Registration(optional)

2. Prerequisites:
Before starting this tutorial, make sure you have:
       - Python 3.9+
       - Kubeflow SDK installed
       - Access to a Kubernetes cluster with Trainer and Optimizer components
Install the SDK:
       - pip install kubeflow

3. Step 1: Setup and imports:
  from kubeflow.trainer import TrainerClient
  from kubeflow.optimizer import OptimizerClient
  from kubeflow.registry import ModelRegistryClient

4. Step 2: Define the training function:
- A simple, fast toy example (e.g. linear model on random data).
- Pure Python: no Kubeflow specifics in the function itself.
    def train_model(config):
    import torch
    import torch.nn as nn

    model = nn.Linear(10, 1)
    # simple training loop
    return model

5. Step 3: Create a TrainJob template and run a baseline training job:
- Create TrainJobTemplate with a CustomTrainer.
- Submit via TrainerClient().train(**template).
- Show how to:
  - Print the returned job_id.
  - wait_for_job_status(job_id) and optionally stream logs.
  trainer = TrainerClient()
  job_id = trainer.train(template)
  trainer.wait_for_job_status(job_id)
  print(trainer.get_job_logs(job_id))

6. Step 4: Run hyperparameter optimization:
- Hyperparameter optimization runs multiple training trials with different parameter values and selects the best performing configuration.
- Use the same TrainJobTemplate.
- Define TrialConfig + search_space + Objective.
- Call OptimizerClient().optimize(...).
- Wait for completion and fetch get_best_results.
 optimizer = OptimizerClient()
  result = optimizer.optimize(search_space, trial_config)
  print(optimizer.get_best_results(result))

7. Step 5: Register the best model (optional but powerful):
- Show conceptually:
  - You would produce a model artifact at some URI (e.g. s3://... or gs://...).
  - Use ModelRegistryClient to call register_model(...).
    registry = ModelRegistryClient()
    registry.register_model(
    name="example-model",
    uri="s3://my-bucket/model"
  ) 

8. wrap up:
A complete runnable example is provided in:
  - examples/end_to_end_workflow.py
The script demonstrates the full workflow:
  - Baseline training
  - Hyperparameter optimization
  - Optional model registration.