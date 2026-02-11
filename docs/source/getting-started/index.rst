Getting Started
===============

Get up and running with Kubeflow SDK in minutes.

What is Kubeflow SDK?
---------------------

Kubeflow SDK is a Python library that makes it easy to:

- **Train ML models** on your laptop or Kubernetes cluster
- **Scale training** from 1 GPU to hundreds without changing your code
- **Tune hyperparameters** to find the best model configuration

Think of it as "PyTorch Lightning for Kubernetes" - you write normal Python training code,
and Kubeflow handles the infrastructure.

Who is this for?
----------------

- **Data Scientists** who want to scale their training without learning Kubernetes
- **ML Engineers** who need reliable distributed training
- **Teams** who want reproducible ML workflows

Quick Example
-------------

Here's how simple it is to train a model:

.. code-block:: python

   from kubeflow.trainer import TrainerClient
   from kubeflow.trainer.types import CustomTrainer

   def train():
       """Your normal training code - nothing special needed."""
       import torch

       model = torch.nn.Linear(10, 1)
       optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

       for epoch in range(100):
           loss = model(torch.randn(32, 10)).sum()
           loss.backward()
           optimizer.step()
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

   # Submit to Kubernetes - that's it!
   client = TrainerClient()
   job_name = client.train(trainer=CustomTrainer(func=train))

   # Wait and get logs
   client.wait_for_job_status(job_name)

Next Steps
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install the SDK and verify it works.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Train your first model step-by-step.
