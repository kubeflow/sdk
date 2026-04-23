Quickstart
==========

This guide walks you through training your first model with Kubeflow SDK.

Prerequisites
-------------

Before you begin, make sure you have:

1. Python 3.10 or higher installed
2. The Kubeflow SDK installed (see :doc:`installation`)
3. Access to a Kubernetes cluster with Kubeflow Trainer installed

   .. note::

      Don't have a cluster? You can use the **local backend** to test on your laptop first.
      See :ref:`local-development` below.

Step 1: Write Your Training Function
------------------------------------

Write a normal Python function that trains your model. Nothing special is needed -
just regular PyTorch, TensorFlow, or any framework you prefer:

.. code-block:: python

   def train_mnist():
       """Train a simple model on MNIST."""
       import torch
       import torch.nn as nn
       from torchvision import datasets, transforms

       # Load data
       transform = transforms.ToTensor()
       train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
       train_loader = torch.utils.data.DataLoader(train_data, batch_size=64)

       # Simple model
       model = nn.Sequential(
           nn.Flatten(),
           nn.Linear(784, 128),
           nn.ReLU(),
           nn.Linear(128, 10)
       )

       optimizer = torch.optim.Adam(model.parameters())
       criterion = nn.CrossEntropyLoss()

       # Train
       for epoch in range(5):
           for batch_idx, (data, target) in enumerate(train_loader):
               optimizer.zero_grad()
               output = model(data)
               loss = criterion(output, target)
               loss.backward()
               optimizer.step()

               if batch_idx % 100 == 0:
                   print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

Step 2: Submit the Training Job
-------------------------------

Use the :class:`~kubeflow.trainer.TrainerClient` to submit your function to Kubernetes:

.. code-block:: python

   from kubeflow.trainer import TrainerClient
   from kubeflow.trainer.types import CustomTrainer

   # Create client (connects to your Kubernetes cluster)
   client = TrainerClient()

   # Submit the training job
   job_name = client.train(
       trainer=CustomTrainer(func=train_mnist)
   )

   print(f"Training job started: {job_name}")

That's it! Your training function is now running on Kubernetes.

Step 3: Monitor Progress
------------------------

Watch the logs in real-time:

.. code-block:: python

   # Stream logs as they happen
   for line in client.get_job_logs(job_name, follow=True):
       print(line)

Or check the job status:

.. code-block:: python

   job = client.get_job(job_name)
   print(f"Status: {job.status}")

Step 4: Wait for Completion
---------------------------

Wait for the job to finish:

.. code-block:: python

   # Blocks until complete (or timeout)
   client.wait_for_job_status(job_name)
   print("Training complete!")

.. _local-development:

Local Development (No Kubernetes)
---------------------------------

Want to test without a Kubernetes cluster? Use the local backend:

.. code-block:: python

   from kubeflow.trainer import TrainerClient
   from kubeflow.trainer.backends.localprocess import LocalProcessBackendConfig

   # Run locally instead of on Kubernetes
   client = TrainerClient(backend_config=LocalProcessBackendConfig())

   job_name = client.train(trainer=CustomTrainer(func=train_mnist))

This runs your training function as a local process - great for development and debugging.

What's Next?
------------

- :doc:`../train/index` - Learn about distributed training, GPUs, and more
- :doc:`../tune/index` - Automatically tune hyperparameters
