Custom Training Functions
=========================

The easiest way to train models is by writing a Python function and letting
Kubeflow SDK run it on Kubernetes.

Basic Example
-------------

.. code-block:: python

   from kubeflow.trainer import TrainerClient
   from kubeflow.trainer.types import CustomTrainer

   def my_training_function():
       """Your training code runs inside this function."""
       import torch

       # Create model
       model = torch.nn.Linear(10, 1)
       optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

       # Training loop
       for step in range(1000):
           loss = model(torch.randn(32, 10)).sum()
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()

           if step % 100 == 0:
               print(f"Step {step}, Loss: {loss.item():.4f}")

   # Submit to Kubernetes
   client = TrainerClient()
   job_name = client.train(trainer=CustomTrainer(func=my_training_function))

How It Works
------------

When you use ``CustomTrainer(func=...)``:

1. **Serialization**: Your function is serialized using cloudpickle
2. **Packaging**: The SDK creates a container with your code
3. **Submission**: A Kubernetes TrainJob is created
4. **Execution**: Your function runs in a pod on the cluster

.. note::

   Your function must be self-contained. Import libraries **inside** the function
   if they're not available in the runtime image.

Passing Parameters
------------------

Use ``func_args`` to pass arguments:

.. code-block:: python

   def train(learning_rate, epochs):
       print(f"Training with lr={learning_rate} for {epochs} epochs")
       # ... training code ...

   client.train(
       trainer=CustomTrainer(
           func=train,
           func_args={"learning_rate": 0.001, "epochs": 10}
       )
   )

Choosing a Runtime
------------------

Runtimes are pre-configured environments with frameworks like PyTorch or TensorFlow:

.. code-block:: python

   # List available runtimes
   runtimes = client.list_runtimes()
   for rt in runtimes:
       print(f"{rt.name}")

   # Use a specific runtime
   client.train(
       runtime="torch-distributed",
       trainer=CustomTrainer(func=my_training_function)
   )

See :doc:`runtimes` for more details.

Using GPUs
----------

Request GPU resources:

.. code-block:: python

   client.train(
       trainer=CustomTrainer(
           func=train,
           resources_per_node={"gpu": 2}
       ),
   )

Tips and Best Practices
-----------------------

**Keep functions self-contained:**

.. code-block:: python

   # Good - imports inside function
   def train():
       import torch
       model = torch.nn.Linear(10, 1)

   # Avoid - imports at module level may not be available
   import torch
   def train():
       model = torch.nn.Linear(10, 1)

**Report progress to the controller:**

When the ``TrainJobStatus`` feature gate is enabled, the controller injects environment
variables into your training pods and you can report real-time progress:

.. code-block:: python

   import time
   from kubeflow.trainer import update_trainjob_status

   def train():
       total_steps = 1000
       start_time = time.time()

       for step in range(total_steps):
           loss = train_step(batch)
           progress = int((step / total_steps) * 100)
           elapsed = time.time() - start_time
           eta = int((elapsed / (step + 1)) * (total_steps - step - 1)) if step > 0 else None
           update_trainjob_status(
               progress_percent=progress,
               estimated_remaining_seconds=eta,
               metrics={"loss": loss, "step": step},
           )  # SDK throttles to max 1 update per 5 seconds

This is safe to call in any environment — it returns ``False`` silently if not running
inside a Kubeflow TrainJob with the feature gate enabled.

.. note::

   For HuggingFace Transformers, use ``KubeflowCallback`` which reports progress
   automatically without any code changes.

**Print progress for monitoring:**

Your ``print()`` statements appear in the job logs:

.. code-block:: python

   def train():
       for epoch in range(10):
           # ... training ...
           print(f"Epoch {epoch} complete, loss: {loss:.4f}")

**Save checkpoints to persistent storage:**

.. code-block:: python

   def train():
       # ... training ...
       torch.save(model.state_dict(), "/mnt/output/model.pt")

When Things Go Wrong
--------------------

**Check job status:**

.. code-block:: python

   job = client.get_job(job_name)
   print(f"Status: {job.status}")

**Get error logs:**

.. code-block:: python

   for line in client.get_job_logs(job_name):
       print(line)

**Common issues:**

- **ImportError**: Library not in runtime image. Import inside function or use custom container.
- **OOM**: Out of memory. Reduce batch size or request more resources.
- **Timeout**: Training took too long. Increase timeout in ``wait_for_job_status()``.
