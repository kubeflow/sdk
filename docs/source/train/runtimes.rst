Training Runtimes
=================

Runtimes are pre-configured environments for running your training jobs.

What is a Runtime?
------------------

A runtime provides:

- **Container image** with ML frameworks pre-installed
- **Distributed training setup** (environment variables, process management)
- **Framework-specific optimizations**

Think of runtimes as "batteries included" environments for specific frameworks.

Listing Available Runtimes
--------------------------

See what runtimes are available on your cluster:

.. code-block:: python

   from kubeflow.trainer import TrainerClient

   client = TrainerClient()
   runtimes = client.list_runtimes()

   for runtime in runtimes:
       print(f"Name: {runtime.name}")

Common Runtimes
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Runtime
     - Description
     - Use Case
   * - ``torch-distributed``
     - PyTorch with distributed training support
     - Most PyTorch training jobs
   * - ``tensorflow-distributed``
     - TensorFlow with MultiWorkerMirroredStrategy
     - TensorFlow training jobs
   * - ``mpi``
     - MPI-based distributed training
     - Custom distributed frameworks

Using a Runtime
---------------

Specify a runtime when creating a training job:

.. code-block:: python

   client.train(
       runtime="torch-distributed",
       trainer=CustomTrainer(func=my_train_function)
   )

Checking Runtime Details
------------------------

Inspect what's installed in a runtime:

.. code-block:: python

   runtime = client.get_runtime("torch-distributed")

   # Print installed packages
   client.get_runtime_packages(runtime)

This is useful for debugging import errors or version conflicts.

Default Runtime
---------------

If you don't specify a runtime, the SDK uses ``torch-distributed`` by default:

.. code-block:: python

   # These are equivalent
   client.train(trainer=CustomTrainer(func=train))
   client.train(runtime="torch-distributed", trainer=CustomTrainer(func=train))

Using Custom Containers
-----------------------

If the built-in runtimes don't have what you need, use a custom container instead:

.. code-block:: python

   from kubeflow.trainer.types import CustomTrainerContainer

   client.train(
       trainer=CustomTrainerContainer(
           image="my-registry/my-training-image:latest",
           command=["python", "train.py"],
       )
   )

This gives you full control over the environment.
