Training Models
===============

Learn how to train ML models with Kubeflow SDK.

Overview
--------

The Kubeflow SDK makes it easy to run training jobs on Kubernetes. You can:

- **Use any framework** - PyTorch, TensorFlow, JAX, or custom code
- **Scale horizontally** - Distribute training across multiple nodes
- **Use GPUs** - Request GPU resources for your training jobs
- **Track progress** - Monitor logs and job status in real-time

How It Works
------------

1. You write a Python training function (or use a container)
2. The SDK packages your code and submits it to Kubernetes
3. Kubeflow Trainer runs your code on the cluster
4. You monitor progress and retrieve results

Three Ways to Train
-------------------

Choose the approach that fits your workflow:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Approach
     - Best For
     - Example
   * - **Custom Function**
     - Quick experiments, Jupyter notebooks
     - ``CustomTrainer(func=train_fn)``
   * - **Custom Container**
     - Production, reproducible builds
     - ``CustomTrainerContainer(image="my-image")``
   * - **Built-in Trainer**
     - LLM fine-tuning, standard workflows
     - ``BuiltinTrainer(...)``

Guides
------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Custom Training Functions
      :link: custom-training
      :link-type: doc

      Package your Python code and run it on Kubernetes.

   .. grid-item-card:: Distributed Training
      :link: distributed
      :link-type: doc

      Scale training across multiple GPUs and nodes.

   .. grid-item-card:: Training Runtimes
      :link: runtimes
      :link-type: doc

      Understand pre-configured environments for PyTorch, TensorFlow, etc.

Common Patterns
---------------

**Get logs while training:**

.. code-block:: python

   for line in client.get_job_logs(job_name, follow=True):
       print(line)

**Wait for completion with timeout:**

.. code-block:: python

   client.wait_for_job_status(job_name, timeout=3600)  # 1 hour max

**List all your training jobs:**

.. code-block:: python

   jobs = client.list_jobs()
   for job in jobs:
       print(f"{job.name}: {job.status}")

**Delete a job:**

.. code-block:: python

   client.delete_job(job_name)
