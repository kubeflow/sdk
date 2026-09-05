Pipelines
=========

Build, run, and monitor ML pipelines on Kubeflow Pipelines.

Overview
--------

The Kubeflow SDK provides a ``PipelinesClient`` that gives users the full
author, compile, upload, run, and monitor pipeline workflow from a single
``kubeflow`` import. Using the Pipelines SDK, you can:

- **Define pipelines** - Use the KFP DSL to author pipelines and components
- **Upload and version** - Register pipelines on the server with automatic versioning
- **Run pipelines** - Execute by name, from a function, or from a compiled YAML file
- **Monitor runs** - Wait for specific states with flexible callbacks
- **Manage experiments** - Organize runs into experiments

The client is implemented in the KFP repository and re-exported by the Kubeflow SDK,
providing a simplified, name-first API alongside the existing ``kfp.Client``.

Installation
------------

.. code-block:: bash

   pip install "kubeflow[pipelines]"

Quick Example
-------------

.. code-block:: python

   from kubeflow.pipelines import PipelinesClient, dsl

   @dsl.component(base_image="python:3.11")
   def say_hello(greeting: str) -> str:
       return f"Hello, {greeting}!"

   @dsl.pipeline
   def hello_pipeline(greeting: str = "world"):
       say_hello(greeting=greeting)

   client = PipelinesClient()

   # Upload and run
   client.upload_pipeline(hello_pipeline, name="hello-pipeline")
   run = client.run("hello-pipeline", params={"greeting": "Kubeflow"})

   # Wait for completion
   completed = client.wait_for_run_status(run, timeout=300)
   print(f"Run finished: {completed.state}")

How It Works
------------

1. **Define** - Author pipelines using the KFP DSL (``dsl``, ``compiler``, ``components``, ``kubernetes``)
2. **Upload** - Register pipelines on the server with ``upload_pipeline``
3. **Run** - Execute pipelines by name, callable, or YAML file
4. **Monitor** - Wait for specific run states with ``wait_for_run_status``

The ``PipelinesClient`` uses a name-first API: pipeline and experiment operations
work with display names, not internal IDs. ``wait_for_run_status`` accepts either
a ``Run`` object or a run ID string; ``get_run`` takes a run ID string.

Key Concepts
------------

**Pipeline**: A definition of an ML workflow composed of one or more components.

**Pipeline Version**: An immutable snapshot of a pipeline. Each ``upload_pipeline``
call creates a new version.

**Run**: A single execution of a pipeline with specific parameters.
``Run.state`` returns uppercase values from the API (e.g. ``"SUCCEEDED"``),
while ``constants.RUN_SUCCEEDED`` is lowercase (``"succeeded"``). Use
``state.lower()`` when comparing against constants.

**Experiment**: A logical grouping of runs for organization.

Common Patterns
---------------

**Upload once, run many times:**

.. code-block:: python

   client.upload_pipeline(my_pipeline, name="training")
   client.create_experiment("sweep")

   for lr in [0.01, 0.001, 0.0001]:
       client.run("training", params={"lr": lr}, experiment="sweep")

**Quick inline run from a callable (no upload):**

.. code-block:: python

   run = client.run(my_pipeline, params={"epochs": 5})

**Quick inline run from a compiled YAML file:**

.. code-block:: python

   run = client.run("pipeline.yaml", params={"epochs": 5})

**Wait for a non-terminal state:**

.. code-block:: python

   running = client.wait_for_run_status(run, status={"running"}, timeout=120)

**Access the underlying kfp.Client (escape hatch):**

.. code-block:: python

   kfp_client = client.kfp_client
   kfp_client.list_experiments()
