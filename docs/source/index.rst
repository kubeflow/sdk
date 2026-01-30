Kubeflow SDK
============

**A Pythonic API to Run AI Workloads at Scale**

Run any AI workload at any scale — without the need to learn Kubernetes. The Kubeflow SDK provides
simple and consistent Python APIs across the Kubeflow ecosystem, enabling you to focus on building
AI applications rather than managing infrastructure.

----

Quick Start
-----------
*Get up and running in minutes.*

.. code-block:: bash

   pip install kubeflow

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer

   def train_fn():
       import torch
       model = torch.nn.Linear(10, 1)
       print("Training complete!")

   client = TrainerClient()
   client.train(trainer=CustomTrainer(func=train_fn, num_nodes=3))

:doc:`Full installation guide → <getting-started/installation>`

----

Why Kubeflow SDK?
-----------------

.. grid:: 2
   :gutter: 4

   .. grid-item::

      🎯 **Unified Experience**

      Single SDK to interact with multiple Kubeflow projects through consistent Python APIs.

   .. grid-item::

      🐍 **Simplified AI Workloads**

      Abstract away Kubernetes complexity using familiar Python APIs.

   .. grid-item::

      🚀 **Built for Scale**

      From local laptop to production cluster with thousands of GPUs using the same APIs.

   .. grid-item::

      💻 **Local Development**

      First-class support for local development without a Kubernetes cluster.

----

Supported Projects
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Project
     - Status
     - Description
   * - :doc:`Trainer <train/index>`
     - ✅ Available
     - Train and fine-tune AI models with various frameworks
   * - :doc:`Katib <optimize/index>`
     - ✅ Available
     - Hyperparameter optimization
   * - :doc:`Model Registry <hub/index>`
     - ✅ Available
     - Manage model artifacts and versions
   * - Pipelines
     - 🚧 Planned
     - Build, run, and track AI workflows
   * - Spark Operator
     - 🚧 Planned
     - Data processing and feature engineering
   * - Feast
     - 🚧 Planned
     - Feature store for machine learning

----

Getting Involved
----------------
*Join the community and help shape the future of ML on Kubernetes.*

.. grid:: 3
   :gutter: 4

   .. grid-item::

      💬 **Community**

      - `Slack <https://kubeflow.slack.com>`_
      - `Discussions <https://github.com/kubeflow/sdk/discussions>`_
      - `Meetings <https://www.kubeflow.org/docs/about/community/>`_

   .. grid-item::

      🤝 **Contribute**

      - `GitHub <https://github.com/kubeflow/sdk>`_
      - `Issues <https://github.com/kubeflow/sdk/issues>`_
      - `Contributing <https://github.com/kubeflow/sdk/blob/main/CONTRIBUTING.md>`_

   .. grid-item::

      📚 **Resources**

      - `Website <https://www.kubeflow.org>`_
      - `Blog Post <https://blog.kubeflow.org/sdk/intro/>`_
      - `DeepWiki <https://deepwiki.com/kubeflow/sdk>`_

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Trainer

   train/index
   train/custom-training
   train/distributed
   train/runtimes
   train/api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Optimizer

   optimize/index
   optimize/search-space
   optimize/algorithms
   optimize/api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Model Registry

   hub/index
   hub/api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing

   contributing
