Hyperparameter Tuning
=====================

Automatically find the best configuration for your model.

Overview
--------

Hyperparameter tuning helps you find optimal values for:

- **Learning rate** - Too high and training diverges, too low and it's slow
- **Batch size** - Affects memory usage and convergence
- **Model architecture** - Number of layers, hidden dimensions, etc.
- **Regularization** - Dropout, weight decay, etc.

Instead of manually trying different values, let Kubeflow search for you.

Quick Example
-------------

.. code-block:: python

   from kubeflow.optimizer import OptimizerClient
   from kubeflow.optimizer.types import Search, Objective
   from kubeflow.trainer.types import TrainJobTemplate, CustomTrainer

   # Define what to optimize
   search_space = {
       "learning_rate": Search.loguniform(1e-5, 1e-1),
       "batch_size": Search.choice([16, 32, 64, 128]),
       "dropout": Search.uniform(0.1, 0.5),
   }

   # Create optimization job
   client = OptimizerClient()
   job_name = client.optimize(
       trial_template=TrainJobTemplate(trainer=CustomTrainer(func=train)),
       search_space=search_space,
       objectives=[Objective(name="accuracy", type="maximize")],
   )

   # Get best results
   best = client.get_best_results(job_name)
   print(f"Best hyperparameters: {best.hyperparameters}")
   print(f"Best accuracy: {best.metrics['accuracy']}")

How It Works
------------

1. **Define search space** - Specify ranges for each hyperparameter
2. **Create trial template** - Define how to train with given hyperparameters
3. **Run optimization** - Kubeflow tries different combinations
4. **Get best results** - Retrieve the winning configuration

Kubeflow uses Katib under the hood, which supports various search algorithms
like random search, Bayesian optimization, and more.

Guides
------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Defining Search Spaces
      :link: search-space
      :link-type: doc

      Learn how to specify parameter ranges and distributions.

   .. grid-item-card:: Search Algorithms
      :link: algorithms
      :link-type: doc

      Choose the right algorithm for your problem.

Key Concepts
------------

**Trial**: A single training run with a specific set of hyperparameters.

**Objective**: The metric you want to optimize (e.g., accuracy, loss).

**Search Space**: The range of values to try for each hyperparameter.

**Algorithm**: The strategy for choosing which values to try next.

Common Patterns
---------------

**Maximize accuracy:**

.. code-block:: python

   objectives=[Objective(name="accuracy", type="maximize")]

**Minimize loss:**

.. code-block:: python

   objectives=[Objective(name="loss", type="minimize")]

**Get logs from the best trial:**

.. code-block:: python

   for line in client.get_job_logs(job_name):
       print(line)

**Monitor optimization progress:**

.. code-block:: python

   job = client.get_job(job_name)
   print(f"Status: {job.status}")
   print(f"Trials completed: {job.completed_trials}")
