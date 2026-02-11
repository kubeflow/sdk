Search Algorithms
=================

Choose how Kubeflow searches for the best hyperparameters.

Overview
--------

Different algorithms have different trade-offs:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Algorithm
     - Best For
     - Trade-off
   * - **Random Search**
     - Quick exploration, simple problems
     - May miss optimal regions
   * - **Bayesian Optimization**
     - Expensive training, small budgets
     - More overhead per trial
   * - **Grid Search**
     - Exhaustive search, few parameters
     - Doesn't scale well

Random Search (Default)
-----------------------

Randomly samples hyperparameters from the search space:

.. code-block:: python

   from kubeflow.optimizer.types import RandomSearch

   client.optimize(
       trial_template=template,
       search_space=search_space,
       algorithm=RandomSearch(),
   )

**When to use:**

- You have a large search space
- Training is relatively fast
- You want a simple baseline

**Pros:** Simple, parallelizes well, surprisingly effective.

**Cons:** No learning between trials, may miss optimal regions.

Bayesian Optimization
---------------------

Uses a probabilistic model to guide the search:

.. code-block:: python

   from kubeflow.optimizer.types import BayesianOptimization

   client.optimize(
       trial_template=template,
       search_space=search_space,
       algorithm=BayesianOptimization(),
   )

**When to use:**

- Training is expensive (hours per run)
- You have a limited compute budget
- You want to minimize the number of trials

**Pros:** Learns from previous trials, converges faster.

**Cons:** Doesn't parallelize as well, more complex.

Grid Search
-----------

Exhaustively tries all combinations:

.. code-block:: python

   from kubeflow.optimizer.types import GridSearch

   # Grid search works best with discrete choices
   search_space = {
       "learning_rate": Search.choice([0.001, 0.01, 0.1]),
       "batch_size": Search.choice([32, 64]),
   }
   # This will try all 6 combinations

   client.optimize(
       trial_template=template,
       search_space=search_space,
       algorithm=GridSearch(),
   )

**When to use:**

- You have very few hyperparameters
- You need to try all combinations
- Search space is already discrete

**Pros:** Exhaustive, guaranteed to find best in the grid.

**Cons:** Exponential scaling, impractical for many parameters.

Controlling the Search
----------------------

**Limit number of trials:**

.. code-block:: python

   from kubeflow.optimizer.types import TrialConfig

   client.optimize(
       trial_template=template,
       search_space=search_space,
       trial_config=TrialConfig(max_trials=20),  # Stop after 20 trials
   )

**Run trials in parallel:**

.. code-block:: python

   client.optimize(
       trial_template=template,
       search_space=search_space,
       trial_config=TrialConfig(
           max_trials=50,
           parallel_trials=5,  # Run 5 at a time
       ),
   )

Algorithm Recommendations
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Scenario
     - Recommended Algorithm
   * - First exploration of a new model
     - Random Search with 20-50 trials
   * - Training takes hours per run
     - Bayesian Optimization
   * - Only 2-3 hyperparameters to tune
     - Grid Search
   * - Large compute budget available
     - Random Search with many parallel trials
   * - Need to find good config quickly
     - Bayesian Optimization with early stopping
