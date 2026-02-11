Defining Search Spaces
======================

A search space defines what hyperparameter values to explore.

Overview
--------

Use the ``Search`` class to define ranges for each hyperparameter:

.. code-block:: python

   from kubeflow.optimizer.types import Search

   search_space = {
       "learning_rate": Search.loguniform(1e-5, 1e-1),
       "batch_size": Search.choice([16, 32, 64]),
       "dropout": Search.uniform(0.0, 0.5),
   }

Search Types
------------

Uniform (Continuous)
^^^^^^^^^^^^^^^^^^^^

Sample uniformly between min and max:

.. code-block:: python

   # Any value between 0.1 and 0.9
   Search.uniform(0.1, 0.9)

Best for: Dropout rate, regularization strength, momentum.

Log-Uniform (Continuous)
^^^^^^^^^^^^^^^^^^^^^^^^

Sample uniformly in log space - good for values that span orders of magnitude:

.. code-block:: python

   # Values like 0.00001, 0.0001, 0.001, 0.01, 0.1
   Search.loguniform(1e-5, 1e-1)

Best for: Learning rate, weight decay.

Choice (Discrete)
^^^^^^^^^^^^^^^^^

Choose from a fixed set of values:

.. code-block:: python

   # One of these exact values
   Search.choice([16, 32, 64, 128])

Best for: Batch size, number of layers, activation functions.

Integer Range
^^^^^^^^^^^^^

Sample integers in a range:

.. code-block:: python

   # Any integer from 1 to 10
   Search.randint(1, 10)

Best for: Number of epochs, hidden layer sizes.

Complete Example
----------------

Here's a realistic search space for training a neural network:

.. code-block:: python

   search_space = {
       # Learning rate: explore across orders of magnitude
       "learning_rate": Search.loguniform(1e-5, 1e-2),

       # Batch size: try common values
       "batch_size": Search.choice([32, 64, 128, 256]),

       # Architecture
       "hidden_size": Search.choice([128, 256, 512, 1024]),
       "num_layers": Search.randint(2, 6),

       # Regularization
       "dropout": Search.uniform(0.0, 0.5),
       "weight_decay": Search.loguniform(1e-6, 1e-2),

       # Optimizer
       "optimizer": Search.choice(["adam", "sgd", "adamw"]),
   }

Using Hyperparameters in Training
---------------------------------

Your training function receives hyperparameters as arguments:

.. code-block:: python

   def train(learning_rate, batch_size, dropout):
       """Training function that uses the hyperparameters."""
       import torch
       import torch.nn as nn

       model = nn.Sequential(
           nn.Linear(784, 256),
           nn.ReLU(),
           nn.Dropout(dropout),  # Use the hyperparameter
           nn.Linear(256, 10),
       )

       optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

       # ... training loop with batch_size ...

       # Report the metric back
       print(f"accuracy={accuracy:.4f}")

Tips for Search Space Design
----------------------------

**Start broad, then narrow:**

.. code-block:: python

   # First: wide range
   "learning_rate": Search.loguniform(1e-6, 1.0)

   # Later: narrow based on results
   "learning_rate": Search.loguniform(1e-4, 1e-2)

**Use log scale for learning rate:**

Learning rates often work best when explored logarithmically. The difference
between 0.001 and 0.01 is more significant than between 0.5 and 0.51.

**Don't search too many parameters at once:**

Each additional parameter exponentially increases the search space.
Start with 3-5 most impactful parameters.

**Consider dependencies:**

Some hyperparameters interact. For example, larger batch sizes often
need higher learning rates.
