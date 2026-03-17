API Reference
=============

TrainerClient
-------------

.. autoclass:: kubeflow.trainer.TrainerClient
   :members:
   :show-inheritance:

Using ``options`` with TrainerClient
------------------------------------

The ``options`` parameter in ``TrainerClient.train()`` allows passing backend-specific configuration for job execution, such as environment variables and runtime settings.

Example:

.. code-block:: python

    from kubeflow.trainer import TrainerClient, ContainerBackendConfig, CustomTrainer

    client = TrainerClient(
        backend_config=ContainerBackendConfig()
    )

    def train_simple():
        print("we can put our test code here")

    job_id = client.train(
        trainer=CustomTrainer(
            func=train_simple,
            image="python:3.10"
        ),
        options={
            "env": {"EXAMPLE_VAR": "value"}
        }
    )

Notes:

- ``options`` is a dict of backend-specific settings passed at job submission time
- Supported keys when using ``ContainerBackendConfig``:

  - ``env``: A dict of environment variables injected into the training container

- ``image`` in ``CustomTrainer`` is required — it specifies the Docker image for execution

Trainers
--------

.. autoclass:: kubeflow.trainer.CustomTrainer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.CustomTrainerContainer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.BuiltinTrainer
   :members:
   :show-inheritance:

Backend Configurations
----------------------

.. autoclass:: kubeflow.trainer.KubernetesBackendConfig
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.LocalProcessBackendConfig
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.ContainerBackendConfig
   :members:
   :show-inheritance:
