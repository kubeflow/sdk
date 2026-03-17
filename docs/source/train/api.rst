API Reference
=============

TrainerClient
-------------

.. autoclass:: kubeflow.trainer.TrainerClient
   :members:
   :show-inheritance:

Using ``options`` with TrainerClient
------------------------------------

The ``options`` parameter in ``TrainerClient.train()`` allows passing backend-specific configuration for job execution (for example, job name, labels, or Kubernetes runtime / pod-spec patches for compatible backends).

Example (Container backend, configuring env vars via an option instead of ``CustomTrainer.env``):

.. code-block:: python

    from kubeflow.trainer import TrainerClient, ContainerBackendConfig, CustomTrainer
    from kubeflow.trainer import options as trainer_options
    import os
    client = TrainerClient(
        backend_config=ContainerBackendConfig()
    )

    def train_simple():
        print("ENV VAR:", os.getenv("MY_VAR"))

    job_id = client.train(
        trainer=CustomTrainer(
            func=train_simple,
            image="python:3.10"
        ),
        options=[
            trainer_options.env({"MY_VAR": "value"})
        ]
    )

Notes:

- ``options`` is a list of backend-specific option callables passed at job submission time (for example, to tweak job naming, labels, or apply backend-specific runtime / pod-spec patches)
- Environment variables are primarily configured on the trainer itself via ``CustomTrainer.env``; for some backends you can also use an ``env`` option:

  - ``trainer_options.env({...})`` (with ``ContainerBackendConfig``): injects the given environment variables into the training container

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
