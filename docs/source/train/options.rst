Options Reference
=================

.. autoclass:: kubeflow.trainer.options.Name
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.Labels
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.Annotations
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.TrainerCommand
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.TrainerArgs
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.RuntimePatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.TrainingRuntimeSpecPatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.JobSetTemplatePatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.JobSetSpecPatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.ReplicatedJobPatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.JobTemplatePatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.JobSpecPatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.PodTemplatePatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.PodSpecPatch
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.options.ContainerPatch
   :members:
   :show-inheritance:

Using options with TrainerClient
===============================

The ``options`` parameter in ``TrainerClient`` allows users to customize runtime behavior
and backend-specific configurations for training jobs.

It provides flexibility to control how training jobs are executed depending on the
selected backend (e.g., Kubernetes, local, container).

Example
-------

.. code-block:: python

    from kubeflow.trainer import TrainerClient, CustomTrainer

    def train_fn():
        print("Training...")

    client = TrainerClient()

    job_id = client.train(
        trainer=CustomTrainer(func=train_fn),
        options={
            "epochs": 10,
            "batch_size": 32
        }
    )

    client.wait_for_job_status(job_id)

The ``options`` dictionary can include different parameters depending on the backend
and runtime configuration.