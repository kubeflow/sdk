Options Reference
=================

Options let to customize how a TrainJob is created and executed. Pass them as a list to the '''options''' parameter of the
:py:meth:`kubeflow.trainer.TrainerClient.train` method.
.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import Name, Labels, Annotations

   trainer_client = TrainerClient()
   job_id = trainer_client.train(
       trainer=CustomTrainer(func=train_function),
       options=[
           Name("my-train-job"),
           Labels({"team": "ml", "env": "prod"}),
           Annotations({"note": "experiment-42"}),
       ],
   )

.. note::
   Not all options work with every backend. Each option documents
   which backends it supports. An unsupported option will raise a
   `ValueError` at runtime.

----

Usage Guide
-----------

Name
----

Set a custom name for the TrainJob resource. Works with all backends.

.. code-block:: python
   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import Name

   trainer_client = TrainerClient()

   job_id = trainer_client.train(
       trainer=CustomTrainer(func=train_function),
       options=[Name("my-custom-job")],
   )

Labels
------

Add labels to the TrainJob resource metadata (``metadata.labels``). Only supported on the **Kubernetes backend**.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import Labels

   trainer_client = TrainerClient()

   job_id = trainer_client.train(
       trainer=CustomTrainer(func=train_function),
       options=[Labels({"team": "ml-platform", "version": "v2"})],
   )

Annotations
-----------

Add annotations to the TrainJob resource metadata(``metadata.annotations``). Only supported on the Kubernetes backend.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import Annotations

   trainer_client = TrainerClient()

   job_id = trainer_client.train(
       trainer=CustomTrainer(func=train_function),
       options=[Annotations({"owner": "alice", "ticket": "JIRA-42"})],
   )

TrainerCommand
--------------

Override the trainer container command (``spec.trainer.command``).
Can Only be used with ''CustomTrainerContainer'' not with ''CustomTrainer''' or ''BuiltinTrainer''.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainerContainer
   from kubeflow.trainer.options import TrainerCommand

   trainer_client = TrainerClient()

   job_id = trainer_client.train(
       trainer=CustomTrainerContainer(image="my-image:latest"),
       options=[TrainerCommand(["python", "train.py", "--epochs", "10"])],
   )

TrainerArgs
-----------

Append extra arguments to the trainer container command.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import TrainerArgs

   trainer_client = TrainerClient()

   job_id = trainer_client.train(
       trainer=CustomTrainer(func=train_function),
       options=[TrainerArgs(["--lr", "0.001", "--batch-size", "32"])],
   )

RuntimePatch
--------------

Apply structured patches to the TrainJob (``spec.runtimePatches``) Use this for advanced Kubernetes-level customisation such as adding init containers, volumes, or tolerations. Only supported on the ''Kubernetes backend''.

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import (
       RuntimePatch,
       TrainingRuntimeSpecPatch,
       JobSetTemplatePatch,
       JobSetSpecPatch,
       ReplicatedJobPatch,
       JobTemplatePatch,
       JobSpecPatch,
       PodTemplatePatch,
       PodSpecPatch,
       ContainerPatch,
   )

   trainer_client = TrainerClient()

   patch = RuntimePatch(
       training_runtime_spec=TrainingRuntimeSpecPatch(
           template=JobSetTemplatePatch(
               spec=JobSetSpecPatch(
                   replicated_jobs=[
                       ReplicatedJobPatch(
                           name="node",
                           template=JobTemplatePatch(
                               spec=JobSpecPatch(
                                   template=PodTemplatePatch(
                                       spec=PodSpecPatch(
                                           containers=[
                                               ContainerPatch(
                                                   name="trainer",
                                                   env=[{
                                                       "name": "MY_VAR",
                                                       "value": "hello",
                                                   }],
                                               )
                                           ]
                                       )
                                   )
                               )
                           ),
                       )
                   ]
               )
           )
       )
   )

   job_id = trainer_client.train(
       trainer=CustomTrainer(func=train_function),
       options=[patch],
   )

----

Combining Multiple Options
--------------------------

You can pass multiple options together in a single list:

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import Name, Labels, Annotations

   trainer_client = TrainerClient()

   job_id = trainer_client.train(
       trainer=CustomTrainer(func=train_function),
       options=[
           Name("experiment-001"),
           Labels({"project": "llm-finetune"}),
           Annotations({"author": "alice"}),
       ],
   )

----

API Reference
-------------

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
