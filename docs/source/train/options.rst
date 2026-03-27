Options Reference
=================

Options lets us customize how the TrainJob is created. It passes them as a list to the  "options" parameter of 
:py:meth:'kubeflow.trainer.TrainerClient.train' method.

.. code-block:: python

   from kubeflow.trainer import TrainerClient
   from kubeflow.trainer.options import Name, Labels, Annotations

   trainer_client = TrainerClient()
   job_id = trainer_client.train(
      trainer=CustomTrainer(fuc=train_function),
      options=[
         Name("my-train-job"),
         Labels({"env": "prod", "team": "ml"}),
         Annotations({"description": "This is a training experiment-42"})
      ],
   )

.. note:: 
   Not all options work with every backend, each option documents which backend it supports. An uusupported option will raise an "ValueError" during the runtime.

----

Usage Guide
-----------

Name
----
Set a custom name for the TrainJob.
Will work with "all backends".

.. code-block:: python

   from kubeflow.trainer.options import Name

   job_id = trainer_client.train(
      trainer=CustomTrainer(fuc=train_function),
      options=[
         Name("my-custom-job"),
      ],
   )

Labels
------

Add labels to the TrainJob resource metadata "metadata.labels". Will support only on the "kubernetes backend".

.. code-block:: python 
   from kubeflow.trainer.options import Labels

   job_id = trainer_client.train(
      trainer=CustomTrainer(fuc=train_function),
      options=[Labels({"team": "ml-platform", "version": "v2"})],
   )

Annotations
-----------

Add annotations to the TrainJob resource metadata "metadata.annotations". Will support only on the "kubernetes backend".

.. code-block:: python 
   from kubeflow.trainer.options import Annotations

   job_id = trainer_client.train(
      trainer=CustomTrainer(fuc=train_function),
      options=[Annotations({"owner": "ashley", "ticket": "JIRA-42"})],
   )

TrainerCommand
----------------

Override the trainer container command "spec.trainer.command". It can only be used with ''CustomTrainerContainer'' not with ''CustomTrainer'' or ''BuildinTrainer''.

.. code-block:: python

   from kubeflow.trainer.options import (RuntimePatch, TrainingRuntimeSpecPatch, JobSetTemplatePatch, JobSetSpecPatch, ReplicatedJobPatch, JobTemplatePatch, JobSpecPatch, PodTemplatePatch, PodSpecPatch, ContainerPatch)
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

   job_id=client.train(
      trainer = CustomTrainer(func=train_function),
      options=[patch],
      )

Combining Multiple Options
--------------------------

To pass multiple options together in a single list:

..code-block:: python

   from kubeflow.trainer.options import Name, Labels, Annotations

   job_id = client.train(
      trainer=CustomTrainer(fuc=train_function),
      options=[
         Name("experiment-01"),
         Labels({"project": "llm-finetune"}),
         Annotations({"owner": "ashley"})
      ],
   )

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
