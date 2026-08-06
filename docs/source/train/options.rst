Training Options
================

Training options allow you to customize the metadata and configuration of your training
jobs. They are optional arguments that can be passed to the ``options`` parameter of
``TrainerClient.train()``.

Using Options
-------------

You can import options directly from ``kubeflow.trainer.options``. Here is a simple
example showing how to set a custom job name, labels, and annotations:

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer.options import Name, Labels, Annotations

   client = TrainerClient()

   def train_fn():
       print("Training job running")

   # Submit job with custom name, labels, and annotations
   client.train(
       trainer=CustomTrainer(func=train_fn),
       options=[
           Name("custom-mnist-job"),
           Labels({"team": "ml-platform", "env": "prod"}),
           Annotations({"created-by": "kubeflow-sdk"}),
       ]
   )

Customizing with RuntimePatch
-----------------------------

The ``RuntimePatch`` option is a powerful mechanism for applying structured patches to the
underlying ``TrainJob`` spec (specifically ``.spec.runtimePatches``) on Kubernetes. It
enables you to customize pod templates, volumes, tolerations, node selectors,
scheduling parameters, and container-level settings.

Structure of RuntimePatch
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RuntimePatch`` class contains a nested hierarchy of dataclasses representing
different sections of the job specification:

* ``RuntimePatch``
   * ``training_runtime_spec`` (TrainingRuntimeSpecPatch)
      * ``template`` (JobSetTemplatePatch)
         * ``metadata`` (dict) - Metadata patches (labels, annotations) for the JobSet.
         * ``spec`` (JobSetSpecPatch)
            * ``replicated_jobs`` (list[ReplicatedJobPatch])
               * ``name`` (str) - Name of the replicated job to patch (e.g., ``"node"``
                 or ``"launcher"``).
               * ``template`` (JobTemplatePatch)
                  * ``metadata`` (dict)
                  * ``spec`` (JobSpecPatch)
                     * ``template`` (PodTemplatePatch)
                        * ``metadata`` (dict)
                        * ``spec`` (PodSpecPatch)
                           * ``service_account_name`` (str)
                           * ``volumes`` (list[dict])
                           * ``init_containers`` (list[ContainerPatch])
                           * ``containers`` (list[ContainerPatch])
                              * ``name`` (str) - Name of the container to patch. Must
                                exist in the Runtime (e.g. ``"node"``).
                              * ``env`` (list[dict]) - Not allowed for the ``node``,
                                ``dataset-initializer``, or ``model-initializer``
                                containers; use ``CustomTrainer(env=...)`` or the
                                Initializer API instead.
                              * ``volume_mounts`` (list[dict])
                              * ``security_context`` (dict)
                           * ``image_pull_secrets`` (list[dict])
                           * ``security_context`` (dict)
                           * ``node_selector`` (dict[str, str])
                           * ``affinity`` (dict)
                           * ``tolerations`` (list[dict])
                           * ``scheduling_gates`` (list[dict])

Advanced Patching Example
^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a complete example of using ``RuntimePatch`` to mount a PVC to the training
container, configure a node selector, add a scheduling toleration, and set a container
security context:

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

   client = TrainerClient()

   def train_fn():
       print("Training complete!")

   # Configure the RuntimePatch
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
                                           # Select specific nodes
                                           node_selector={"disk": "ssd"},
                                           # Add tolerations for node taints
                                           tolerations=[
                                               {
                                                   "key": "gpu-taint",
                                                   "operator": "Exists",
                                                   "effect": "NoSchedule",
                                               }
                                           ],
                                           # Define volumes
                                           volumes=[
                                               {
                                                   "name": "data-volume",
                                                   "persistentVolumeClaim": {
                                                       "claimName": "mnist-pvc"
                                                   }
                                               }
                                           ],
                                           # Patch the training container
                                           containers=[
                                               ContainerPatch(
                                                   # Must match a container in the
                                                   # Runtime; "node" is the trainer.
                                                   name="node",
                                                   # Mount the volume
                                                   volume_mounts=[
                                                       {
                                                           "name": "data-volume",
                                                           "mountPath": "/mnt/data",
                                                       }
                                                   ],
                                                   # Harden the container
                                                   security_context={
                                                       "allowPrivilegeEscalation": False,
                                                       "capabilities": {"drop": ["ALL"]},
                                                   },
                                               )
                                           ],
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

   # Submit training job with the patch. Env vars for the "node" container must be
   # set on the trainer, not via ContainerPatch.
   client.train(
       trainer=CustomTrainer(func=train_fn, env={"DATA_DIR": "/mnt/data"}),
       options=[patch],
   )

API Reference
-------------

Below is the complete list of options available in the ``kubeflow.trainer.options`` module.

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
