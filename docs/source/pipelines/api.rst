API Reference
=============

Reference documentation for the Pipelines client and related types.

All classes below are available via ``from kubeflow.pipelines import ...``.

Pipelines Client
----------------

.. autoclass:: kubeflow.pipelines.PipelinesClient
   :members:
   :undoc-members:
   :show-inheritance:

Backend Configuration
---------------------

.. autoclass:: kubeflow.pipelines.KubernetesBackendConfig
   :members:
   :undoc-members:

Types
-----

Available via ``from kubeflow.pipelines import Pipeline, PipelineVersion, Run, Experiment``.

These are type aliases for the auto-generated ``kfp_server_api`` model classes:

- ``Pipeline`` — a pipeline registered on the server
- ``PipelineVersion`` — an immutable snapshot of a pipeline
- ``Run`` — a single execution of a pipeline
- ``Experiment`` — a logical grouping of runs
- ``ListPipelinesResponse`` — paginated list of pipelines
- ``ListPipelineVersionsResponse`` — paginated list of pipeline versions
- ``ListRunsResponse`` — paginated list of runs
- ``ListExperimentsResponse`` — paginated list of experiments

Constants
---------

Available via ``from kubeflow.pipelines import constants``.

.. data:: kubeflow.pipelines.constants.RUN_SUCCEEDED

   ``'succeeded'``

.. data:: kubeflow.pipelines.constants.RUN_FAILED

   ``'failed'``

.. data:: kubeflow.pipelines.constants.RUN_SKIPPED

   ``'skipped'``

.. data:: kubeflow.pipelines.constants.RUN_CANCELED

   ``'canceled'``

.. data:: kubeflow.pipelines.constants.RUN_CANCELING

   ``'canceling'``

.. data:: kubeflow.pipelines.constants.RUN_RUNNING

   ``'running'``

.. data:: kubeflow.pipelines.constants.RUN_PENDING

   ``'pending'``

.. data:: kubeflow.pipelines.constants.RUN_PAUSED

   ``'paused'``

.. data:: kubeflow.pipelines.constants.RUN_COMPLETE

   Alias for :data:`RUN_SUCCEEDED`.

.. data:: kubeflow.pipelines.constants.TERMINAL_STATES

   ``frozenset({'succeeded', 'failed', 'skipped', 'canceled'})``
