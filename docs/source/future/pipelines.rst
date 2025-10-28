Pipelines Integration
======================

.. note::

   Kubeflow Pipelines integration with the unified SDK is planned for a future release.

Overview
--------

Kubeflow Pipelines (KFP) is a platform for building and deploying portable,
scalable machine learning workflows. Future versions of the Kubeflow SDK will
provide unified access to Pipelines alongside Trainer and other components.

Planned Features
----------------

The Pipelines SDK integration will include:

- **Pipeline Authoring**: Create ML pipelines using Python
- **Component Library**: Reusable pipeline components
- **Experiment Tracking**: Track pipeline runs and experiments
- **Artifact Management**: Manage inputs, outputs, and models
- **Scheduling**: Schedule recurring pipeline runs
- **Versioning**: Version control for pipelines

Expected API
------------

The API is expected to follow similar patterns to the Trainer SDK:

.. code-block:: python

    from kubeflow.pipelines import PipelinesClient, pipeline, component

    # Future API (not yet implemented)
    @component
    def train_model(data_path: str, model_path: str):
        # Training logic
        pass

    @component
    def evaluate_model(model_path: str) -> float:
        # Evaluation logic
        pass

    @pipeline
    def ml_pipeline(data_path: str):
        train_task = train_model(data_path=data_path, model_path="/tmp/model")
        eval_task = evaluate_model(model_path=train_task.outputs["model_path"])

    # Create and run pipeline
    client = PipelinesClient()
    run_id = client.create_run(pipeline=ml_pipeline, arguments={"data_path": "/data"})

Stay Updated
------------

Follow the Kubeflow SDK repository for updates on Pipelines integration:

- `GitHub Repository <https://github.com/kubeflow/sdk>`_
- `Kubeflow Documentation <https://www.kubeflow.org/docs/>`_
- `Pipelines Documentation <https://www.kubeflow.org/docs/components/pipelines/>`_

Current Pipelines SDK
---------------------

The current Kubeflow Pipelines SDK is available separately:

.. code-block:: bash

    pip install kfp

For documentation and examples, see:

- `KFP SDK Documentation <https://kubeflow-pipelines.readthedocs.io/>`_
- `KFP GitHub <https://github.com/kubeflow/pipelines>`_
- `KFP Examples <https://github.com/kubeflow/pipelines/tree/master/samples>`_

Integration Goals
-----------------

The unified SDK aims to provide:

**Consistent Experience**
    Same client patterns across Trainer, Katib, and Pipelines

**Shared Types**
    Common type definitions for seamless integration

**Simplified Workflows**
    Easy combination of training, tuning, and pipeline execution

**Unified Authentication**
    Single authentication mechanism across components

For more information, see the `Pipelines documentation <https://www.kubeflow.org/docs/components/pipelines/>`_.
