Katib Integration
==================

.. note::

   Katib integration with the Kubeflow SDK is planned for a future release.

Overview
--------

Katib is Kubeflow's hyperparameter tuning and neural architecture search (NAS)
component. Future versions of the Kubeflow SDK will provide unified access to
Katib for automated ML optimization.

Planned Features
----------------

The Katib SDK integration will include:

- **Hyperparameter Optimization**: Automated hyperparameter tuning
- **Neural Architecture Search**: Automated model architecture discovery
- **Early Stopping**: Intelligent trial termination
- **Multiple Algorithms**: Support for various search algorithms (Grid, Random, Bayesian, etc.)
- **Metric Collection**: Automatic training metrics collection
- **Distributed Trials**: Parallel trial execution

Expected API
------------

The API is expected to follow similar patterns to the Trainer SDK:

.. code-block:: python

    from kubeflow.katib import KatibClient, Experiment

    # Future API (not yet implemented)
    client = KatibClient()

    experiment = Experiment(
        objective={"type": "maximize", "goal": 0.99, "metric": "accuracy"},
        parameters=[
            {"name": "lr", "type": "double", "min": "0.001", "max": "0.1"},
            {"name": "batch_size", "type": "int", "min": "16", "max": "128"},
        ],
        algorithm="bayesianoptimization",
        max_trials=20,
        parallel_trials=3,
    )

    experiment_id = client.create_experiment(experiment)

Stay Updated
------------

Follow the Kubeflow SDK repository for updates on Katib integration:

- `GitHub Repository <https://github.com/kubeflow/sdk>`_
- `Kubeflow Documentation <https://www.kubeflow.org/docs/>`_
- `Katib Documentation <https://www.kubeflow.org/docs/components/katib/>`_

Current Workarounds
-------------------

Until the SDK integration is available, you can use Katib directly:

- Use the Katib Python SDK directly
- Use kubectl to create Katib experiments
- Use the Katib UI for experiment management

For more information, see the `Katib documentation <https://www.kubeflow.org/docs/components/katib/>`_.
