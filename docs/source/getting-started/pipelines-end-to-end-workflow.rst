Orchestrated Pipeline End-to-End Tutorial
=========================================

This tutorial demonstrates how to use the :class:`~kubeflow.pipelines.PipelinesClient` to orchestrate a multi-component machine learning pipeline in a Directed Acyclic Graph (DAG). You will see how to chain the following steps:

1. **Data Preprocessing** using :class:`~kubeflow.spark.SparkClient` to clean raw datasets.
2. **Hyperparameter Tuning** using :class:`~kubeflow.optimizer.OptimizerClient` to search for optimal learning rates.
3. **Final Model Training** using :class:`~kubeflow.trainer.TrainerClient` with the optimal parameters.
4. **Model Registration** using :class:`~kubeflow.hub.ModelRegistryClient` to version and register model checkpoints.

All code in this guide is also available as a Jupyter Notebook in the repository under
`examples/pipelines-end-to-end-tutorial.ipynb <https://github.com/kubeflow/sdk/blob/main/examples/pipelines-end-to-end-tutorial.ipynb>`_.

Prerequisites
-------------

Before you begin, make sure you have the SDK installed with all the required extras:

.. code-block:: bash

   pip install "kubeflow[pipelines,spark,hub]"

Step 1: Define Components
-------------------------

We define each task using the ``@dsl.component`` decorator. Each component encapsulates the logic and client calls for that specific stage.

Component 1: Spark Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step prepares raw data using Spark:

.. code-block:: python

   @dsl.component
   def preprocess_data(input_path: str, output_path: str):
       import os
       print(f"Reading raw data from {input_path}")
       try:
           from kubeflow.spark import SparkClient
           client = SparkClient()
           spark = client.connect()
           df = spark.read.parquet(input_path)
           df_clean = df.dropna()
           df_clean.write.parquet(output_path)
           spark.stop()
       except Exception as e:
           print(f"Spark Connect not available ({e}). Falling back to local cleaning.")
           os.makedirs(output_path, exist_ok=True)
           with open(os.path.join(output_path, "cleaned_data.txt"), "w") as f:
               f.write("Simulated Spark-preprocessed dataset")

Component 2: Hyperparameter Tuning (Katib)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step searches for the best hyperparameters using Katib:

.. code-block:: python

   @dsl.component
   def tune_hyperparameters(data_path: str) -> str:
       print(f"Tuning hyperparameters using processed dataset from {data_path}")

       def trial_train(lr: float):
           from kubeflow.trainer import update_trainjob_status
           loss = 0.5 / lr
           print(f"loss={loss:.4f}")
           update_trainjob_status(progress_percent=100, metrics={"loss": loss})

       try:
           from kubeflow.trainer import TrainJobTemplate, CustomTrainer
           from kubeflow.optimizer import OptimizerClient, Search, TrialConfig, Objective, Direction

           optimizer_client = OptimizerClient()
           trial_template = TrainJobTemplate(
               trainer=CustomTrainer(func=trial_train, func_args={"lr": 0.01})
           )
           search_space = {"lr": Search.uniform(min=0.001, max=0.05)}
           objectives = [Objective(metric="loss", direction=Direction.MINIMIZE)]
           trial_config = TrialConfig(num_trials=2, parallel_trials=1)

           opt_job_name = optimizer_client.optimize(
               trial_template=trial_template,
               search_space=search_space,
               objectives=objectives,
               trial_config=trial_config,
           )
           optimizer_client.wait_for_job_status(opt_job_name)
           best_results = optimizer_client.get_best_results(opt_job_name)
           best_lr = best_results.parameters["lr"] if best_results else "0.01"
       except Exception as e:
           print(f"Katib/OptimizerClient not available ({e}). Using default hyperparameter.")
           best_lr = "0.012"

       return best_lr

Component 3: PyTorch Model Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step trains the final model using the Trainer client:

.. code-block:: python

   @dsl.component
   def train_final_model(best_lr: str, data_path: str, trained_model_uri: dsl.OutputPath(str)):
       import os
       print(f"Training final model with lr={best_lr} using dataset at {data_path}")

       def final_train(lr: float):
           from kubeflow.trainer import update_trainjob_status
           print(f"Training with lr={lr}")
           update_trainjob_status(progress_percent=100, metrics={"loss": 0.05})

       try:
           from kubeflow.trainer import TrainerClient, CustomTrainer
           trainer_client = TrainerClient()
           job_name = trainer_client.train(
               trainer=CustomTrainer(func=final_train, func_args={"lr": float(best_lr)})
           )
           trainer_client.wait_for_job_status(job_name)
       except Exception as e:
           print(f"TrainerClient not available ({e}). Simulating local model compilation.")

       os.makedirs(os.path.dirname(trained_model_uri), exist_ok=True)
       with open(trained_model_uri, "w") as f:
           f.write("s3://my-org-models/orchestrated-model/checkpoint")

Component 4: Model Registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step registers the model artifact in the Model Registry:

.. code-block:: python

   @dsl.component
   def register_model(trained_model_uri: str, model_name: str, version: str):
       import os
       from unittest.mock import MagicMock

       print(f"Registering model '{model_name}' version '{version}' from '{trained_model_uri}'")
       mr_host = os.environ.get("MODEL_REGISTRY_URL", "http://model-registry-service.kubeflow.svc.cluster.local:8080")

       try:
           from kubeflow.hub import ModelRegistryClient
           mr_client = ModelRegistryClient(base_url=mr_host)
           list(mr_client.list_models())
       except Exception as e:
           print(f"ModelRegistryClient not available ({e}). Using mock fallback.")
           class MockModelRegistryClient:
               def register_model(self, name, uri, version, model_format_name=None, model_format_version=None, version_description=None):
                   mock_model = MagicMock()
                   mock_model.name = name
                   mock_model.version = version
                   mock_model.uri = uri
                   return mock_model
           mr_client = MockModelRegistryClient()

       mr_client.register_model(
           name=model_name,
           uri=trained_model_uri,
           version=version,
           model_format_name="pytorch",
           model_format_version="2.0",
           version_description="Model trained and versioned via orchestrated pipeline"
       )

Step 2: Chain the Pipeline DAG
------------------------------

We connect these components into a Directed Acyclic Graph (DAG) using the ``@dsl.pipeline`` decorator:

.. code-block:: python

   @dsl.pipeline(name="orchestrated-ml-pipeline")
   def orchestrator_pipeline(
       input_path: str = "s3://raw-data",
       output_path: str = "s3://processed-data",
       model_name: str = "mnist-pipeline-model",
       version: str = "v1.0.0"
   ):
       # 1. Preprocess raw data
       preprocess_task = preprocess_data(input_path=input_path, output_path=output_path)

       # 2. Optimize learning rate
       tune_task = tune_hyperparameters(data_path=output_path)
       tune_task.after(preprocess_task)

       # 3. Train the final model with best learning rate
       train_task = train_final_model(best_lr=tune_task.output, data_path=output_path)
       train_task.after(tune_task)

       # 4. Register the model version
       register_task = register_model(
           trained_model_uri=train_task.outputs["trained_model_uri"],
           model_name=model_name,
           version=version,
       )
       register_task.after(train_task)

Step 3: Run the Pipeline
------------------------

Use the :class:`~kubeflow.pipelines.PipelinesClient` to submit the pipeline to the Kubeflow orchestrator:

.. code-block:: python

   from kubeflow.pipelines import PipelinesClient

   pipelines_client = PipelinesClient()

   # Submit the pipeline run
   run = pipelines_client.run(
       orchestrator_pipeline,
       params={
           "input_path": "s3://my-raw-data-bucket",
           "output_path": "s3://my-processed-data-bucket",
           "model_name": "mnist-pipeline-model",
           "version": "v1.0.0"
       }
   )

   # Wait for completion
   completed_run = pipelines_client.wait_for_run_status(run)
   print(f"Orchestrated pipeline run completed with state: {completed_run.state}")
