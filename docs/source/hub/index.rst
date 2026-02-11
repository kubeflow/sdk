Model Registry
==============

Store, version, and manage your machine learning models.

Overview
--------

The Kubeflow Model Registry provides a centralized repository for managing machine learning models. You can:

- **Register models** - Store model metadata and artifact locations
- **Version models** - Track multiple versions of the same model
- **Query models** - Find models by name and retrieve their artifacts
- **Update metadata** - Modify model descriptions and custom metadata

The Model Registry integrates with KServe for model serving, allowing you to deploy registered models directly.

Quick Example
-------------

.. code-block:: python

   from kubeflow.hub import ModelRegistryClient

   # Connect to the registry
   client = ModelRegistryClient(
       base_url="https://registry.example.com",
       author="your-name",
   )

   # Register a model
   model = client.register_model(
       name="my-classifier",
       uri="s3://bucket/models/classifier",
       version="1.0.0",
       model_format_name="pytorch",
   )

   # List all models
   for model in client.list_models():
       print(f"{model.name}: {model.description}")

   # Get a specific version
   version = client.get_model_version("my-classifier", "1.0.0")
   artifact = client.get_model_artifact("my-classifier", "1.0.0")
   print(f"Model URI: {artifact.uri}")

How It Works
------------

1. **Connect** - Create a client pointing to your Model Registry server
2. **Register** - Store model metadata with a URI pointing to the model artifacts
3. **Version** - Each registration creates a new version of the model
4. **Retrieve** - Query models by name and version to get artifact locations

The Model Registry stores metadata only - your model artifacts remain in their original storage location (S3, GCS, etc.).

Key Concepts
------------

**Registered Model**: A named model entity that can have multiple versions.

**Model Version**: A specific version of a registered model, with its own metadata.

**Model Artifact**: The actual model file location (URI) associated with a version.

**Model Format**: The framework format (e.g., "pytorch", "tensorflow", "onnx") used by KServe to select the serving runtime.

Common Patterns
---------------

**Register a PyTorch model:**

.. code-block:: python

   model = client.register_model(
       name="text-classifier",
       uri="s3://models/text-classifier-v1",
       version="1.0.0",
       model_format_name="pytorch",
       model_format_version="2.0",
   )

**List all versions of a model:**

.. code-block:: python

   for version in client.list_model_versions("my-model"):
       print(f"Version: {version.name}, State: {version.state}")

**Update model metadata:**

.. code-block:: python

   model = client.get_model("my-model")
   model.description = "Updated description"
   client.update_model(model)

**Get artifact URI for deployment:**

.. code-block:: python

   artifact = client.get_model_artifact("my-model", "1.0.0")
   print(f"Deploy from: {artifact.uri}")
