Data and Model Initializers
===========================

Initializers are pre-training containers that download datasets and pre-trained
models before your training job starts. You declare *what* to fetch; the SDK
runs the download as a separate step and makes the data available to your
training container.

.. note::

   Initializers are supported on the **Container backend** and the
   **Kubernetes backend**. They have no effect on ``LocalProcessBackend``.
   ``DataCacheInitializer`` is only supported on the **Kubernetes backend**.

Available Initializers
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Kind
     - Source
     - Class
   * - Dataset
     - HuggingFace Hub
     - ``HuggingFaceDatasetInitializer``
   * - Dataset
     - S3-compatible
     - ``S3DatasetInitializer``
   * - Dataset
     - Distributed cache
     - ``DataCacheInitializer`` *(Kubernetes only)*
   * - Model
     - HuggingFace Hub
     - ``HuggingFaceModelInitializer``
   * - Model
     - S3-compatible
     - ``S3ModelInitializer``

Pass them via the ``Initializer`` wrapper to ``client.train()``. When both
``dataset`` and ``model`` are set they download **in parallel**, so total wait
time equals the longer of the two.

Dataset Initializers
--------------------

**HuggingFace Hub:**

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer
   from kubeflow.trainer import Initializer, HuggingFaceDatasetInitializer
   from kubeflow.trainer.backends.container.types import ContainerBackendConfig

   client = TrainerClient(backend_config=ContainerBackendConfig())
   client.train(
       initializer=Initializer(
           dataset=HuggingFaceDatasetInitializer(
               storage_uri="hf://username/my-dataset",
               access_token="hf_...",        # required for private repos
           )
       ),
       trainer=CustomTrainer(func=train),
   )

The dataset is available inside the training container at ``/workspace/dataset``.

**S3-compatible storage:**

.. code-block:: python

   from kubeflow.trainer import Initializer, S3DatasetInitializer

   client.train(
       initializer=Initializer(
           dataset=S3DatasetInitializer(
               storage_uri="s3://my-bucket/datasets/my-dataset",
               endpoint="https://minio.example.com",  # omit for AWS S3
               access_key_id="...",
               secret_access_key="...",
               region="us-east-1",
           )
       ),
       trainer=CustomTrainer(func=train),
   )

**Distributed cache (Kubernetes only):**

.. code-block:: python

   from kubeflow.trainer import Initializer, DataCacheInitializer

   client.train(
       initializer=Initializer(
           dataset=DataCacheInitializer(
               storage_uri="cache://my_schema/my_table",
               metadata_loc="s3://my-bucket/iceberg/my_table/metadata/v1.metadata.json",
               num_data_nodes=4,          # must be > 1
               iam_role="arn:aws:iam::123456789012:role/my-role",  # optional
           )
       ),
       trainer=CustomTrainer(func=train),
   )

.. note::

   ``DataCacheInitializer`` requires the **Kubernetes backend**. The
   ``storage_uri`` must follow the ``cache://<SCHEMA_NAME>/<TABLE_NAME>``
   format and ``num_data_nodes`` must be greater than 1.

Model Initializers
------------------

**HuggingFace Hub:**

.. code-block:: python

   from kubeflow.trainer import Initializer, HuggingFaceModelInitializer

   client.train(
       initializer=Initializer(
           model=HuggingFaceModelInitializer(
               storage_uri="hf://meta-llama/Llama-3.2-1B",
               access_token="hf_...",
           )
       ),
       trainer=CustomTrainer(func=fine_tune),
   )

Model weights are available at ``/workspace/model``. By default,
redundant formats (``*.msgpack``, ``*.h5``, ``*.bin``, ``*.pt``, ``*.pth``)
are skipped. Pass ``ignore_patterns=[]`` to download everything.

**S3-compatible storage:**

.. code-block:: python

   from kubeflow.trainer import Initializer, S3ModelInitializer

   client.train(
       initializer=Initializer(
           model=S3ModelInitializer(
               storage_uri="s3://my-models/llama-3.2-1b",
               access_key_id="...",
               secret_access_key="...",
               region="us-east-1",
           )
       ),
       trainer=CustomTrainer(func=fine_tune),
   )

Using Both Together
-------------------

.. code-block:: python

   from kubeflow.trainer import (
       Initializer,
       HuggingFaceDatasetInitializer,
       HuggingFaceModelInitializer,
   )

   client.train(
       initializer=Initializer(
           dataset=HuggingFaceDatasetInitializer(storage_uri="hf://tatsu-lab/alpaca"),
           model=HuggingFaceModelInitializer(
               storage_uri="hf://meta-llama/Llama-3.2-1B",
               access_token="hf_...",
           ),
       ),
       trainer=CustomTrainer(func=fine_tune),
   )

Container Backend Configuration
---------------------------------

Override default images or increase the timeout via ``ContainerBackendConfig``:

.. code-block:: python

   from kubeflow.trainer.backends.container.types import ContainerBackendConfig

   client = TrainerClient(backend_config=ContainerBackendConfig(
       dataset_initializer_image="ghcr.io/kubeflow/trainer/dataset-initializer:latest",
       model_initializer_image="ghcr.io/kubeflow/trainer/model-initializer:latest",
       initializer_timeout=1800,  # seconds, default 600
   ))

Debugging
---------

Fetch logs from a specific initializer step:

.. code-block:: python

   for line in client.get_job_logs(job_name, step="dataset-initializer"):
       print(line)

   for line in client.get_job_logs(job_name, step="model-initializer"):
       print(line)
