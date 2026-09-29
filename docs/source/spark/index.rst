Spark
=====

Run distributed data processing workloads using Apache Spark.

Overview
--------

Kubeflow provides integration with Apache Spark to run scalable data processing jobs
on Kubernetes. Using the Spark SDK, you can:

- **Run interactive sessions** - Connect to a Spark cluster from a notebook or script
- **Submit batch jobs** - Run existing Spark applications as managed Kubernetes workloads
- **Scale compute resources** - Configure executor counts and resources
- **Process large datasets** - Perform transformations and aggregations across a cluster
- **Track progress** - Monitor logs and job status in real-time

Spark jobs are executed on Kubernetes using the Spark Operator. The operator manages
the lifecycle of Spark driver and executor pods, allowing Spark workloads to run
alongside machine learning pipelines.

Spark is commonly used for:

- Feature engineering
- Data preprocessing
- Dataset generation
- Large-scale batch analytics

Installation
------------

To use Spark with the Kubeflow SDK, install the Spark dependencies:

.. code-block:: bash

   pip install "kubeflow[spark]"

For full setup instructions, see `the Spark installation guide <https://www.kubeflow.org/docs/components/spark-operator/getting-started/>`_.

How It Works
------------

1. You create a ``SparkClient``, optionally pointed at a specific namespace via
   ``KubernetesBackendConfig`` (the only backend supported today)
2. You either connect interactively or submit a batch job
3. The Spark Operator schedules the driver and executor pods on the cluster
4. You monitor progress and retrieve logs or results

By default, SparkClient provisions **1 CPU** and **512Mi** of memory per
executor. You can customize the number of executors and their resource requests
using ``num_executors`` and ``resources_per_executor``.

.. note::

   Batch job submission requires the ``spark-operator-spark`` ServiceAccount to
   exist in the target namespace, with the required SparkApplication RBAC
   permissions bound to it. Otherwise, ``submit_job()`` requests will fail.

   This is a current Spark Operator requirement and is expected to be simplified
   once `kubeflow/spark-operator#3049 <https://github.com/kubeflow/spark-operator/issues/3049>`_
   is resolved.

Two Ways to Run Spark
-----------------------

Choose the approach that fits your workflow:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Approach
     - Best For
     - Example
   * - **Interactive Sessions**
     - Notebooks, ad-hoc exploration, iterative development
     - ``client.connect(num_executors=5, ...)``
   * - **Batch Jobs**
     - Scheduled ETL, CI/CD pipelines, production workflows
     - ``client.submit_job(job=FileJob(...))``

Both approaches share the same ``SparkClient`` and the same resource and Spark
configuration model. Everything below applies across both.

Capabilities
------------

SparkClient is organized around what you're trying to do, not just how you launch
a job. As new capabilities land, they get their own guide here rather than being
folded into Sessions or Batch Jobs.

**Run Spark**

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Interactive Sessions
      :link: sessions
      :link-type: doc

      Connect to Spark from a notebook or script using Spark Connect.

   .. grid-item-card:: Batch Jobs
      :link: batch-jobs
      :link-type: doc

      Submit existing Spark applications as SparkApplication jobs.

**Monitor**

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Job Lifecycle
      :link: lifecycle
      :link-type: doc

      Status model, list/get/wait/logs/delete, and common monitoring patterns.

**Configure**

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Options Reference
      :link: options
      :link-type: doc

      Labels, annotations, node selection, and tolerations. Shared by both
      Sessions and Batch Jobs.

Quick Examples
--------------

**Interactive session:**

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()
   spark = client.connect(
       num_executors=5,
       resources_per_executor={"cpu": "2", "memory": "2Gi"},
   )

   df = spark.range(10)
   df.show()

**Batch job:**

.. code-block:: python

   from kubeflow.spark import FileJob, SparkClient

   client = SparkClient()
   job_name = client.submit_job(
       job=FileJob(
           file_source="https://raw.githubusercontent.com/<repo>/<branch>/daily_pipeline.py",
           args=["--date", "2026-06-18"],
       )
   )

   client.wait_for_job_status(job_name)
