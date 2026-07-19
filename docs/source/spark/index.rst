Spark
=====

Run distributed data processing workloads using Apache Spark.

Overview
--------

Kubeflow provides integration with Apache Spark to run scalable data processing jobs on Kubernetes. Using the Spark SDK, you can:

- **Create Spark sessions** - Connect to a Spark cluster from Python
- **Run distributed workloads** - Execute Spark DataFrame and SQL operations
- **Scale compute resources** - Configure executor counts and resources
- **Process large datasets** - Perform transformations and aggregations across a cluster

Spark jobs are executed on Kubernetes using the Spark Operator. The operator manages the lifecycle of Spark driver and executor pods, allowing Spark workloads to run alongside machine learning pipelines.

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

1. **Connect** - Create a Spark client and establish a Spark session
2. **Configure resources** - Specify executor count and resource allocation
3. **Submit operations** - Execute DataFrame or SQL transformations
4. **Execute on cluster** - Spark driver coordinates tasks across executor pods

When a Spark session is created, a Spark application is started on the Kubernetes cluster. The Spark driver schedules tasks across executor pods, which perform distributed computation on the data.

Key Concepts
------------

**Spark Driver**: The central coordinator that schedules tasks and manages the execution of a Spark application.

**Executor**: Worker processes that execute Spark tasks and store data partitions.

**Spark Session**: The entry point for interacting with Spark using the DataFrame and SQL APIs.

**Spark Operator**: A Kubernetes controller that manages the lifecycle of Spark applications.

Guides
------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Spark Examples
      :link: examples
      :link-type: doc

      Create sessions, run distributed operations, and manage Spark workloads.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Explore the Spark client, configuration types, and public methods.

When Things Go Wrong
--------------------

**Common issues:**

- **Connection timeout:** Verify that the Spark Connect server is running and reachable.

- **Session creation failure:** Check Spark Connect logs and available cluster resources.

- **Port-forward errors:** When connecting from outside the cluster, ensure the Spark Connect server is running and reachable. You can also connect directly to an existing Spark Connect endpoint using ``base_url``.

- **Spark application startup issues:** Inspect the Spark Connect server logs and verify the Spark Operator is running correctly.
