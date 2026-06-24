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

For full setup instructions, see `the Spark installation guide. <https://www.kubeflow.org/docs/components/spark-operator/getting-started/>` _

Quick Example
-------------

.. code-block:: python

   from kubeflow.spark import SparkClient

   # Connect to a Spark cluster
   client = SparkClient()

   spark = client.connect(
       num_executors=5,
       resources_per_executor={
           "cpu": "2",
           "memory": "2Gi",
       },
   )

   # Create a distributed DataFrame
   df = spark.range(10)

   # Run a distributed computation
   df.show()

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

Common Patterns
---------------

**Configure executor resources:**

.. code-block:: python

   spark = client.connect(
       num_executors=3,
       resources_per_executor={
           "cpu": "4",
           "memory": "4Gi",
       },
   )

**Create a DataFrame from a range:**

.. code-block:: python

   df = spark.range(100)
   df.show()

**Perform transformations:**

.. code-block:: python

   df = spark.range(10)
   result = df.withColumn("value_squared", df.id * df.id)
   result.show()

**Run SQL queries:**

.. code-block:: python

   df = spark.range(10)
   df.createOrReplaceTempView("numbers")

   result = spark.sql("SELECT id, id * id AS square FROM numbers")
   result.show()

**Aggregate data:**

.. code-block:: python

   df = spark.range(100)

   result = df.groupBy().count()
   result.show()
