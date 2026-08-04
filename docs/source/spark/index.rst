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

**Set Spark configuration properties:**

.. code-block:: python

   spark = client.connect(
       num_executors=3,
       resources_per_executor={"cpu": "4", "memory": "4Gi"},
       spark_conf={
           "spark.sql.adaptive.enabled": "true",
           "spark.sql.shuffle.partitions": "200",
           "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
       },
   )

``spark_conf`` maps directly to Spark configuration properties and is applied when
the session is created.

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

Advanced Options
-----------------

Beyond ``num_executors``, ``resources_per_executor``, and ``spark_conf``, ``connect()``
accepts an ``options`` list for Kubernetes-native configuration — labels, annotations,
node placement, tolerations, pod template overrides, and session naming. The options
pattern is designed for extensibility: new option types can be added in future SDK
versions without changing the core ``connect()`` signature.

**Labels and annotations**, for resource organization and tooling metadata:

.. code-block:: python

   from kubeflow.spark import Annotations, Labels, SparkClient

   client = SparkClient()

   spark = client.connect(
       num_executors=3,
       resources_per_executor={"cpu": "2", "memory": "4Gi"},
       options=[
           Labels(
               {
                   "app": "spark",
                   "team": "data-engineering",
                   "environment": "production",
               }
           ),
           Annotations(
               {
                   "description": "Daily ETL pipeline for customer data",
                   "owner": "data-team@company.com",
               }
           ),
       ],
   )

**Node selection**, to constrain pods to nodes with matching labels — useful for
dedicated Spark infrastructure or GPU nodes:

.. code-block:: python

   from kubeflow.spark import NodeSelector, SparkClient

   client = SparkClient()

   spark = client.connect(
       num_executors=5,
       resources_per_executor={"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
       options=[
           NodeSelector({"node-type": "spark-gpu", "workload": "ml"}),
       ],
   )

**Tolerations**, to allow scheduling on tainted nodes — for example, dedicated Spark
nodes or spot instances:

.. code-block:: python

   from kubeflow.spark import SparkClient, Toleration

   client = SparkClient()

   spark = client.connect(
       num_executors=10,
       resources_per_executor={"cpu": "8", "memory": "32Gi"},
       options=[
           Toleration(
               key="spot-instance",
               operator="Exists",
               effect="NoSchedule",
           ),
       ],
   )

**Custom session name**, via the ``Name`` option. If not specified, a name is
auto-generated in the form ``spark-connect-{uuid}``:

.. code-block:: python

   from kubeflow.spark import Name, SparkClient

   client = SparkClient()

   spark = client.connect(
       num_executors=3,
       resources_per_executor={"cpu": "2", "memory": "4Gi"},
       options=[Name("custom-session-name")],
   )

**Pod template overrides**, for full control over pod specifications — for example,
security contexts, volumes, or sidecars. Use with caution, since overrides can
conflict with SDK-managed settings:

.. code-block:: python

   from kubeflow.spark import Driver, Executor, PodTemplateOverride, SparkClient

   client = SparkClient()

   spark = client.connect(
       driver=Driver(resources={"cpu": "2", "memory": "4Gi"}),
       executor=Executor(
           num_instances=5,
           resources_per_executor={"cpu": "4", "memory": "8Gi"},
       ),
       options=[
           PodTemplateOverride(
               role="executor",
               template={
                   "spec": {
                       "securityContext": {
                           "runAsUser": 1000,
                           "runAsNonRoot": True,
                       },
                   }
               },
           ),
       ],
   )

Options are composable — production setups typically combine several at once (name,
labels, annotations, node selection, and tolerations together) to fully describe how
a session should run and be scheduled.

Connecting to Existing Spark Connect Servers
--------------------------------------------

You can connect to an existing Spark Connect server instead of creating a new Spark session.

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()

   spark = client.connect(
       base_url="sc://localhost:15002"
   )

   spark.range(10).show()

This pattern is useful when Spark Connect is already running and managed independently of your application.

Session Management
------------------

Use the Spark SDK to inspect and manage Spark Connect sessions in the configured Kubernetes namespace (defaults to ``default``).

**List active sessions:**

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()

   sessions = client.list_sessions()

   for session in sessions:
       print(session.name)
       print(session.state.value)

**Get session information:**

.. code-block:: python

   session = client.get_session(
       "spark-connect-example"
   )

   print(f"Name: {session.name}")
   print(f"State: {session.state.value}")
   print(f"Namespace: {session.namespace}")

**View session logs:**

.. code-block:: python

   for line in client.get_session_logs(
       "spark-connect-example"
   ):
       print(line)

**Delete a session:**

.. code-block:: python

   client.delete_session(
       "spark-connect-example"
   )

When Things Go Wrong
--------------------

**Common issues:**

- **Connection timeout:** Verify that the Spark Connect server is running and reachable.

- **Session creation failure:** Check Spark Connect logs and available cluster resources.

- **Port-forward errors:** When connecting from outside the cluster, ensure the Spark Connect server is running and reachable. You can also connect directly to an existing Spark Connect endpoint using ``base_url``.

- **Spark application startup issues:** Inspect the Spark Connect server logs and verify the Spark Operator is running correctly.
