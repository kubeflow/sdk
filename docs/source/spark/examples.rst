Spark Examples
==============

Use these examples to create Spark sessions, run distributed data operations, and manage sessions with the Kubeflow SDK.

Create a Spark Session
----------------------

Create a session with five executors and configure the resources available to each executor:

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()

   spark = client.connect(
       num_executors=5,
       resources_per_executor={
           "cpu": "2",
           "memory": "2Gi",
       },
   )

   spark.range(10).show()

Run Data Operations
-------------------

Once connected, use the standard PySpark DataFrame and SQL APIs.

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

Connect to an Existing Server
-----------------------------

Connect to an existing Spark Connect server instead of creating a Kubernetes session:

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()
   spark = client.connect(base_url="sc://localhost:15002")

   spark.range(10).show()

This pattern is useful when Spark Connect is already running and managed independently of your application.

Manage Spark Sessions
---------------------

Use the Spark SDK to inspect and manage Spark Connect sessions in the configured Kubernetes namespace, which defaults to ``default``.

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

   session = client.get_session("spark-connect-example")

   print(f"Name: {session.name}")
   print(f"State: {session.state.value}")
   print(f"Namespace: {session.namespace}")

**View session logs:**

.. code-block:: python

   for line in client.get_session_logs("spark-connect-example"):
       print(line)

**Delete a session:**

.. code-block:: python

   client.delete_session("spark-connect-example")
