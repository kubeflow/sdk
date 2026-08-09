Interactive Sessions
=====================

Connect to Spark interactively from a notebook or script using Spark Connect.

Overview
--------

An interactive session gives you a standard PySpark ``SparkSession`` backed by a
Spark Connect server running on Kubernetes. The SDK provisions the server and
executors for you, but does not delete them automatically — call
``client.delete_session(name)`` when you are done, otherwise the ``SparkConnect``
resource keeps running after your script exits.

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

Key Concepts
------------

**Spark Driver**: The central coordinator that schedules tasks and manages the
execution of a Spark application.

**Executor**: Worker processes that execute Spark tasks and store data partitions.

**Spark Session**: The entry point for interacting with Spark using the DataFrame
and SQL APIs.

**Spark Operator**: A Kubernetes controller that manages the lifecycle of Spark
applications.

The Unified ``connect()`` API
------------------------------

``connect()`` automatically determines the mode based on the parameters you pass:

- **Create mode** - when ``base_url`` is not provided, creates a new Spark Connect
  session with the specified configuration
- **Connect mode** - when ``base_url`` is provided, connects to an existing Spark
  Connect server

.. code-block:: python

   # Create a new session
   spark = client.connect(
       num_executors=5,
       resources_per_executor={"cpu": "5", "memory": "10Gi"},
       spark_conf={"spark.sql.adaptive.enabled": "true"},
   )

   # Connect to an existing server
   spark = client.connect(base_url="sc://server:15002", token="team-token")

   # Minimal usage - default configuration
   spark = client.connect()

.. important::
   In **create mode**, if you're running outside the cluster — a laptop, a CI
   runner, anywhere without ``KUBERNETES_SERVICE_HOST`` set — the SDK
   automatically opens a ``kubectl port-forward`` to the new session's driver
   pod so ``connect()`` can reach it, and keeps that process alive for the
   life of the session. This means:

   - ``kubectl`` must be installed and already configured to reach your
     cluster (same context/credentials ``kubectl get pods`` would use).
   - The first call to ``connect()`` in create mode can take a few seconds
     longer than you might expect — it's waiting for the session to become
     ready, then for the port-forward and the gRPC server behind it to come up.
   - Running from inside the cluster (for example, a Kubeflow Notebook) skips
     this entirely and connects directly via the in-cluster Service.

   **Connect mode** (``base_url=...``) never does this — you're responsible for
   making sure the URL you pass is already reachable.

Common Patterns
----------------

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

**Per-role driver/executor overrides:** ``connect()`` also accepts ``driver``
and ``executor`` objects for settings that apply specifically to the driver or
to executors (rather than to the session as a whole). These are for advanced
cases — most sessions only need ``num_executors``, ``resources_per_executor``,
and ``spark_conf`` above. See :doc:`api` for the current field list.

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

Connecting to an Existing Spark Connect Server
------------------------------------------------

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()

   spark = client.connect(base_url="sc://localhost:15002")

   spark.range(10).show()

This pattern is useful when Spark Connect is already running and managed
independently of your application.

For Kubernetes-native customization of sessions — labels, annotations, node
selection, tolerations, and custom session names — see :doc:`options`.

Session Management
-------------------

Use the Spark SDK to inspect and manage Spark Connect sessions in the configured
Kubernetes namespace (defaults to ``default``).

**List active sessions:**

.. code-block:: python

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

When Things Go Wrong
----------------------

- **Connection timeout** - Verify that the Spark Connect server is running and
  reachable.
- **Session creation failure** - Check Spark Connect logs and available cluster
  resources.
- **Port-forward errors** - When connecting from outside the cluster, ensure the
  Spark Connect server is running and reachable. You can also connect directly to
  an existing Spark Connect endpoint using ``base_url``.
- **Spark application startup issues** - Inspect the Spark Connect server logs and
  verify the Spark Operator is running correctly.
