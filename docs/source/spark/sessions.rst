Session Management
==================

Use the Spark SDK to inspect and manage Spark Connect sessions in the
configured Kubernetes namespace (defaults to ``default``).

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
