Connecting to Existing Spark Connect Servers
============================================

You can connect to an existing Spark Connect server instead of creating a new
Spark session.

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()

   spark = client.connect(
       base_url="sc://localhost:15002"
   )

   spark.range(10).show()

This pattern is useful when Spark Connect is already running and managed
independently of your application.
