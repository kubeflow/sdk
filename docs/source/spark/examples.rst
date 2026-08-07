Spark Examples
==============

Use these end-to-end examples as starting points for data engineering workloads
with Spark Connect. For installation, basic usage, and session management, start
with the :doc:`Spark overview <index>`.

.. note::

   ``SparkClient`` creates or connects to interactive Spark Connect sessions. It
   does not currently submit Python files or JARs. Package application and data
   source dependencies in the Spark image, or make them available to an existing
   Spark Connect server, before running these workflows.

Spark SQL Analytics
-------------------

Create a DataFrame, expose it as a temporary view, and use Spark SQL to calculate
daily revenue by region:

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()
   spark = client.connect(
       num_executors=3,
       resources_per_executor={"cpu": "2", "memory": "4Gi"},
       spark_conf={"spark.sql.adaptive.enabled": "true"},
   )

   orders = spark.createDataFrame(
       [
           ("2025-01-15", "apac", 120.0),
           ("2025-01-15", "emea", 85.5),
           ("2025-01-15", "apac", 64.0),
           ("2025-01-16", "emea", 140.0),
       ],
       schema="order_date string, region string, amount double",
   )
   orders.createOrReplaceTempView("orders")

   daily_revenue = spark.sql(
       """
       SELECT order_date, region, ROUND(SUM(amount), 2) AS revenue
       FROM orders
       GROUP BY order_date, region
       ORDER BY order_date, revenue DESC
       """
   )
   daily_revenue.show()
   spark.stop()

DataFrame ETL Pipeline
----------------------

Spark DataFrame readers and writers accept filesystem URIs supported by the
configured Spark runtime. This pipeline reads partitioned Parquet data, validates
records, enriches them, and writes a partitioned analytics dataset:

.. code-block:: python

   from kubeflow.spark import SparkClient
   from pyspark.sql import functions as F

   client = SparkClient()
   spark = client.connect(
       num_executors=5,
       resources_per_executor={"cpu": "4", "memory": "8Gi"},
       spark_conf={
           "spark.sql.adaptive.enabled": "true",
           "spark.sql.sources.partitionOverwriteMode": "dynamic",
       },
   )

   input_uri = "s3a://raw-data/orders/"
   output_uri = "s3a://analytics/orders-by-day/"

   orders = spark.read.parquet(input_uri)
   cleaned = (
       orders.filter(F.col("order_id").isNotNull())
       .filter(F.col("amount") > 0)
       .withColumn("order_day", F.to_date("created_at"))
       .withColumn("net_amount", F.round(F.col("amount") - F.col("discount"), 2))
       .dropDuplicates(["order_id"])
   )

   daily_summary = cleaned.groupBy("order_day", "region").agg(
       F.count("order_id").alias("order_count"),
       F.round(F.sum("net_amount"), 2).alias("net_revenue"),
   )

   daily_summary.write.mode("overwrite").partitionBy("order_day").parquet(output_uri)
   spark.stop()

The Spark image must include the connector for the URI scheme. Configure
credentials through your platform's workload identity or Kubernetes Secrets
rather than embedding credentials in source code.

Iceberg Tables on MinIO
-----------------------

Use ``spark_conf`` to configure an Iceberg catalog backed by an S3-compatible
MinIO service. The Spark image must contain compatible Iceberg and S3A runtime
dependencies, and the driver and executors must receive credentials through a
secure credential provider.

.. code-block:: python

   from kubeflow.spark import SparkClient

   client = SparkClient()
   spark = client.connect(
       num_executors=4,
       resources_per_executor={"cpu": "2", "memory": "6Gi"},
       spark_conf={
           "spark.sql.catalog.lakehouse": "org.apache.iceberg.spark.SparkCatalog",
           "spark.sql.catalog.lakehouse.type": "hadoop",
           "spark.sql.catalog.lakehouse.warehouse": "s3a://warehouse/iceberg/",
           "spark.hadoop.fs.s3a.endpoint": "http://minio.minio.svc:9000",
           "spark.hadoop.fs.s3a.path.style.access": "true",
           "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
       },
   )

   spark.sql("CREATE NAMESPACE IF NOT EXISTS lakehouse.analytics")
   spark.sql(
       """
       CREATE TABLE IF NOT EXISTS lakehouse.analytics.daily_orders (
           order_day DATE,
           region STRING,
           order_count BIGINT,
           net_revenue DOUBLE
       ) USING iceberg
       PARTITIONED BY (order_day)
       """
   )

   spark.sql(
       """
       INSERT INTO lakehouse.analytics.daily_orders VALUES
           (DATE '2025-01-15', 'apac', 42, 12050.25),
           (DATE '2025-01-15', 'emea', 31, 9980.00)
       """
   )

   spark.sql(
       "SELECT * FROM lakehouse.analytics.daily_orders ORDER BY net_revenue DESC"
   ).show()
   spark.stop()

Production Session Configuration
--------------------------------

Combine explicit driver and executor resources with Kubernetes metadata and
scheduling constraints for a production ETL session:

.. code-block:: python

   from kubeflow.spark import (
       Annotations,
       Driver,
       Executor,
       Labels,
       Name,
       NodeSelector,
       SparkClient,
   )

   session_name = "daily-orders-etl"
   client = SparkClient()
   spark = client.connect(
       driver=Driver(
           resources={"cpu": "2", "memory": "4Gi"},
           service_account="spark-data-pipeline",
       ),
       executor=Executor(
           num_instances=10,
           resources_per_executor={"cpu": "4", "memory": "16Gi"},
       ),
       spark_conf={
           "spark.app.name": "daily-orders-etl",
           "spark.sql.adaptive.enabled": "true",
           "spark.sql.adaptive.coalescePartitions.enabled": "true",
       },
       options=[
           Name(session_name),
           Labels({"team": "data-platform", "workload": "orders-etl"}),
           Annotations({"owner": "data-platform@example.com"}),
           NodeSelector({"node-pool": "data-processing"}),
       ],
   )

   try:
       spark.read.parquet("s3a://raw-data/orders/").groupBy("region").count().show()
   finally:
       spark.stop()
       client.delete_session(session_name)

Use :class:`~kubeflow.spark.types.options.PodTemplateOverride` only when the
higher-level resource and scheduling options cannot express a required pod
configuration, because pod template changes can conflict with SDK-managed fields.
