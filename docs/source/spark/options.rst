Options Reference
==================

Advanced resource configuration options shared by interactive sessions (:doc:`sessions`)
and batch jobs (:doc:`batch-jobs`).

Overview
--------

Beyond ``num_executors``, ``resources_per_executor``, and ``spark_conf``, both
``connect()`` and ``submit_job()`` accept an ``options`` list for advanced resource
configuration — labels, annotations, node placement, tolerations, dependencies, and naming.
The options pattern is designed for extensibility: new option types can be added in
future SDK versions without changing the core method signatures.

Labels and Annotations
------------------------

For resource organization and tooling metadata:

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

Node Selection
--------------

Constrain Spark pods to nodes with matching Kubernetes labels:

.. code-block:: python

   from kubeflow.spark import NodeSelector, SparkClient

   client = SparkClient()

   spark = client.connect(
        num_executors=5,
        resources_per_executor={
            "cpu": "4",
            "memory": "16Gi",
        },
        options=[
            NodeSelector(
                {
                    "kubernetes.io/os": "linux",
                    "node-pool": "batch",
                }
            ),
        ],
    )

Tolerations
-----------

Allow scheduling on tainted nodes — for example, dedicated Spark nodes or spot
instances:

.. code-block:: python

   from kubeflow.spark import SparkClient, Toleration

   client = SparkClient()

   spark = client.connect(
       num_executors=10,
       resources_per_executor={"cpu": "8", "memory": "32Gi"},
       options=[
           Toleration(key="spot-instance", operator="Exists", effect="NoSchedule"),
       ],
   )

Dependencies
------------

Add JARs and Maven packages that the Spark image does not already include. The SDK
writes them to the resource's ``spec.deps`` field and the Spark Operator passes them to
Spark as ``--jars``, ``--packages``, ``--exclude-packages`` and ``--repositories``:

.. code-block:: python

   from kubeflow.spark import Dependencies, SparkClient

   client = SparkClient()

   spark = client.connect(
       options=[
           Dependencies(
               jars=["local:///opt/spark/jars/my-plugin.jar"],
               packages=["io.delta:delta-spark_2.13:4.0.0"],
               exclude_packages=["org.slf4j:slf4j-api"],
           ),
       ],
   )

The SDK does not add any dependencies on its own. Spark 4.x images, including the
default ``apache/spark:4.0.4`` image, already bundle Spark Connect.

``Dependencies`` also works with ``submit_job()``. Prefer it over setting
``spark.jars``, ``spark.jars.packages``, ``spark.jars.excludes`` or
``spark.jars.repositories`` in ``spark_conf``.

Custom Name
-----------

Set a custom session or job name via the ``Name`` option. If not specified, a name
is auto-generated (``spark-connect-{uuid}`` for sessions, ``spark-job-{uuid}`` for
batch jobs):

.. code-block:: python

   from kubeflow.spark import Name, SparkClient

   client = SparkClient()

   spark = client.connect(
       num_executors=3,
       resources_per_executor={"cpu": "2", "memory": "4Gi"},
       options=[Name("custom-session-name")],
   )

For batch jobs:

.. code-block:: python

   client.submit_job(
       job=FileJob(file_source="https://raw.githubusercontent.com/<repo>/<branch>/etl.py"),
       options=[Name("daily-etl-2026-06-18")],
   )

Composing Options
------------------

Options are composable — production setups typically combine several at once
(name, labels, annotations, node selection, tolerations, and dependencies together) to fully
describe how a session or job should run and be scheduled.
