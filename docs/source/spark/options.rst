Options Reference
==================

Advanced resource configuration options shared by interactive sessions (:doc:`sessions`)
and batch jobs (:doc:`batch-jobs`).

Overview
--------

Beyond ``num_executors``, ``resources_per_executor``, and ``spark_conf``, both
``connect()`` and ``submit_job()`` accept an ``options`` list for advanced resource
configuration — labels, annotations, node placement, tolerations, dynamic executor
allocation, and naming. The options pattern is designed for extensibility: new
option types can be added in future SDK versions without changing the core method
signatures.

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

Dynamic Allocation
------------------

Scale the number of executors with the workload instead of running a fixed
executor count:

.. code-block:: python

   from kubeflow.spark import DynamicAllocation, SparkClient

   client = SparkClient()

   spark = client.connect(
       resources_per_executor={"cpu": "2", "memory": "4Gi"},
       options=[
           DynamicAllocation(
               max_executors=20,
               min_executors=1,
               initial_executors=2,
           ),
       ],
   )

The same option works for batch jobs:

.. code-block:: python

   client.submit_job(
       job=FileJob(file_source="https://raw.githubusercontent.com/<repo>/<branch>/etl.py"),
       options=[DynamicAllocation(max_executors=10)],
   )

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
(name, labels, annotations, node selection, tolerations, and dynamic
allocation together) to fully describe how a session or job should run and
be scheduled.
