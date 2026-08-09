Batch Jobs
==========

Submit existing Spark applications as managed, long-running Kubernetes workloads.

Overview
--------

Batch jobs are built for production workflows — scheduled ETL, CI/CD pipelines, and
anything that should run to completion without an interactive session attached.
``submit_job()`` translates a job definition into a ``SparkApplication`` resource
managed by the Spark Operator.

.. code-block:: python

   from kubeflow.spark import FileJob, SparkClient

   client = SparkClient()

   job_name = client.submit_job(
       job=FileJob(
           file_source="https://raw.githubusercontent.com/<repo>/<branch>/daily_pipeline.py",
           args=["--date", "2024-01-15", "--output", "s3a://bucket/output/"],
       ),
       spark_conf={"spark.sql.adaptive.enabled": "true"},
   )

.. note::
   The job script itself is fetched from ``https://raw.githubusercontent.com/...``
   here — that's the URI scheme with a working, tested example today. The job's
   *output* can still go anywhere Spark can write, including ``s3a://`` — see the
   note on remote sources below for why the script source and the data paths are
   different concerns.

Once submitted, use the :doc:`lifecycle` APIs to monitor, wait on, and clean up the
job.

Two Submission Modes
---------------------

``submit_job()`` dispatches on the type of ``job`` you pass — anything other than
``FileJob`` or ``FuncJob`` raises ``TypeError``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Job Type
     - Mode
     - Use Case
   * - ``job=FileJob(...)``
     - File mode
     - Existing scripts, CI/CD pipelines
   * - ``job=FuncJob(...)``
     - Function mode
     - Python functions kept in an importable module

File Mode
---------

``FileJob`` submits an existing Spark application, referenced by a local or remote
file source.

.. code-block:: python

   from kubeflow.spark import FileJob

   job_name = client.submit_job(
       job=FileJob(
           file_source="https://raw.githubusercontent.com/<repo>/<branch>/daily_pipeline.py",
           args=["--date", "2024-01-15"],
       )
   )

**Remote sources** — ``file_source`` can point at ``s3a://``, ``gs://``,
``hdfs://``, or ``https://``. The SDK doesn't validate or fetch the file itself —
The SDK passes the URI through to the Spark runtime, which is responsible for resolving and accessing it:

.. code-block:: python

   client.submit_job(job=FileJob(file_source="https://raw.githubusercontent.com/<repo>/<branch>/etl.py"))

.. note::
   ``https://`` is the scheme with a working, tested example today (see
   ``examples/spark`` in the ``kubeflow/sdk`` repo). ``s3a://``, ``gs://``, and
   ``hdfs://`` are supported the same way in principle — the SDK just passes the
   URI to Spark — but they haven't been verified end-to-end and may need
   additional filesystem connector packages that aren't guaranteed to be
   preinstalled in every Spark image. Until there's an example notebook
   covering that setup, start with ``https://`` if you want something that's
   known to work.

**Local sources** — when ``file_source`` uses a local URI
(``local:///opt/spark/app/etl.py``), the file must already be available to the
``SparkApplication`` — for example through a mounted PersistentVolumeClaim (PVC) or
a pre-built container image. SparkClient does not package or upload local files
automatically:

.. code-block:: python

   client.submit_job(job=FileJob(file_source="local:///opt/spark/app/etl.py"))

This is passed through as ``spec.mainApplicationFile`` on the generated
``SparkApplication``:

.. code-block:: yaml

   mainApplicationFile: local:///opt/spark/app/etl.py


``FileJob`` fields:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``file_source``
     - ``str``
     - Local or remote path to the Spark application file (required)
   * - ``args``
     - ``list[str] | None``
     - Command-line arguments passed to the application

Function Mode
-------------

``FuncJob`` lets you submit a Python function directly, without authoring a
standalone application file yourself.

**In short:** write a normal Python function, put any imports it needs
*inside* the function body, and pass it to ``FuncJob``. The SDK turns it into
a runnable script for you. The details below explain exactly which functions
qualify and why.

.. code-block:: python

   from kubeflow.spark import FuncJob

   def etl_pipeline(date: str, output_path: str):
       from pyspark.sql import SparkSession
       import pyspark.sql.functions as F

       spark = SparkSession.builder.getOrCreate()
       df = spark.read.parquet(f"s3a://data/raw/{date}/")

       result = (
           df.filter(df.status == "valid")
           .groupBy("category")
           .agg(F.sum("amount").alias("total"))
       )

       result.write.parquet(output_path)

   job_name = client.submit_job(
       job=FuncJob(
           func=etl_pipeline,
           func_args={"date": "2024-01-15", "output_path": "s3a://data/processed/"},
       )
   )

**Function requirements.** The SDK reads the function's source directly via
``inspect.getsource()`` and serializes it — so ``func`` must be a plain,
top-level function defined in an importable ``.py`` module. Lambdas, decorated
functions, async functions, and anything defined interactively (a REPL or
notebook cell) aren't supported — the SDK validates this at submit time and
raises a ``ValueError`` with a specific reason if ``func`` doesn't qualify. If
you're prototyping in a notebook, the function still needs to live in a module
you import from, not be defined inline in a cell.

**func_args must be simple, JSON-like values.** Keys must be strings, and
values are limited to ``None``, ``str``, ``int``, ``bool``, finite floats, and
lists/dicts built from those — no arbitrary Python objects (DataFrames,
custom classes, open file handles, etc.), since ``func_args`` has to be
serialized alongside the function. The SDK also checks that ``func_args``
actually matches your function's signature before submitting, so a typo'd
keyword argument fails fast locally instead of surfacing as a cryptic error
inside the driver pod.

**Imports must live inside the function body.** Only the function's source is
serialized, not its surrounding module — so the function needs to be
self-contained. Any imports it needs (``pyspark.sql``, ``pyspark.sql.functions``,
etc.) have to be placed inside the function itself, as in the example above,
so they're available when the generated script runs:

.. code-block:: python

   def transform():
       from pyspark.sql import SparkSession  # required: import inside the function

       spark = SparkSession.builder.getOrCreate()
       # ...

Under the hood, the SDK serializes the function into a generated script, writes
it to a shared ``emptyDir`` volume through an init container, and points the
driver at the generated file — this is the one case where the SDK does the
packaging step for you. Once submitted, a ``FuncJob`` converges on the same
``SparkJob`` shape as a ``FileJob``, so every :doc:`lifecycle` call treats them
identically.

``FuncJob`` fields:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``func``
     - ``Callable``
     - Top-level, importable Python function (no lambdas, decorators, or
       ``async def``) to run as the Spark application
   * - ``func_args``
     - ``dict``
     - Keyword arguments passed to ``func``. String keys; values limited to
       JSON-like primitives (``None``, ``str``, ``int``, ``bool``, finite
       ``float``, or lists/dicts of the same) — must match ``func``'s
       signature

Resource and Spark Configuration
----------------------------------

Batch jobs use the same resource and Spark configuration model as interactive
sessions:

.. code-block:: python

   client.submit_job(
       job=FileJob(file_source="https://raw.githubusercontent.com/<repo>/<branch>/etl.py"),
       num_executors=10,
       resources_per_executor={"cpu": "4", "memory": "16Gi"},
       spark_conf={"spark.sql.adaptive.enabled": "true"},
   )

For Kubernetes-native customization — labels, annotations, node selection,
tolerations, and custom job names — see :doc:`options`.

.. code-block:: python

   from kubeflow.spark import Labels, Name

   client.submit_job(
       job=FileJob(file_source="https://raw.githubusercontent.com/<repo>/<branch>/etl.py"),
       options=[
           Name("daily-etl-2026-06-18"),
           Labels({"team": "data-engineering"}),
       ],
   )

Next Steps
----------

Once a job is submitted, head to :doc:`lifecycle` to monitor its status, stream
logs, wait for completion, and clean it up.
