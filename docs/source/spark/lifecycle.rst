Job Lifecycle
=============

Monitor, list, wait on, and clean up batch Spark jobs.

Overview
--------

Once a job is submitted with ``submit_job()`` (see :doc:`batch-jobs`), SparkClient
provides lifecycle management APIs for tracking it through to completion. These
APIs follow the same pattern as ``TrainerClient``, so if you've used Kubeflow
Trainer, this will look familiar:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - TrainerClient
     - SparkClient
   * - ``train()``
     - ``submit_job()``
   * - ``list_jobs()``
     - ``list_jobs()``
   * - ``get_job()``
     - ``get_job()``
   * - ``get_job_logs()``
     - ``get_job_logs()``
   * - ``wait_for_job_status()``
     - ``wait_for_job_status()``
   * - ``delete_job()``
     - ``delete_job()``

Status Model
------------

Job state is derived from the underlying ``SparkApplication`` resource and
simplified into four SDK-level states:

.. code-block:: python

   class SparkJobStatus(str, Enum):
       CREATED = "Created"
       RUNNING = "Running"
       COMPLETED = "Completed"
       FAILED = "Failed"

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - SDK Status
     - SparkApplication States
   * - ``CREATED``
     - SUBMITTED, or no state reported yet
   * - ``RUNNING``
     - RUNNING, SUCCEEDING, SUSPENDING, SUSPENDED, RESUMING
   * - ``COMPLETED``
     - COMPLETED
   * - ``FAILED``
     - FAILED, SUBMISSION_FAILED, FAILING, PENDING_RERUN, INVALIDATING, UNKNOWN

Any SparkApplication state the SDK doesn't recognize maps to ``FAILED`` and logs a
warning, so newly introduced operator states are handled conservatively.

The ``SparkJob`` Model
------------------------

.. code-block:: python

   @dataclass
   class SparkJob:
       name: str
       namespace: str
       status: SparkJobStatus | None = None
       creation_timestamp: datetime | None = None
       num_executors: int | None = None
       driver_pod_name: str | None = None

Lifecycle APIs
--------------

**Get a job:**

.. code-block:: python

   job = client.get_job(job_name)
   print(f"Status: {job.status}")

**List jobs, optionally filtered by status:**

.. code-block:: python

   from kubeflow.spark import SparkJobStatus

   jobs = client.list_jobs()
   for job in jobs:
       print(f"{job.name}: {job.status}")

   running = client.list_jobs(status={SparkJobStatus.RUNNING})

**Wait for a job to reach a desired status:**

.. code-block:: python

   completed_job = client.wait_for_job_status(job_name, timeout=3600)
   print(f"Final status: {completed_job.status}")

.. important::
   By default, ``wait_for_job_status()`` waits for ``COMPLETED``. If the job
   instead reaches ``FAILED`` — and ``FAILED`` isn't in the ``status`` set
   you're waiting for — it raises a ``RuntimeError`` immediately rather than
   waiting out the timeout. If you want to handle both outcomes yourself
   without an exception, wait on both explicitly:

   .. code-block:: python

      job = client.wait_for_job_status(
          job_name,
          status={SparkJobStatus.COMPLETED, SparkJobStatus.FAILED},
          timeout=3600,
      )
      if job.status == SparkJobStatus.FAILED:
          ...  # handle failure

   ``timeout`` and ``polling_interval`` must both be positive — a zero or
   negative value raises ``ValueError`` before any polling starts.

**Stream logs:**

.. code-block:: python

   for line in client.get_job_logs(job_name, follow=True):
       print(line)

.. note::
   ``get_job_logs()`` reads from the **driver pod only**, via the Kubernetes API.
   Executor-level log access isn't wired in yet — the driver is where Spark
   surfaces stage failures, exceptions, and final job status, so it covers the
   common debugging path. Log retrieval is only available while the driver pod
   exists — if it has been deleted (for example, due to TTL-based cleanup),
   logs may no longer be available.

**Delete a job:**

.. code-block:: python

   client.delete_job(job_name)

.. note::
   ``get_job_logs()`` above covers raw pod logs. Structured metrics, job health,
   event streaming, and Spark UI access are planned as a dedicated
   **Observability** guide that builds on the job model defined here — watch
   this page's "See also" once that lands.

Common Patterns
----------------

**Submit and wait for completion:**

.. code-block:: python

   job_name = client.submit_job(job=FileJob(file_source="https://raw.githubusercontent.com/<repo>/<branch>/etl.py"))
   completed_job = client.wait_for_job_status(job_name, timeout=3600)

**Wait for completion with a timeout:**

.. code-block:: python

   client.wait_for_job_status(job_name, timeout=3600)  # 1 hour max

**List all your running jobs:**

.. code-block:: python

   from kubeflow.spark import SparkJobStatus

   jobs = client.list_jobs(status={SparkJobStatus.RUNNING})
   for job in jobs:
       print(f"{job.name}: {job.status}")

**Clean up after inspecting logs:**

.. code-block:: python

   for line in client.get_job_logs(job_name):
       print(line)

   client.delete_job(job_name)
