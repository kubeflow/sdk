Tools
=====

The Kubeflow MCP Server exposes tools organized by workflow phase — see the catalog below for
the current count.

Tool Catalog
------------

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Phase
     - Tools
     - Description
   * - Planning
     - ``pre_flight``, ``check_compatibility``, ``get_cluster_resources``, ``estimate_resources``
     - Environment validation and resource estimation
   * - Discovery
     - ``list_training_jobs``, ``get_training_job``, ``list_runtimes``, ``get_runtime``
     - Browse jobs and available runtimes
   * - Training
     - ``fine_tune``, ``run_custom_training``, ``run_container_training``
     - Submit LoRA/QLoRA fine-tuning, custom scripts, or container jobs
   * - Monitoring
     - ``get_training_logs``, ``get_training_events``, ``wait_for_training``
     - Track progress and debug failures
   * - Lifecycle
     - ``delete_training_job``, ``update_training_job``
     - Manage existing jobs (ownership-guarded)
   * - Platform
     - ``inspect_crd``, ``inspect_controller``, ``patch_runtime``, ``create_runtime``, ``delete_runtime``
     - Cluster inspection and runtime management
   * - Health
     - ``health_check``, ``get_server_logs``
     - Server diagnostics

Every mutating tool (``fine_tune``, ``run_custom_training``, ``run_container_training``,
``delete_training_job``, ``update_training_job``, ``patch_runtime``, ``create_runtime``,
``delete_runtime``) requires ``confirmed=True`` — the first call always returns a preview.

Tool Discovery Modes
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Mode
     - Tools exposed
     - Description
   * - ``full``
     - up to 23
     - All persona-allowed tools registered directly (default mode; the default ``readonly``
       persona exposes 12 of the 23)
   * - ``progressive``
     - 3 meta-tools
     - Hierarchical discovery (~85 tokens); agents drill down by phase
   * - ``semantic``
     - 2 meta-tools
     - Embedding-search discovery (~69 tokens); agents query by intent

``progressive`` and ``semantic`` modes significantly reduce token consumption for agent
workflows compared to registering all persona-allowed tools directly.

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - MCP Server
     - Kubeflow Trainer
     - Kubeflow SDK
     - Python
     - Kubernetes
   * - 0.1.x
     - >= 2.2.0
     - == 0.4.0
     - 3.10 - 3.12
     - >= 1.27

Health and Readiness
---------------------

Container and Kubernetes probes are available without MCP authentication:

.. code-block:: text

   GET /health  # liveness: the server process is accepting HTTP requests
   GET /ready   # readiness: configured clients imported and packaged resources loaded

``/ready`` returns ``{"status": "ready"}`` with HTTP 200 only when the configured clients
imported successfully **and** all packaged resources loaded; otherwise it returns
``{"status": "not_ready"}`` with HTTP 503. It does not contact Kubernetes or other APIs, so it
is not a live dependency check. A missing packaged resource Markdown file keeps ``/ready`` at
503 even though ``/health`` and registered tools remain available; check the server logs and
package contents rather than cluster dependencies.
