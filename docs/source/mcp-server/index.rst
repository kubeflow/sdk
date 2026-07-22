MCP Server
==========

AI-agent access to Kubeflow Training through the Model Context Protocol.

Overview
--------

The Kubeflow MCP Server exposes Kubeflow Training operations as
`Model Context Protocol <https://modelcontextprotocol.io/>`_ (MCP) tools, enabling AI agents
to plan, submit, monitor, and manage training jobs through natural language.

Instead of writing Kubernetes manifests or Python SDK calls by hand, you describe what you
want in plain English and your AI agent (Claude, Cursor, or any MCP-compatible client)
translates that into the right API calls.

- **Agent-Native** - Tools are auto-discovered via MCP; no manual API wiring needed
- **Guided Workflow** - Phase ordering with next-step hints (Plan, Discover, Train, Monitor)
- **Preview-Before-Submit** - Every mutating operation requires explicit confirmation
- **Security-First** - Persona gating, namespace enforcement, input validation, bearer/JWT auth
- **Multi-Platform** - Auto-detects OpenShift, EKS, GKE with platform-specific guidance

How It Works
------------

1. **Start the server** - Run ``kubeflow-mcp serve --transport http --persona ml-engineer``
2. **Connect an AI agent** - Point Claude, Cursor, or any MCP client at the server
3. **Describe your task** - "Fine-tune Llama-3.2-1B on my dataset with 2 GPUs"
4. **Review and confirm** - The agent shows a preview; you approve before submission
5. **Monitor** - Ask for logs, status, or events in natural language

Quick Example
-------------

Once the MCP server is running, a conversation with your AI agent looks like:

.. code-block:: text

   User: Fine-tune meta-llama/Llama-3.2-1B on hf://my-org/my-dataset
         using 2 GPUs with LoRA

   Agent: I'll set that up. Here's the training job preview:
          - Model: meta-llama/Llama-3.2-1B
          - Dataset: hf://my-org/my-dataset
          - GPUs: 2
          - PEFT: LoRA (rank=8, alpha=16)
          Shall I submit this?

   User: Yes, go ahead.

   Agent: Training job "llama-finetune-abc123" submitted.
          Use "show me the logs" to monitor progress.

Getting Started
---------------

**Install from source:**

.. code-block:: bash

   git clone https://github.com/kubeflow/mcp-server.git
   cd mcp-server
   pip install .

**Run the server (HTTP transport):**

.. code-block:: bash

   kubeflow-mcp serve --transport http --persona ml-engineer

The server connects to your current ``kubeconfig`` context and exposes MCP tools
on ``http://localhost:8000/mcp``. The default transport is ``stdio`` and the default
persona is ``readonly``; use ``--transport http`` for HTTP and a training-capable
persona (``data-scientist``, ``ml-engineer``, or ``platform-admin``) to enable
write operations like ``fine_tune``.

**Run the server (stdio transport for local agents):**

.. code-block:: bash

   kubeflow-mcp serve --persona ml-engineer

When using stdio, configure your AI agent to launch the server directly (see below).

**Run with Docker:**

.. code-block:: bash

   docker run --rm -p 127.0.0.1:8000:8000 \
     -e KUBEFLOW_MCP_AUTH_TOKEN=my-secret-token \
     -e KUBEFLOW_MCP_PERSONA=ml-engineer \
     -e KUBECONFIG=/kubeconfig \
     -v ~/.kube/config:/kubeconfig:ro \
     ghcr.io/kubeflow/mcp-server:latest

The container runs as a non-root user (UID 65532). Mount your kubeconfig to an
explicit path and set ``KUBECONFIG`` accordingly.

Configuring Your AI Agent
-------------------------

**HTTP transport** - add to your MCP client config (e.g. ``.mcp.json``):

.. code-block:: json

   {
     "mcpServers": {
       "kubeflow": {
         "url": "http://localhost:8000/mcp",
         "headers": { "Authorization": "Bearer my-secret-token" }
       }
     }
   }

**stdio transport** - the agent launches the server directly:

.. code-block:: json

   {
     "mcpServers": {
       "kubeflow": {
         "command": "uv",
         "args": ["run", "kubeflow-mcp", "serve"]
       }
     }
   }

The agent will automatically discover all available training tools.

Available Tools
---------------

The MCP server exposes 23 tools organized across seven workflow phases:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Phase
     - Tools
     - Description
   * - **Planning**
     - ``pre_flight``, ``check_compatibility``, ``estimate_resources``, ``get_cluster_resources``
     - Check cluster readiness and GPU requirements
   * - **Discovery**
     - ``list_runtimes``, ``get_runtime``, ``list_training_jobs``, ``get_training_job``
     - Explore available runtimes and existing jobs
   * - **Training**
     - ``fine_tune``, ``run_custom_training``, ``run_container_training``
     - Submit training jobs
   * - **Monitoring**
     - ``get_training_logs``, ``get_training_events``, ``wait_for_training``
     - Track job logs, events, and wait for completion
   * - **Lifecycle**
     - ``delete_training_job``, ``update_training_job``
     - Delete, suspend, or resume training jobs
   * - **Platform**
     - ``inspect_crd``, ``inspect_controller``, ``patch_runtime``, ``create_runtime``, ``delete_runtime``
     - Inspect and manage cluster-level training infrastructure
   * - **Health**
     - ``health_check``, ``get_server_logs``
     - Check server health and view server logs

Security
--------

The server supports multiple authentication methods:

- **Bearer token** - Set ``KUBEFLOW_MCP_AUTH_TOKEN`` for simple token auth
- **JWT/OIDC** - Configure ``KUBEFLOW_MCP_JWKS_URI`` for production deployments

Access is controlled through **personas** (``readonly``, ``data-scientist``,
``ml-engineer``, ``platform-admin``), each with different tool permissions.

Resources
---------

- `GitHub Repository <https://github.com/kubeflow/mcp-server>`_
- `KEP-936 Proposal <https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server>`_
- `Demo Video <https://youtu.be/cZ2BP5hQjc8>`_
- `Roadmap <https://github.com/kubeflow/mcp-server/blob/main/ROADMAP.md>`_
- `Contributing Guide <https://github.com/kubeflow/mcp-server/blob/main/CONTRIBUTING.md>`_
