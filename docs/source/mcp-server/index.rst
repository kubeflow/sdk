MCP Server
==========

Let AI agents drive Kubeflow training workflows through natural language.

Overview
--------

The `Kubeflow MCP Server <https://github.com/kubeflow/mcp-server>`_ exposes Kubeflow Trainer
operations as `Model Context Protocol <https://modelcontextprotocol.io/>`_ tools, built on top
of the Kubeflow SDK's ``TrainerClient``. It lets AI agents (Claude, Cursor, Claude Code, or any
custom agent) plan, submit, monitor, and manage training jobs conversationally, without users
needing to learn Kubernetes or the Kubeflow SDK directly.

It is developed and released as a separate project from the SDK — see
`KEP-936 <https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server>`_
for the design proposal.

Benefits
--------

- **Agent-Native** - Tools are auto-discovered via MCP, with no manual API wiring
- **Guided Workflow** - Phase ordering with next-step hints (Plan → Discover → Train → Monitor)
- **Preview-Before-Submit** - Every mutating operation requires explicit confirmation
- **Security-First** - Persona gating, namespace enforcement, input validation, bearer/JWT auth
- **Multi-Platform** - Auto-detects OpenShift, EKS, and GKE with platform-specific guidance
- **Token-Efficient** - Progressive/semantic modes compress 23 tools into 2-3 meta-tools
- **Extensible** - Plugin architecture for additional Kubeflow clients (optimizer and hub planned)

Installation
------------

.. code-block:: bash

   pip install kubeflow-mcp

Alternatively, install from source:

.. code-block:: bash

   git clone https://github.com/kubeflow/mcp-server.git
   cd mcp-server
   pip install .

Or run the pre-built multi-arch image published to GHCR on every release:

.. code-block:: bash

   docker run --rm -p 8000:8000 \
     -e KUBEFLOW_MCP_AUTH_TOKEN=my-secret-token \
     ghcr.io/kubeflow/mcp-server:latest

The server listens on ``http://localhost:8000/mcp``. See :doc:`configuration` for environment
variables, authentication, and CLI flags.

Quick Start
-----------

.. code-block:: bash

   kubeflow-mcp serve

This defaults to the ``stdio`` transport. For Claude Code, register it directly:

.. code-block:: bash

   claude mcp add kubeflow -- kubeflow-mcp serve

To use the HTTP transport instead (e.g. the Docker image, which defaults to HTTP), start the
server with ``--transport http`` and add it to your agent's MCP client config:

.. code-block:: bash

   kubeflow-mcp serve --transport http --auth-token my-secret-token

.. code-block:: json

   {
     "mcpServers": {
       "kubeflow": {
         "url": "http://localhost:8000/mcp",
         "headers": { "Authorization": "Bearer my-secret-token" }
       }
     }
   }

Example: Fine-Tune a Model via AI Agent
----------------------------------------

Once connected, your AI agent can run a complete training workflow through natural language:

.. code-block:: text

   User: "Fine-tune Llama 3.2 1B on the alpaca dataset"

   Agent calls: check_compatibility()        → ✅ K8s 1.29, Trainer CRD installed
   Agent calls: get_cluster_resources()      → 4x A100 GPUs available
   Agent calls: estimate_resources("meta-llama/Llama-3.2-1B") → needs ~8GB GPU, 1x A100
   Agent calls: list_runtimes()              → torchtune-llama3.2-1b, torchtune-llama3.2-3b, ...
   Agent calls: fine_tune(                   → preview config (confirmed=False)
       model="hf://meta-llama/Llama-3.2-1B",
       dataset="hf://tatsu-lab/alpaca",
       runtime="torchtune-llama3.2-1b"
   )
   Agent calls: fine_tune(..., confirmed=True) → TrainJob "train-llama-abc" created
   Agent calls: get_training_logs("train-llama-abc") → training progress...

Every mutating tool requires ``confirmed=True`` — agents always preview before submitting.

How It Works
------------

1. **Connect** - The server loads Kubeflow SDK clients (``trainer``, with ``optimizer`` and
   ``hub`` planned) and registers their operations as MCP tools
2. **Filter by persona** - The active persona (``readonly``, ``data-scientist``,
   ``ml-engineer``, ``platform-admin``) determines which tools are visible to the caller
3. **Preview, then confirm** - Mutating tools return a preview when called without
   ``confirmed=True``, and only apply the change on a confirmed follow-up call
4. **Guide by phase** - Tool responses include next-step hints so agents follow the
   Plan → Discover → Train → Monitor workflow instead of guessing what to call next

Key Concepts
------------

**Persona**: A server-side role filter restricting which tools are visible to a caller.
Defaults to ``readonly``, which exposes only planning, discovery, monitoring, and health tools.

**Phase**: One of Planning, Discovery, Training, Monitoring, Lifecycle, Platform, or Health —
see :doc:`tools` for the full catalog.

**Two-Phase Confirmation**: Write tools require ``confirmed=True``; the first call always
returns a preview so agents (and the humans supervising them) can review before mutating
cluster state.

**Mode**: ``full`` exposes all tools directly; ``progressive`` and ``semantic`` collapse them
into 2-3 meta-tools for hierarchical or embedding-based discovery, reducing token usage.

See Also
--------

- :doc:`tools` - Full tool catalog by workflow phase, and version compatibility
- :doc:`configuration` - Environment variables, CLI reference, authentication, observability
- `Demo (OSS India) <https://youtu.be/cZ2BP5hQjc8>`_ - Recorded walkthrough of the MCP Server
- `kubeflow/mcp-server on GitHub <https://github.com/kubeflow/mcp-server>`_
- `ROADMAP <https://github.com/kubeflow/mcp-server/blob/main/ROADMAP.md>`_ and
  `SECURITY <https://github.com/kubeflow/mcp-server/blob/main/SECURITY.md>`_
