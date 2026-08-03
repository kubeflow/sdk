Configuration
=============

Environment variables, CLI flags, authentication, and observability for the Kubeflow MCP Server.

Environment Variables
----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``MCP_TRANSPORT``
     - ``stdio``
     - Transport protocol (``stdio``, ``http``, ``sse``). The Docker image overrides this
       default to ``http``.
   * - ``KUBEFLOW_MCP_AUTH_TOKEN``
     - *(none)*
     - Bearer token for HTTP auth
   * - ``KUBEFLOW_MCP_JWKS_URI``
     - *(none)*
     - JWKS endpoint for JWT verification (production)
   * - ``KUBEFLOW_MCP_JWT_ISSUER``
     - *(none)*
     - Expected JWT issuer
   * - ``KUBEFLOW_MCP_JWT_AUDIENCE``
     - *(none)*
     - Expected JWT audience
   * - ``KUBEFLOW_MCP_CLIENTS``
     - ``trainer``
     - Comma-separated client modules to load
   * - ``KUBEFLOW_MCP_PERSONA``
     - ``readonly``
     - Tool persona (``readonly``, ``data-scientist``, ``ml-engineer``, ``platform-admin``)
   * - ``KUBEFLOW_MCP_ALLOWED_HOSTS``
     - *(loopback)*
     - Comma-separated ``Host`` header allowlist for DNS rebinding protection; ``:*`` port
       wildcard supported (e.g. ``mcp.example.com,mcp.example.com:*``)
   * - ``KUBEFLOW_MCP_ALLOWED_ORIGINS``
     - *(loopback)*
     - Comma-separated ``Origin`` header allowlist; ``:*`` port wildcard supported
       (e.g. ``https://mcp.example.com``)
   * - ``KUBEFLOW_MCP_DNS_REBINDING_PROTECTION``
     - ``true``
     - Set ``false`` to disable Host/Origin validation (not recommended)
   * - ``LOG_FORMAT``
     - *(auto)*
     - Log format (``json``, ``console``); auto-detected when unset — ``console`` when stderr
       is a TTY, otherwise ``json``
   * - ``LOG_LEVEL``
     - ``INFO``
     - Log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``)

.. note::

   DNS rebinding protection allows only loopback ``Host``/``Origin`` headers by default. When
   exposing the server through a Service or Ingress, set ``KUBEFLOW_MCP_ALLOWED_HOSTS`` (e.g.
   ``KUBEFLOW_MCP_ALLOWED_HOSTS=kubeflow-mcp.kubeflow.svc:*,mcp.example.com``) or requests will
   be rejected with HTTP 421.

For in-cluster deployments, replace ``localhost:8000`` with the Kubernetes Service address and
mount ``KUBEFLOW_MCP_AUTH_TOKEN`` from a Secret.

CLI Reference
--------------

``kubeflow-mcp serve``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   kubeflow-mcp serve \
     --clients trainer \             # modules: trainer, optimizer (stub), hub (stub)
     --persona ml-engineer \         # readonly | data-scientist | ml-engineer | platform-admin
     --mode full \                   # full | progressive | semantic
     --instruction-tier full \       # full | compact | minimal
     --transport stdio \             # stdio | http | sse
     --auth-token SECRET \           # bearer token for HTTP auth (dev/staging)
     --otel-endpoint URL \           # OTLP HTTP endpoint (optional tracing)
     --log-level INFO \              # DEBUG | INFO | WARNING | ERROR
     --log-format console \          # console | json (auto-detected if omitted)
     --no-banner                     # suppress startup banner

Authentication
--------------

When using ``--transport http``, configure auth to secure the endpoint:

.. code-block:: bash

   # Simple API key (dev/staging)
   kubeflow-mcp serve --transport http --auth-token my-secret-token

   # Or via env var
   export KUBEFLOW_MCP_AUTH_TOKEN=my-secret-token
   kubeflow-mcp serve --transport http

   # JWT verification (production)
   export KUBEFLOW_MCP_JWKS_URI=https://auth.example.com/.well-known/jwks.json
   export KUBEFLOW_MCP_JWT_ISSUER=https://auth.example.com
   export KUBEFLOW_MCP_JWT_AUDIENCE=kubeflow-mcp
   kubeflow-mcp serve --transport http

Without auth configured, the server logs a warning that the HTTP endpoint is open.

Security Model
---------------

- **Persona-based tool filtering** restricts which tools are visible to the AI agent (default
  ``--persona readonly``, which exposes only planning, discovery, monitoring, and health tools)
- **Policy file** (``~/.kf-mcp-policy.yaml``) can further restrict tools and namespaces
- **Two-phase confirmation** requires ``confirmed=True`` on write tools (preview first, submit
  after)
- **Input validation** covers Kubernetes name format, CPU/memory format, resource limits, and
  training parameter bounds (batch size, epochs, nodes, GPU count, LoRA rank, script size,
  package count)

``kubeflow-mcp serve`` is single-tenant: one token, one persona, one cluster per instance. In
multi-user HTTP deployments there is no per-user Kubernetes RBAC enforcement at the MCP layer —
configure ``--auth-token`` or JWT and enforce identity at the ingress/gateway layer. See
`ARCHITECTURE.md#security-model <https://github.com/kubeflow/mcp-server/blob/main/ARCHITECTURE.md#security-model>`_
for the full threat model.

Observability
-------------

OpenTelemetry tracing is optional and can be enabled without changing tool code.

.. code-block:: bash

   # Install optional dependencies (a uv dependency group, not a pip extra)
   uv sync --group otel

   # Enable tracing with CLI flag or env var
   kubeflow-mcp serve --otel-endpoint http://localhost:4318
   # or
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   kubeflow-mcp serve

Each tool invocation emits a span with attributes: ``gen_ai.tool.name``, ``tool.args_preview``
(masked, truncated to 300 chars), ``tool.success``, ``tool.duration_ms``, ``kubeflow.persona``,
and ``correlation_id``. The OTLP exporter uses a 2s timeout to avoid blocking tool calls, and
tracing is a no-op when OTel packages are not installed.
