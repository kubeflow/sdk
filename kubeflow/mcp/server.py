"""FastMCP server for Kubeflow Trainer.

Wraps the Kubeflow Trainer SDK so AI agents can interact with training
through natural language instead of writing Python code.
"""

from mcp.server.fastmcp import FastMCP

from kubeflow.mcp.tools import discovery


MCP_NAME = "kubeflow-mcp"

MCP_INSTRUCTIONS = """
Kubeflow MCP Server - AI Model Training on Kubernetes

This server provides tools for discovering cluster resources and managing
Kubeflow Training jobs.

WORKFLOW: Discovery
1. get_cluster_info() → Check cluster connectivity
2. get_cluster_resources() → Check available GPUs and resources

More tools (training, monitoring, lifecycle) will be added in follow-up PRs.
"""


def create_server() -> FastMCP:
    """Creates the FastMCP server and registers all tools."""
    server = FastMCP(MCP_NAME, instructions=MCP_INSTRUCTIONS)

    # Only discovery tools for now, more coming later
    discovery.register_tools(server)

    return server


# Main server instance that MCP clients will use
mcp = create_server()

__all__ = ["mcp", "create_server"]
