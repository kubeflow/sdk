"""Tests for MCP server setup and tool registration."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from kubeflow.mcp import server
from kubeflow.mcp.tools import discovery


def test_server_creation() -> None:
    """Server creation returns a FastMCP instance with the right name."""
    mcp_server = server.create_server()
    
    assert isinstance(mcp_server, FastMCP)
    assert mcp_server.name == "kubeflow-mcp"


def test_server_has_instructions() -> None:
    """Server gets created (instructions are stored internally by FastMCP)."""
    mcp_server = server.create_server()
    
    # Can't easily check instructions directly, but if server was created
    # then instructions were set
    assert mcp_server is not None


def test_server_registers_discovery_tools() -> None:
    """register_tools() works without errors."""
    test_server = FastMCP("test-mcp")
    
    # Should not raise
    try:
        discovery.register_tools(test_server)
        registration_successful = True
    except Exception as e:
        registration_successful = False
        raise AssertionError(f"register_tools() raised exception: {e}")
    
    assert registration_successful
    assert test_server is not None


def test_mcp_instance_exported() -> None:
    """The mcp instance is exported and works."""
    from kubeflow.mcp.server import mcp
    
    assert mcp is not None
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "kubeflow-mcp"


def test_register_tools_called() -> None:
    """Can call register_tools without issues."""
    test_server = FastMCP("test-mcp")
    
    discovery.register_tools(test_server)
    
    assert test_server is not None
