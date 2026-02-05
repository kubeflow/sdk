"""Basic integration tests - checks that tools work together.

These are light integration tests using mocks. Real integration tests
with actual TrainerClient will come later when we add training tools.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.mcp import server
from kubeflow.mcp.tools import discovery


@pytest.fixture(autouse=True)
def prevent_kubeconfig_load(monkeypatch):
    """Stops tests from accidentally loading real kubeconfig."""
    def fail_load(*args, **kwargs):
        raise RuntimeError(
            "Tests should not load real kubeconfig. "
            "Use mocks instead."
        )
    
    monkeypatch.setattr(
        "kubernetes.config.load_kube_config",
        fail_load
    )


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.VersionApi")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_discovery_tools_workflow(
    mock_core_api: MagicMock,
    mock_version_api: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Typical workflow: check cluster, then check resources."""
    # Mock cluster info
    version_obj = SimpleNamespace(git_version="v1.28.0")
    mock_version_api.return_value.get_code.return_value = version_obj
    
    # Mock cluster resources
    node = SimpleNamespace(
        metadata=SimpleNamespace(name="gpu-node"),
        status=SimpleNamespace(
            capacity={"nvidia.com/gpu": "8", "cpu": "32", "memory": "128Gi"}
        ),
    )
    mock_core_api.return_value.list_node.return_value.items = [node]
    
    # Check cluster connectivity
    cluster_info = discovery.get_cluster_info_impl()
    assert cluster_info["connected"] is True
    assert cluster_info["kubernetes_version"] == "v1.28.0"
    
    # Check available resources
    resources = discovery.get_cluster_resources_impl(namespace="ml-team")
    assert resources["namespace"] == "ml-team"
    assert resources["total_gpus"] == 8
    assert len(resources["nodes"]) == 1


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
def test_tools_handle_connection_failure_gracefully(mock_load: MagicMock) -> None:
    """Tools don't crash when connection fails, they return error dicts."""
    mock_load.side_effect = FileNotFoundError("kubeconfig not found")
    
    # Both should return error dicts, not raise
    cluster_info = discovery.get_cluster_info_impl()
    assert cluster_info["connected"] is False
    assert "error" in cluster_info
    
    resources = discovery.get_cluster_resources_impl()
    assert "error" in resources
    assert resources["namespace"] == "default"


def test_server_can_be_imported() -> None:
    """Can import and use the server module."""
    from kubeflow.mcp import server
    from kubeflow.mcp.server import mcp, create_server
    
    assert server is not None
    assert mcp is not None
    assert create_server is not None
    
    # Should be able to create a new server
    new_server = create_server()
    assert new_server is not None
    assert isinstance(new_server, type(mcp))


def test_tools_module_can_be_imported() -> None:
    """Can import the tools module and access its functions."""
    from kubeflow.mcp.tools import discovery
    
    assert discovery is not None
    assert hasattr(discovery, "get_cluster_info_impl")
    assert hasattr(discovery, "get_cluster_resources_impl")
    assert hasattr(discovery, "register_tools")
