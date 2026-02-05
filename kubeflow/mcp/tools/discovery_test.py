"""Tests for discovery tools - covers edge cases and error handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.mcp.tools import discovery


@pytest.fixture(autouse=True)
def prevent_kubeconfig_load(monkeypatch):
    """Stops tests from accidentally loading real kubeconfig.
    
    Patches load_kube_config to raise an error if called, so we're forced
    to use mocks instead of connecting to real clusters.
    """
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
def test_get_cluster_info_success(
    mock_version_api: MagicMock, mock_load: MagicMock
) -> None:
    """Happy path - returns connected=True when everything works."""
    version_obj = SimpleNamespace(git_version="v1.28.0")
    mock_version_api.return_value.get_code.return_value = version_obj

    result = discovery.get_cluster_info_impl()

    assert result["connected"] is True
    assert result["kubernetes_version"] == "v1.28.0"
    assert "error" not in result


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.VersionApi")
def test_get_cluster_info_version_missing(
    mock_version_api: MagicMock, mock_load: MagicMock
) -> None:
    """Handles case where version object doesn't have git_version."""
    version_obj = SimpleNamespace()  # No git_version
    mock_version_api.return_value.get_code.return_value = version_obj

    result = discovery.get_cluster_info_impl()

    assert result["connected"] is True
    assert result["kubernetes_version"] is None


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.VersionApi")
def test_get_cluster_info_failure_api_error(
    mock_version_api: MagicMock, mock_load: MagicMock
) -> None:
    """Returns connected=False when API call fails."""
    mock_version_api.return_value.get_code.side_effect = RuntimeError("API connection failed")

    result = discovery.get_cluster_info_impl()

    assert result["connected"] is False
    assert "error" in result
    assert "API connection failed" in result["error"]


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
def test_get_cluster_info_failure_config_error(mock_load: MagicMock) -> None:
    """Returns connected=False when kubeconfig can't be loaded."""
    mock_load.side_effect = FileNotFoundError("kubeconfig not found")

    result = discovery.get_cluster_info_impl()

    assert result["connected"] is False
    assert "error" in result
    assert "kubeconfig not found" in result["error"]


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_with_gpus(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Counts GPUs correctly when nodes have them."""
    node = SimpleNamespace(
        metadata=SimpleNamespace(name="node-1"),
        status=SimpleNamespace(
            capacity={"nvidia.com/gpu": "4", "cpu": "16", "memory": "64Gi"}
        ),
    )
    mock_core_api.return_value.list_node.return_value.items = [node]

    result = discovery.get_cluster_resources_impl(namespace="test-ns")

    assert result["namespace"] == "test-ns"
    assert result["total_gpus"] == 4
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["name"] == "node-1"
    assert result["nodes"][0]["gpus"] == 4
    assert result["nodes"][0]["cpu"] == "16"
    assert result["nodes"][0]["memory"] == "64Gi"
    assert "error" not in result


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_multiple_nodes(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Sums GPUs correctly across multiple nodes."""
    node1 = SimpleNamespace(
        metadata=SimpleNamespace(name="gpu-node-1"),
        status=SimpleNamespace(
            capacity={"nvidia.com/gpu": "8", "cpu": "32", "memory": "128Gi"}
        ),
    )
    node2 = SimpleNamespace(
        metadata=SimpleNamespace(name="gpu-node-2"),
        status=SimpleNamespace(
            capacity={"nvidia.com/gpu": "4", "cpu": "16", "memory": "64Gi"}
        ),
    )
    node3 = SimpleNamespace(
        metadata=SimpleNamespace(name="cpu-node"),
        status=SimpleNamespace(
            capacity={"cpu": "8", "memory": "32Gi"}  # No GPUs
        ),
    )
    mock_core_api.return_value.list_node.return_value.items = [node1, node2, node3]

    result = discovery.get_cluster_resources_impl()

    assert result["total_gpus"] == 12  # 8 + 4 + 0
    assert len(result["nodes"]) == 3
    assert result["nodes"][0]["gpus"] == 8
    assert result["nodes"][1]["gpus"] == 4
    assert result["nodes"][2]["gpus"] == 0


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_no_gpus(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Handles nodes that don't have GPUs."""
    node = SimpleNamespace(
        metadata=SimpleNamespace(name="cpu-only-node"),
        status=SimpleNamespace(
            capacity={"cpu": "8", "memory": "32Gi"}
        ),
    )
    mock_core_api.return_value.list_node.return_value.items = [node]

    result = discovery.get_cluster_resources_impl()

    assert result["total_gpus"] == 0
    assert result["nodes"][0]["gpus"] == 0
    assert result["nodes"][0]["cpu"] == "8"
    assert result["nodes"][0]["memory"] == "32Gi"


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_empty_cluster(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Works fine when cluster has no nodes."""
    mock_core_api.return_value.list_node.return_value.items = []

    result = discovery.get_cluster_resources_impl(namespace="empty-ns")

    assert result["namespace"] == "empty-ns"
    assert result["total_gpus"] == 0
    assert result["nodes"] == []
    assert "error" not in result


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_missing_capacity(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Handles weird nodes that don't have capacity info."""
    node = SimpleNamespace(
        metadata=SimpleNamespace(name="weird-node"),
        status=SimpleNamespace(),  # No capacity
    )
    mock_core_api.return_value.list_node.return_value.items = [node]

    result = discovery.get_cluster_resources_impl()

    assert result["total_gpus"] == 0
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["name"] == "weird-node"
    assert result["nodes"][0]["gpus"] == 0
    assert result["nodes"][0]["cpu"] is None
    assert result["nodes"][0]["memory"] is None


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_missing_node_name(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Handles nodes without names (shouldn't happen but you never know)."""
    node = SimpleNamespace(
        metadata=SimpleNamespace(),  # No name
        status=SimpleNamespace(
            capacity={"nvidia.com/gpu": "2", "cpu": "4"}
        ),
    )
    mock_core_api.return_value.list_node.return_value.items = [node]

    result = discovery.get_cluster_resources_impl()

    assert result["total_gpus"] == 2
    assert result["nodes"][0]["name"] is None
    assert result["nodes"][0]["gpus"] == 2


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_failure_api_error(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Returns error dict when API call fails."""
    mock_core_api.return_value.list_node.side_effect = RuntimeError("API connection failed")

    result = discovery.get_cluster_resources_impl(namespace="test-ns")

    assert "error" in result
    assert result["namespace"] == "test-ns"
    assert "API connection failed" in result["error"]
    assert "total_gpus" not in result
    assert "nodes" not in result


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
def test_get_cluster_resources_failure_config_error(mock_load: MagicMock) -> None:
    """Returns error dict when kubeconfig can't be loaded."""
    mock_load.side_effect = FileNotFoundError("kubeconfig not found")

    result = discovery.get_cluster_resources_impl()

    assert "error" in result
    assert result["namespace"] == "default"
    assert "kubeconfig not found" in result["error"]


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_different_namespaces(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Works with different namespace values (even though it doesn't filter)."""
    node = SimpleNamespace(
        metadata=SimpleNamespace(name="node-1"),
        status=SimpleNamespace(capacity={"nvidia.com/gpu": "1"}),
    )
    mock_core_api.return_value.list_node.return_value.items = [node]

    # Try a few different namespaces
    for ns in ["default", "kubeflow", "ml-team", "production"]:
        result = discovery.get_cluster_resources_impl(namespace=ns)
        assert result["namespace"] == ns
        assert result["total_gpus"] == 1


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.VersionApi")
def test_get_cluster_info_json_serializable(
    mock_version_api: MagicMock, mock_load: MagicMock
) -> None:
    """Makes sure the result can be converted to JSON (for LLM consumption)."""
    import json
    
    version_obj = SimpleNamespace(git_version="v1.28.0")
    mock_version_api.return_value.get_code.return_value = version_obj

    result = discovery.get_cluster_info_impl()

    # Should serialize to JSON without issues
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert parsed["connected"] is True
    assert parsed["kubernetes_version"] == "v1.28.0"


@patch("kubeflow.mcp.tools.discovery.k8s_config.load_kube_config")
@patch("kubeflow.mcp.tools.discovery.k8s_client.CoreV1Api")
def test_get_cluster_resources_json_serializable(
    mock_core_api: MagicMock, mock_load: MagicMock
) -> None:
    """Makes sure the result can be converted to JSON."""
    import json
    
    node = SimpleNamespace(
        metadata=SimpleNamespace(name="node-1"),
        status=SimpleNamespace(
            capacity={"nvidia.com/gpu": "4", "cpu": "16", "memory": "64Gi"}
        ),
    )
    mock_core_api.return_value.list_node.return_value.items = [node]

    result = discovery.get_cluster_resources_impl()

    # Should serialize fine
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert parsed["namespace"] == "default"
    assert parsed["total_gpus"] == 4
    assert isinstance(parsed["nodes"], list)
    assert len(parsed["nodes"]) == 1
