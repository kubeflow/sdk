"""Discovery tools for checking cluster info and resources.

These are read-only - they just look at stuff, never change anything.
"""

from __future__ import annotations

from typing import Any, Dict, List

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from mcp.server.fastmcp import FastMCP


def _load_kube_config() -> None:
    """Loads kubeconfig. Helper function to make mocking easier in tests."""
    k8s_config.load_kube_config()


def get_cluster_info_impl() -> Dict[str, Any]:
    """Checks if we can connect to the cluster and gets the K8s version.
    
    Returns a dict with 'connected' (bool) and 'kubernetes_version' (str),
    or 'error' if something went wrong.
    """
    try:
        _load_kube_config()
        version_api = k8s_client.VersionApi()
        version = version_api.get_code()

        return {
            "connected": True,
            "kubernetes_version": getattr(version, "git_version", None),
        }
    except Exception as exc:
        return {
            "connected": False,
            "error": str(exc),
        }


def get_cluster_resources_impl(namespace: str = "default") -> Dict[str, Any]:
    """Gets GPU/CPU/memory info across all nodes in the cluster.
    
    The namespace param is just for context in the response, it doesn't
    actually filter anything since we're looking at cluster-level resources.
    
    Returns a dict with 'namespace', 'total_gpus' (int), 'nodes' (list),
    or 'error' if something failed.
    """
    try:
        _load_kube_config()
        core_api = k8s_client.CoreV1Api()
        nodes = core_api.list_node().items

        total_gpus = 0
        node_info: List[Dict[str, Any]] = []

        for node in nodes:
            capacity = getattr(node.status, "capacity", {}) or {}
            gpu_str = capacity.get("nvidia.com/gpu")
            gpu_count = int(gpu_str) if gpu_str is not None else 0
            total_gpus += gpu_count

            node_info.append(
                {
                    "name": getattr(node.metadata, "name", None),
                    "gpus": gpu_count,
                    "cpu": capacity.get("cpu"),
                    "memory": capacity.get("memory"),
                }
            )

        return {
            "namespace": namespace,
            "total_gpus": total_gpus,
            "nodes": node_info,
        }
    except Exception as exc:
        return {
            "namespace": namespace,
            "error": str(exc),
        }


def register_tools(server: FastMCP) -> None:
    """Registers the discovery tools with the MCP server."""
    
    @server.tool()
    def get_cluster_info() -> Dict[str, Any]:
        """Checks cluster connection and gets K8s version.
        
        Returns a dict with 'connected' (bool) and 'kubernetes_version' (str)
        if connected, or 'error' (str) if connection failed.
        """
        return get_cluster_info_impl()

    @server.tool()
    def get_cluster_resources(namespace: str = "default") -> Dict[str, Any]:
        """Gets available GPUs, CPU, and memory across all nodes.
        
        Args:
            namespace: Just for context in the response, doesn't filter anything.
        
        Returns a dict with:
        - 'namespace' (str): The namespace you passed in
        - 'total_gpus' (int): Total GPUs across all nodes
        - 'nodes' (list): List of node info with 'name', 'gpus', 'cpu', 'memory'
        - 'error' (str): Present if the query failed
        """
        return get_cluster_resources_impl(namespace=namespace)
