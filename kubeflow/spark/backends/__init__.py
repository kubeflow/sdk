"""Spark backends for different execution environments."""

from kubeflow.spark.backends.base import (
    BatchSparkBackend,
    SessionSparkBackend,
    SparkBackend,
)
from kubeflow.spark.backends.connect import ConnectBackend, ConnectBackendConfig
from kubeflow.spark.backends.gateway import GatewayBackend, GatewayBackendConfig
from kubeflow.spark.backends.operator import OperatorBackend, OperatorBackendConfig

__all__ = [
    # Base classes
    "SparkBackend",
    "BatchSparkBackend",
    "SessionSparkBackend",
    # Backend implementations
    "OperatorBackend",
    "OperatorBackendConfig",
    "GatewayBackend",
    "GatewayBackendConfig",
    "ConnectBackend",
    "ConnectBackendConfig",
]
