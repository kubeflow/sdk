# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public API for the Kubeflow Spark client and types. Import from kubeflow.spark."""

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.spark.api.spark_client import SparkClient
from kubeflow.spark.types.options import (
    Annotations,
    Labels,
    Name,
    NodeSelector,
    PodTemplateOverride,
    Toleration,
)
from kubeflow.spark.types.types import (
    Driver,
    Executor,
    SparkConnectInfo,
    SparkConnectState,
)
from kubeflow.spark.types.validation import ValidationError

__all__ = [
    # Core API
    "SparkClient",
    # Types
    "Driver",
    "Executor",
    "SparkConnectInfo",
    "SparkConnectState",
    # Options (KEP-107 extensibility pattern - callable pattern like trainer SDK)
    "Annotations",
    "Labels",
    "Name",
    "NodeSelector",
    "PodTemplateOverride",
    "Toleration",
    # Configuration
    "KubernetesBackendConfig",
    # Exceptions
    "ValidationError",
]
