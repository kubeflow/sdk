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

"""Constants for Kubernetes Spark backend."""

# SparkConnect CRD
SPARK_CONNECT_GROUP = "sparkoperator.k8s.io"
SPARK_CONNECT_VERSION = "v1alpha1"
SPARK_CONNECT_PLURAL = "sparkconnects"
SPARK_CONNECT_KIND = "SparkConnect"

# Default values; keep major.minor aligned with PySpark in pyproject.toml [spark] extra
# Pin 3.4.1 to avoid SerializedLambda vs Scala Function3 driver-executor mismatch (3.5.0)
DEFAULT_SPARK_VERSION = "3.4.1"
DEFAULT_SPARK_IMAGE = "apache/spark:3.4.1"
DEFAULT_NUM_EXECUTORS = 1  # Kind-friendly: 1 driver + 1 executor = 2 cores

# Minimal defaults for Kind / resource-constrained clusters (driver and executor)
# CRD cores is integer minimum 1; use 1 core and small memory so 1 node can schedule driver + executors
DEFAULT_DRIVER_CPU = 1
DEFAULT_DRIVER_MEMORY = "512Mi"
DEFAULT_EXECUTOR_CPU = 1
DEFAULT_EXECUTOR_MEMORY = "512Mi"

# Spark Connect server port (must match Spark ConnectCommon.CONNECT_GRPC_BINDING_PORT)
SPARK_CONNECT_PORT = 15002

# Session name prefix
SESSION_NAME_PREFIX = "spark-connect"

# Spark Connect Maven package (required for Connect server main class on classpath)
SPARK_CONNECT_PACKAGE_SCALA_VERSION = "2.12"
