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

# SparkApplication CRD (batch jobs via Spark Operator v1beta2)
SPARK_APPLICATION_GROUP = "sparkoperator.k8s.io"
SPARK_APPLICATION_VERSION = "v1beta2"
SPARK_APPLICATION_PLURAL = "sparkapplications"
SPARK_APPLICATION_KIND = "SparkApplication"
SPARK_APPLICATION_TYPE_PYTHON = "Python"
SPARK_APPLICATION_MODE = "cluster"

# Batch job naming and image defaults
JOB_NAME_PREFIX = "spark-job"
JOB_IMAGE_PREFIX = "spark-job"
JOB_SCRIPT_MOUNT_PATH = "/opt/spark/scripts"

# Container names inside Spark pods (set by Spark itself, not configurable via CR)
SPARK_DRIVER_CONTAINER_NAME = "spark-kubernetes-driver"
SPARK_EXECUTOR_CONTAINER_NAME = "spark-kubernetes-executor"

# Maps user-friendly container alias → actual container name
SPARK_CONTAINER_NAME_MAP: dict[str, str] = {
    "driver": SPARK_DRIVER_CONTAINER_NAME,
    "executor": SPARK_EXECUTOR_CONTAINER_NAME,
}

# Polling interval for wait_for_job
SPARK_JOB_POLLING_INTERVAL_SEC = 5

# SparkConnect CRD
SPARK_CONNECT_GROUP = "sparkoperator.k8s.io"
SPARK_CONNECT_VERSION = "v1alpha1"
SPARK_CONNECT_PLURAL = "sparkconnects"
SPARK_CONNECT_KIND = "SparkConnect"

# Default values; keep major.minor aligned with pyspark-connect in pyproject.toml
DEFAULT_SPARK_VERSION = "4.0.1"
DEFAULT_SPARK_IMAGE = "apache/spark:4.0.1"
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

# Remote URI schemes
REMOTE_URI_SCHEMES = ("s3://", "s3a://", "gs://", "gcs://", "http://", "https://", "hdfs://")

# Spark Connect Maven package (required for Connect server main class on classpath)
SPARK_CONNECT_PACKAGE_SCALA_VERSION = "2.13"
