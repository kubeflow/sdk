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

"""Shared test utilities and types for Kubeflow Spark tests."""

from dataclasses import dataclass, field
from typing import Any

# Common status constants
SUCCESS = "success"
FAILED = "failed"
TIMEOUT = "timeout"
RUNTIME = "runtime"
DEFAULT_NAMESPACE = "default"

# SparkConnect states for mocking
SPARK_CONNECT_READY = "spark-connect-ready"
SPARK_CONNECT_PROVISIONING = "spark-connect-provisioning"
SPARK_CONNECT_FAILED = "spark-connect-failed"


@dataclass
class TestCase:
    """Test case dataclass for parametrized tests."""

    name: str
    expected_status: str = SUCCESS
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None
    expected_error: type[Exception] | None = None
    __test__ = False
