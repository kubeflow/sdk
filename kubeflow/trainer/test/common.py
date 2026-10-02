# Copyright The Kubeflow Authors.
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

# Shared test utilities and types for Kubeflow Trainer tests.

from dataclasses import dataclass, field
from typing import Any

# Common status constants
SUCCESS = "success"
FAILED = "Failed"
DEFAULT_NAMESPACE = "default"
TIMEOUT = "timeout"
RUNTIME = "runtime"


@dataclass
class TestCase:
    name: str
    expected_status: str = SUCCESS
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None
    expected_error: type[Exception] | None = None
    # Prevent pytest from collecting this dataclass as a test
    __test__ = False
