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

class KubeflowError(Exception):
    """Base class for all Kubeflow SDK errors."""
    pass


class NameResolutionError(KubeflowError):
    """Raised when a resource cannot be found by name."""
    pass


class CompilationError(KubeflowError):
    """Raised when pipeline or job compilation fails."""
    pass


class RunFailedError(KubeflowError):
    """Raised when a run or job reaches a failed state."""
    pass


class KubeflowTimeoutError(KubeflowError, TimeoutError):
    """Raised when a wait operation exceeds its timeout."""
    pass
