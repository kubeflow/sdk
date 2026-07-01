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

"""Backward-compatible re-export of update_trainjob_status.

Prefer importing directly: ``from kubeflow.trainer import update_trainjob_status``
"""

from kubeflow.trainer.backends.kubernetes.utils import update_trainjob_status

# Legacy alias
update_runtime_status = update_trainjob_status

__all__ = ["update_runtime_status", "update_trainjob_status"]
