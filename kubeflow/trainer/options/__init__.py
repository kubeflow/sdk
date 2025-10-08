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

"""Training options for the Kubeflow Trainer SDK.

All options are available from this single import location:
    from kubeflow.trainer.options import Name, Labels, PodTemplateOverrides, ...

Options self-validate their backend compatibility at runtime.
Check each option's docstring for supported backends.
"""

from kubeflow.trainer.options.common import Name
from kubeflow.trainer.options.kubernetes import (
    Annotations,
    ContainerOverride,
    Labels,
    PodSpecOverride,
    PodTemplateOverride,
    PodTemplateOverrides,
    SpecAnnotations,
    SpecLabels,
    TrainerArgs,
    TrainerCommand,
)

__all__ = [
    # Common options (all backends)
    "Name",
    # Kubernetes options
    "Annotations",
    "ContainerOverride",
    "Labels",
    "PodSpecOverride",
    "PodTemplateOverride",
    "PodTemplateOverrides",
    "SpecAnnotations",
    "SpecLabels",
    "TrainerArgs",
    "TrainerCommand",
]
