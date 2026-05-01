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

"""Common options and helper classes used across multiple backends."""

from dataclasses import dataclass
from typing import Any

from kubeflow.trainer.backends.base import RuntimeBackend
from kubeflow.trainer.types.types import BuiltinTrainer, CustomTrainer, CustomTrainerContainer


@dataclass
class Name:
    """Set a custom name for the TrainJob resource.

    This option works with all backends.

    Args:
        name: Custom name for the job. Must be a valid identifier.
    """

    name: str

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: BuiltinTrainer | CustomTrainer | CustomTrainerContainer | None,
        backend: RuntimeBackend,
    ) -> None:
        """Apply custom name to the job specification.

        Args:
            job_spec: Job specification dictionary to modify.
            trainer: Optional trainer instance for context.
            backend: Backend instance for validation and context.
        """
        # Name option is generic - works with all backends
        metadata = job_spec.setdefault("metadata", {})
        metadata["name"] = self.name
