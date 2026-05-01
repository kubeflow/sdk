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


from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.types import types


class LocalProcessBackendConfig(BaseModel):
    cleanup_venv: bool = True


@dataclass
class LocalRuntimeTrainer(types.RuntimeTrainer):
    packages: list[str] = field(default_factory=list)


class LocalBackendStep(BaseModel):
    step_name: str
    job: LocalJob

    class Config:
        arbitrary_types_allowed = True


class LocalBackendJobs(BaseModel):
    steps: list[LocalBackendStep] | None = []
    runtime: types.Runtime | None = None
    name: str
    created: datetime | None = None
    completed: datetime | None = None

    class Config:
        arbitrary_types_allowed = True
