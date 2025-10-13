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

from dataclasses import dataclass
from typing import Optional

from kubernetes import client
from pydantic import BaseModel

from kubeflow.trainer.types import types as trainer_types


class KubernetesBackendConfig(BaseModel):
    namespace: Optional[str] = None
    config_file: Optional[str] = None
    context: Optional[str] = None
    client_configuration: Optional[client.Configuration] = None

    class Config:
        arbitrary_types_allowed = True


# TODO (andreyvelich): Add train() and optimize() methods to this class.
@dataclass
class TrainJobTemplate:
    """TrainJob template configuration.

    Args:
        trainer (`CustomTrainer`): Configuration for a CustomTrainer.
        runtime (`Optional[Runtime]`): Optional, reference to one of the existing runtimes. Defaults
            to the torch-distributed runtime if not provided.
        initializer (`Optional[Initializer]`): Optional configuration for the dataset and model
            initializers.
    """

    trainer: trainer_types.CustomTrainer
    runtime: Optional[trainer_types.Runtime] = None
    initializer: Optional[trainer_types.Initializer] = None

    def keys(self):
        return ["trainer", "runtime", "initializer"]

    def __getitem__(self, key):
        return getattr(self, key)
