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

from dataclasses import dataclass, fields
from typing import Optional

from kubeflow_katib_api import models


# Algorithm implementation
@dataclass
class RandomSearch:
    """Random search algorithm.

    Args:
        random_state (`Optional[int]`): Random seed for reproducibility.
    """

    random_state: Optional[int] = None

    def _to_katib_spec(self):
        settings = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                settings.append(
                    models.V1beta1AlgorithmSetting(
                        name=field.name,
                        value=str(value),
                    )
                )

        return models.V1beta1AlgorithmSpec(
            algorithmName="random",
            algorithmSettings=settings or None,
        )
