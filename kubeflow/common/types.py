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
from datetime import datetime

from kubernetes import client
from pydantic import BaseModel


class KubernetesBackendConfig(BaseModel):
    namespace: str | None = None
    config_file: str | None = None
    context: str | None = None
    client_configuration: client.Configuration | None = None

    class Config:
        arbitrary_types_allowed = True


# Representation for Kubernetes events.
@dataclass
class Event:
    """Event object that represents a Kubernetes event related to a resource.

    Args:
        involved_object_kind (`str`): The kind of object this event is about.
        involved_object_name (`str`): The name of the object this event is about.
        message (`str`): Human-readable description of the event.
        reason (`str`): Short, machine understandable string describing why
            this event was generated.
        event_time (`datetime`): The time at which the event was first recorded.
    """

    involved_object_kind: str
    involved_object_name: str
    message: str
    reason: str
    event_time: datetime
