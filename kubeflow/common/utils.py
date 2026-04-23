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
import os

from kubernetes import config

from kubeflow.common import constants


def is_running_in_k8s() -> bool:
    return os.path.isdir("/var/run/secrets/kubernetes.io/")


def get_default_target_namespace(context: str | None = None) -> str:
    if not is_running_in_k8s():
        try:
            all_contexts, current_context = config.list_kube_config_contexts()
            # If context is set, we should get namespace from it.
            if context:
                for c in all_contexts:
                    if isinstance(c, dict) and c.get("name") == context:
                        return c["context"]["namespace"]
            # Otherwise, try to get namespace from the current context.
            return current_context["context"]["namespace"]
        except Exception:
            return constants.DEFAULT_NAMESPACE
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        return f.readline()
