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
import logging

from kubernetes import client, config

from kubeflow.common import types as common_types
from kubeflow.trainer.backends.base import ExecutionBackend
from kubeflow.trainer.utils import utils

logger = logging.getLogger(__name__)


class KubernetesBackend(ExecutionBackend):
    def __init__(
        self,
        cfg: common_types.KubernetesBackendConfig,
    ):
        if cfg.namespace is None:
            cfg.namespace = utils.get_default_target_namespace(cfg.context)

        # If client configuration is not set, use kube-config to access Kubernetes APIs.
        if cfg.client_configuration is None:
            # Load kube-config or in-cluster config.
            if cfg.config_file or not utils.is_running_in_k8s():
                config.load_kube_config(config_file=cfg.config_file, context=cfg.context)
            else:
                config.load_incluster_config()

        k8s_client = client.ApiClient(cfg.client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)

        self.namespace = cfg.namespace

    def optimize(self):
        pass
