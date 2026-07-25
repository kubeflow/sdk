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

"""Public API for the Kubeflow Pipelines client, types, and KFP DSL re-exports.

Install with::

    pip install 'kubeflow[pipelines]'

Usage::

    from kubeflow.pipelines import PipelinesClient, dsl, compiler, components, kubernetes
"""

import importlib

__all__ = [
    # Core API
    "PipelinesClient",
    # KFP DSL re-exports
    "compiler",
    "components",
    "dsl",
    "kubernetes",
    # Types
    "Experiment",
    "Pipeline",
    "PipelineVersion",
    "Run",
    # List response types
    "ListExperimentsResponse",
    "ListPipelinesResponse",
    "ListPipelineVersionsResponse",
    "ListRunsResponse",
    # Configuration
    "KubernetesBackendConfig",
    # Constants
    "constants",
]

_KFP_INSTALL_MSG = (
    "kfp is required for kubeflow.pipelines. Install it with: pip install 'kubeflow[pipelines]'"
)

_DSL_MODULES = {"compiler", "components", "dsl", "kubernetes"}

_KFP_ATTRS = {
    "Experiment",
    "KubernetesBackendConfig",
    "ListExperimentsResponse",
    "ListPipelinesResponse",
    "ListPipelineVersionsResponse",
    "ListRunsResponse",
    "Pipeline",
    "PipelineVersion",
    "Run",
    "constants",
}


def __getattr__(name: str):
    if name == "PipelinesClient":
        try:
            from kubeflow.pipelines.api.pipelines_client import PipelinesClient

            return PipelinesClient
        except ImportError as e:
            raise ImportError(_KFP_INSTALL_MSG) from e

    if name in _DSL_MODULES:
        try:
            return importlib.import_module(f"kfp.{name}")
        except ImportError as e:
            raise ImportError(_KFP_INSTALL_MSG) from e

    if name in _KFP_ATTRS:
        try:
            from kfp import kubeflow_client

            return getattr(kubeflow_client, name)
        except ImportError as e:
            raise ImportError(_KFP_INSTALL_MSG) from e

    raise AttributeError(f"module 'kubeflow.pipelines' has no attribute {name!r}")
