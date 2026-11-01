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

"""
Unit tests for TrainerClient backend selection.
"""

from unittest.mock import Mock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.test.common import SUCCESS, TestCase
from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.backends.localprocess.types import LocalProcessBackendConfig
from kubeflow.trainer.constants import constants
from kubeflow.trainer.options import RuntimePatch
from kubeflow.trainer.types import types


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "default_backend_is_kubernetes",
            "backend_config": None,
            "expected_backend": "KubernetesBackend",
            "use_k8s_mocks": True,
        },
        {
            "name": "local_process_backend_selection",
            "backend_config": LocalProcessBackendConfig(),
            "expected_backend": "LocalProcessBackend",
            "use_k8s_mocks": False,
        },
        {
            "name": "kubernetes_backend_selection",
            "backend_config": KubernetesBackendConfig(),
            "expected_backend": "KubernetesBackend",
            "use_k8s_mocks": True,
        },
    ],
)
def test_backend_selection(test_case):
    """Test TrainerClient backend selection logic."""
    if test_case["use_k8s_mocks"]:
        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CustomObjectsApi") as mock_custom_api,
            patch("kubernetes.client.CoreV1Api") as mock_core_api,
        ):
            mock_custom_api.return_value = Mock()
            mock_core_api.return_value = Mock()

            if test_case["backend_config"]:
                client = TrainerClient(backend_config=test_case["backend_config"])
            else:
                client = TrainerClient()

            backend_name = client.backend.__class__.__name__
            assert backend_name == test_case["expected_backend"]
    else:
        client = TrainerClient(backend_config=test_case["backend_config"])
        backend_name = client.backend.__class__.__name__
        assert backend_name == test_case["expected_backend"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="single RuntimePatch is submitted",
            expected_status=SUCCESS,
            config={"num_patches": 1},
            expected_output=[{"manager": "trainer.kubeflow.org/kubeflow-sdk"}],
        ),
        TestCase(
            name="multiple RuntimePatch options are rejected before submission",
            expected_status=SUCCESS,
            config={"num_patches": 2},
            expected_error=ValueError,
        ),
    ],
)
def test_train_with_runtime_patches(test_case):
    """Test TrainerClient.train() sends at most one SDK-managed runtime patch."""
    print("Executing test:", test_case.name)

    runtime_trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="torch",
        image="example.com/torch",
    )
    runtime_trainer.set_command(constants.TORCH_COMMAND)
    runtime = types.Runtime(
        name="torch-distributed",
        trainer=runtime_trainer,
        kind=types.RuntimeKind.TRAINING_RUNTIME,
    )
    options = [RuntimePatch() for _ in range(test_case.config["num_patches"])]

    custom_api = Mock()
    with (
        patch("kubernetes.config.load_kube_config"),
        patch("kubernetes.client.CustomObjectsApi", return_value=custom_api),
        patch("kubernetes.client.CoreV1Api", return_value=Mock()),
        patch.object(KubernetesBackend, "verify_backend"),
    ):
        client = TrainerClient(backend_config=KubernetesBackendConfig(namespace="default"))

        if test_case.expected_error:
            with pytest.raises(test_case.expected_error, match="Only one RuntimePatch"):
                client.train(runtime=runtime, options=options)
            assert custom_api.create_namespaced_custom_object.call_count == 0
        else:
            client.train(runtime=runtime, options=options)
            assert custom_api.create_namespaced_custom_object.call_count == 1
            body = custom_api.create_namespaced_custom_object.call_args.args[4]
            assert body["spec"]["runtimePatches"] == test_case.expected_output

    print("test execution complete")
