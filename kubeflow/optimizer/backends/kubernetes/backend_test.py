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

"""Unit tests for the KubernetesBackend class in the Kubeflow Optimizer SDK.

This module uses pytest and unittest.mock to simulate Kubernetes API interactions.
It tests that KubernetesBackend.optimize() does not mutate caller-provided inputs.
"""

from unittest.mock import Mock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.optimizer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.optimizer.types.search_types import Search
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types.types import CustomTrainer, TrainJobTemplate


@pytest.fixture
def optimizer_backend():
    """Provide an optimizer KubernetesBackend with mocked Kubernetes APIs."""
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(return_value=None),
            ),
        ),
        patch("kubernetes.client.CoreV1Api", return_value=Mock()),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.verify_backend",
            return_value=None,
        ),
    ):
        backend = KubernetesBackend(KubernetesBackendConfig())
        backend.trainer_backend._get_trainjob_spec = Mock(
            return_value=Mock(to_dict=Mock(return_value={}))
        )
        yield backend


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="single search space parameter",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
            },
        ),
        TestCase(
            name="multiple search space parameters",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                    "epochs": Search.choice([10, 20, 30]),
                },
            },
        ),
    ],
)
def test_optimize(optimizer_backend, test_case):
    """Test that optimize() does not mutate the caller's input objects."""
    print("Executing test:", test_case.name)

    search_space = test_case.config["search_space"]

    original_names = {
        param_name: param_spec.name for param_name, param_spec in search_space.items()
    }

    trial_template = TrainJobTemplate(
        trainer=CustomTrainer(
            func=lambda: None,
            func_args={"existing_arg": "original_value"},
            num_nodes=1,
        ),
    )
    original_func_args = dict(trial_template.trainer.func_args)

    try:
        optimizer_backend.optimize(
            trial_template=trial_template,
            search_space=search_space,
        )

        assert test_case.expected_status == SUCCESS

        # Verify search_space param_spec.name values are unchanged.
        for param_name, param_spec in search_space.items():
            assert param_spec.name == original_names[param_name]

        # Verify trial_template.trainer.func_args is unchanged.
        assert trial_template.trainer.func_args == original_func_args

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert isinstance(e, test_case.expected_error)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="polling_interval greater than timeout raises ValueError",
            expected_status=FAILED,
            config={"name": "test-job", "timeout": 1, "polling_interval": 2},
            expected_error=ValueError,
        ),
        TestCase(
            name="polling_interval equal to timeout raises ValueError",
            expected_status=FAILED,
            config={"name": "test-job", "timeout": 10, "polling_interval": 10},
            expected_error=ValueError,
        ),
        TestCase(
            name="zero polling_interval raises ValueError",
            expected_status=FAILED,
            config={"name": "test-job", "timeout": 10, "polling_interval": 0},
            expected_error=ValueError,
        ),
        TestCase(
            name="negative polling_interval raises ValueError",
            expected_status=FAILED,
            config={"name": "test-job", "timeout": 10, "polling_interval": -1},
            expected_error=ValueError,
        ),
    ],
)
def test_wait_for_job_status(optimizer_backend, test_case):
    """Test KubernetesBackend.wait_for_job_status with various scenarios."""
    print("Executing test:", test_case.name)
    with pytest.raises(test_case.expected_error):
        optimizer_backend.wait_for_job_status(**test_case.config)
