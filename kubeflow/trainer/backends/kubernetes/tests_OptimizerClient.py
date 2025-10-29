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
Unit tests for the KubernetesBackend class in the Kubeflow Optimizer SDK.

This module uses pytest and unittest.mock to simulate Kubernetes API interactions.
It tests KubernetesBackend's behavior across study listing, resource creation etc
"""

from dataclasses import asdict
import datetime
import multiprocessing
import random
import string
from typing import Optional
from unittest.mock import Mock, patch
import uuid

from kubeflow_trainer_api import models  # CORRECT IMPORT
import pytest

from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import (
    DEFAULT_NAMESPACE,
    FAILED,
    RUNTIME,
    SUCCESS,
    TIMEOUT,
    TestCase,
)
from kubeflow.trainer.types import types
from kubeflow.trainer.utils import utils

# Runtime and study constants
KATIB_RUNTIME = "katib"
KATIB_TUNE_RUNTIME = "katib-tune"

# 2 nodes * 2 nproc
RUNTIME_DEVICES = "4"

FAIL_LOGS = "fail_logs"
LIST_RUNTIMES = "list_runtimes"
BASIC_OPT_STUDY_NAME = "basic-opt-study"
OPT_STUDIES = "experiments"
OPT_STUDY_WITH_BUILTIN = "opt-study-builtin"
OPT_STUDY_WITH_CUSTOM = "opt-study-custom"


# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def kubernetes_backend():
    """Provide a KubernetesBackend with mocked Kubernetes APIs."""
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(side_effect=conditional_error_handler),
                patch_namespaced_custom_object=Mock(side_effect=conditional_error_handler),
                delete_namespaced_custom_object=Mock(side_effect=conditional_error_handler),
                get_namespaced_custom_object=Mock(
                    side_effect=get_namespaced_custom_object_response
                ),
                get_cluster_custom_object=Mock(side_effect=get_cluster_custom_object_response),
                list_namespaced_custom_object=Mock(
                    side_effect=list_namespaced_custom_object_response
                ),
                list_cluster_custom_object=Mock(side_effect=list_cluster_custom_object),
            ),
        ),
        patch(
            "kubernetes.client.CoreV1Api",
            return_value=Mock(
                list_namespaced_pod=Mock(side_effect=list_namespaced_pod_response),
                read_namespaced_pod_log=Mock(side_effect=mock_read_namespaced_pod_log),
            ),
        ),
    ):
        yield KubernetesBackend(KubernetesBackendConfig())


# --------------------------
# Mock Handlers
# --------------------------


def conditional_error_handler(*args, **kwargs):
    """Raise simulated errors based on resource name."""
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[2] == RUNTIME:
        raise RuntimeError()


def list_namespaced_pod_response(*args, **kwargs):
    """Return mock pod list response."""
    pod_list = get_mock_pod_list()
    mock_thread = Mock()
    mock_thread.get.return_value = pod_list
    return mock_thread


def get_mock_pod_list():
    """Create a mocked Kubernetes PodList object with pods for trials."""
    return models.IoK8sApiCoreV1PodList(
        items=[
            models.IoK8sApiCoreV1Pod(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="trial-0-pod",
                    namespace=DEFAULT_NAMESPACE,
                    labels={
                        constants.JOBSET_NAME_LABEL: BASIC_OPT_STUDY_NAME,
                        constants.JOBSET_RJOB_NAME_LABEL: constants.NODE,
                        constants.JOB_INDEX_LABEL: "0",
                    },
                ),
                spec=models.IoK8sApiCoreV1PodSpec(
                    containers=[
                        models.IoK8sApiCoreV1Container(
                            name=constants.NODE,
                            image="katib-trial:latest",
                            command=["python", "-m", "trial"],
                            resources=get_resource_requirements(),
                        )
                    ]
                ),
                status=models.IoK8sApiCoreV1PodStatus(phase="Running"),
            ),
        ]
    )


def get_resource_requirements() -> models.IoK8sApiCoreV1ResourceRequirements:
    """Create a mock ResourceRequirements object for testing."""
    return models.IoK8sApiCoreV1ResourceRequirements(
        requests={
            "nvidia.com/gpu": models.IoK8sApimachineryPkgApiResourceQuantity("1"),
            "memory": models.IoK8sApimachineryPkgApiResourceQuantity("2Gi"),
        },
        limits={
            "nvidia.com/gpu": models.IoK8sApimachineryPkgApiResourceQuantity("1"),
            "memory": models.IoK8sApimachineryPkgApiResourceQuantity("4Gi"),
        },
    )


def get_cluster_custom_object_response(*args, **kwargs):
    """Return a mocked ClusterTrainingRuntime object."""
    mock_thread = Mock()
    if args[3] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[3] == RUNTIME:
        raise RuntimeError()
    if args[2] == constants.CLUSTER_TRAINING_RUNTIME_PLURAL:
        mock_thread.get.return_value = normalize_model(
            create_cluster_training_runtime(name=args[3]),
            models.TrainerV1alpha1ClusterTrainingRuntime,
        )
    return mock_thread


def get_namespaced_custom_object_response(*args, **kwargs):
    """Return a mocked Experiment object."""
    mock_thread = Mock()
    if args[2] == TIMEOUT or args[4] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME or args[4] == RUNTIME:
        raise RuntimeError()
    if args[3] == OPT_STUDIES:
        mock_thread.get.return_value = add_status(create_opt_study(study_name=args[4]))
    return mock_thread


def add_status(
    study: models.KatibV1beta1Experiment,
) -> models.KatibV1beta1Experiment:
    """Add completed status to the study."""
    status = models.KatibV1beta1ExperimentStatus(
        conditions=[
            models.KatibV1beta1ExperimentCondition(
                type="Succeeded",
                status="True",
                lastTransitionTime=datetime.datetime.now(),
                reason="ExperimentCompleted",
                message="All trials finished",
            )
        ]
    )
    study.status = status
    return study


def list_namespaced_custom_object_response(*args, **kwargs):
    """Return a list of mocked Experiment objects."""
    mock_thread = Mock()
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME:
        raise RuntimeError()
    if args[3] == OPT_STUDIES:
        items = [
            add_status(create_opt_study(study_name="opt-study-1")),
            add_status(create_opt_study(study_name="opt-study-2")),
        ]
        mock_thread.get.return_value = normalize_model(
            models.KatibV1beta1ExperimentList(items=items),
            models.KatibV1beta1ExperimentList,
        )
    return mock_thread


def list_cluster_custom_object(*args, **kwargs):
    """Return a generic mocked response for cluster object listing."""
    mock_thread = Mock()
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME:
        raise RuntimeError()
    return mock_thread


def mock_read_namespaced_pod_log(*args, **kwargs):
    """Simulate log retrieval from a pod."""
    if kwargs.get("namespace") == FAIL_LOGS:
        raise Exception("Failed to read logs")
    return "trial log line 1\n"


def normalize_model(model_obj, model_class):
    return model_class.from_dict(model_obj.to_dict())


# --------------------------
# Object Creators
# --------------------------


def create_opt_study(
    study_name: str = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11],
    namespace: str = "default",
) -> models.KatibV1beta1Experiment:
    """Create a mock Katib Experiment object."""
    return models.KatibV1beta1Experiment(
        apiVersion="katib.kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=study_name,
            namespace=namespace,
            creationTimestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        ),
        spec=models.KatibV1beta1ExperimentSpec(
            maxTrialCount=10,
            parallelTrialCount=2,
            objective=models.KatibV1beta1ObjectiveSpec(
                type="minimize",
                goal=0.01,
                objectiveMetricName="validation-loss",
            ),
            algorithm=models.KatibV1beta1AlgorithmSpec(algorithmName="random"),
            trialTemplate=models.KatibV1beta1TrialTemplate(
                primaryContainerName="training-container",
                trialSpec=models.IoK8sApiCoreV1PodSpec(
                    containers=[
                        models.IoK8sApiCoreV1Container(
                            name="training-container",
                            image="katib-trial:latest",
                            command=["python", "-m", "train"],
                        )
                    ]
                ),
            ),
            parameters=[
                models.KatibV1beta1ParameterSpec(
                    name="--lr",
                    parameterType="double",
                    feasibleSpace=models.KatibV1beta1FeasibleSpace(min="0.001", max="0.1"),
                )
            ],
        ),
    )


def create_cluster_training_runtime(
    name: str,
    namespace: str = "default",
) -> models.TrainerV1alpha1ClusterTrainingRuntime:
    """Create a mock ClusterTrainingRuntime (reused from trainer)."""
    return models.TrainerV1alpha1ClusterTrainingRuntime(
        apiVersion=constants.API_VERSION,
        kind="ClusterTrainingRuntime",
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
            labels={constants.RUNTIME_FRAMEWORK_LABEL: name},
        ),
        spec=models.TrainerV1alpha1TrainingRuntimeSpec(
            mlPolicy=models.TrainerV1alpha1MLPolicy(
                torch=models.TrainerV1alpha1TorchMLPolicySource(
                    numProcPerNode=models.IoK8sApimachineryPkgUtilIntstrIntOrString(2)
                ),
                numNodes=2,
            ),
            template=models.TrainerV1alpha1JobSetTemplateSpec(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name=name,
                    namespace=namespace,
                ),
                spec=models.JobsetV1alpha2JobSetSpec(replicatedJobs=[]),
            ),
        ),
    )


def create_runtime_type(name: str) -> types.Runtime:
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=name,
        num_nodes=2,
        device="gpu",
        device_count=RUNTIME_DEVICES,
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return types.Runtime(name=name, pretrained_model=None, trainer=trainer)


def get_opt_study_data_type(runtime_name: str, study_name: str) -> types.OptStudy:
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=runtime_name,
        device="gpu",
        device_count=RUNTIME_DEVICES,
        num_nodes=2,
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return types.OptStudy(
        name=study_name,
        creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        runtime=types.Runtime(name=runtime_name, pretrained_model=None, trainer=trainer),
        steps=[
            types.Step(
                name="trial-0",
                status="Running",
                pod_name="trial-0-pod",
                device="gpu",
                device_count="1",
            )
        ],
        num_nodes=2,
        status="Complete",
    )


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": KATIB_RUNTIME},
            expected_output=create_runtime_type(name=KATIB_RUNTIME),
        ),
        TestCase(
            name="timeout error when getting runtime",
            expected_status=FAILED,
            config={"name": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting runtime",
            expected_status=FAILED,
            config={"name": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_runtime(kubernetes_backend, test_case):
    print("Executing test:", test_case.name)
    try:
        runtime = kubernetes_backend.get_runtime(**test_case.config)
        assert test_case.expected_status == SUCCESS
        assert isinstance(runtime, types.Runtime)
        assert asdict(runtime) == asdict(test_case.expected_output)
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": LIST_RUNTIMES},
            expected_output=[
                create_runtime_type(name="runtime-1"),
                create_runtime_type(name="runtime-2"),
            ],
        ),
    ],
)
def test_list_runtimes(kubernetes_backend, test_case):
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        runtimes = kubernetes_backend.list_runtimes()
        assert test_case.expected_status == SUCCESS
        assert isinstance(runtimes, list)
        assert all(isinstance(r, types.Runtime) for r in runtimes)
        assert [asdict(r) for r in runtimes] == [asdict(r) for r in test_case.expected_output]
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={},
            expected_output=create_opt_study(study_name=BASIC_OPT_STUDY_NAME),
        ),
        TestCase(
            name="timeout error when submitting study",
            expected_status=FAILED,
            config={"namespace": TIMEOUT},
            expected_error=TimeoutError,
        ),
    ],
)
def test_optimize(kubernetes_backend, test_case):
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        study_name = kubernetes_backend.optimize(
            runtime=create_runtime_type(KATIB_RUNTIME),
            search_space={"lr": (0.001, 0.1)},
            objective="minimize",
        )
        assert test_case.expected_status == SUCCESS
        expected = test_case.expected_output
        expected.metadata.name = study_name
        kubernetes_backend.custom_api.create_namespaced_custom_object.assert_called_with(
            "katib.kubeflow.org",
            "v1beta1",
            DEFAULT_NAMESPACE,
            OPT_STUDIES,
            expected.to_dict(),
        )
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_OPT_STUDY_NAME},
            expected_output=get_opt_study_data_type(KATIB_RUNTIME, BASIC_OPT_STUDY_NAME),
        ),
        TestCase(
            name="timeout error when getting study",
            expected_status=FAILED,
            config={"name": TIMEOUT},
            expected_error=TimeoutError,
        ),
    ],
)
def test_get_study(kubernetes_backend, test_case):
    print("Executing test:", test_case.name)
    try:
        study = kubernetes_backend.get_study(**test_case.config)
        assert test_case.expected_status == SUCCESS
        assert asdict(study) == asdict(test_case.expected_output)
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={},
            expected_output=[
                get_opt_study_data_type(KATIB_RUNTIME, "opt-study-1"),
                get_opt_study_data_type(KATIB_RUNTIME, "opt-study-2"),
            ],
        ),
    ],
)
def test_list_studies(kubernetes_backend, test_case):
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        studies = kubernetes_backend.list_studies()
        assert test_case.expected_status == SUCCESS
        assert len(studies) == 2
        assert [asdict(s) for s in studies] == [asdict(r) for r in test_case.expected_output]
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_OPT_STUDY_NAME},
            expected_output=["trial log line 1\n"],
        ),
    ],
)
def test_get_study_logs(kubernetes_backend, test_case):
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        logs = kubernetes_backend.get_study_logs(test_case.config.get("name"))
        logs_list = list(logs)
        assert test_case.expected_status == SUCCESS
        assert logs_list == test_case.expected_output
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_OPT_STUDY_NAME},
            expected_output=None,
        ),
    ],
)
def test_delete_study(kubernetes_backend, test_case):
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        kubernetes_backend.delete_study(test_case.config.get("name"))
        assert test_case.expected_status == SUCCESS
        kubernetes_backend.custom_api.delete_namespaced_custom_object.assert_called_with(
            "katib.kubeflow.org",
            "v1beta1",
            DEFAULT_NAMESPACE,
            OPT_STUDIES,
            name=test_case.config.get("name"),
        )
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")