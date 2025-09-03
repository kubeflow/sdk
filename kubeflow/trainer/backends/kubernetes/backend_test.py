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
Unit tests for the KubernetesBackend class in the Kubeflow Trainer SDK.

This module uses pytest and unittest.mock to simulate Kubernetes API interactions.
It tests KubernetesBackend's behavior across job listing, resource creation etc
"""

import datetime
import multiprocessing
import random
import string
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Type
from unittest.mock import Mock, patch

import pytest
from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types
from kubeflow.trainer.utils import utils
from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend


@dataclass
class TestCase:
    name: str
    expected_status: str
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Any] = None
    expected_error: Optional[Type[Exception]] = None
    __test__ = False


# --------------------------
# Constants for test scenarios
# --------------------------
TIMEOUT = "timeout"
RUNTIME = "runtime"
SUCCESS = "success"
FAILED = "Failed"
DEFAULT_NAMESPACE = "default"
# In all tests runtime name is equal to the framework name.
TORCH_RUNTIME = "torch"
TORCH_TUNE_RUNTIME = "torchtune"

# 2 nodes * 2 nproc
RUNTIME_DEVICES = "4"

FAIL_LOGS = "fail_logs"
LIST_RUNTIMES = "list_runtimes"
BASIC_TRAIN_JOB_NAME = "basic-job"
TRAIN_JOBS = "trainjobs"
TRAIN_JOB_WITH_BUILT_IN_TRAINER = "train-job-with-built-in-trainer"
TRAIN_JOB_WITH_CUSTOM_TRAINER = "train-job-with-custom-trainer"


# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def kubernetes_backend(request):
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
    """Create a mocked Kubernetes PodList object with pods for different training steps."""
    return models.IoK8sApiCoreV1PodList(
        items=[
            # Dataset initializer pod
            models.IoK8sApiCoreV1Pod(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="dataset-initializer-pod",
                    namespace=DEFAULT_NAMESPACE,
                    labels={
                        constants.JOBSET_NAME_LABEL: BASIC_TRAIN_JOB_NAME,
                        constants.JOBSET_RJOB_NAME_LABEL: constants.DATASET_INITIALIZER,
                        constants.JOB_INDEX_LABEL: "0",
                    },
                ),
                spec=models.IoK8sApiCoreV1PodSpec(
                    containers=[
                        models.IoK8sApiCoreV1Container(
                            name=constants.DATASET_INITIALIZER,
                            image="dataset-initializer:latest",
                            command=["python", "-m", "dataset_initializer"],
                        )
                    ]
                ),
                status=models.IoK8sApiCoreV1PodStatus(phase="Running"),
            ),
            # Model initializer pod
            models.IoK8sApiCoreV1Pod(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="model-initializer-pod",
                    namespace=DEFAULT_NAMESPACE,
                    labels={
                        constants.JOBSET_NAME_LABEL: BASIC_TRAIN_JOB_NAME,
                        constants.JOBSET_RJOB_NAME_LABEL: constants.MODEL_INITIALIZER,
                        constants.JOB_INDEX_LABEL: "0",
                    },
                ),
                spec=models.IoK8sApiCoreV1PodSpec(
                    containers=[
                        models.IoK8sApiCoreV1Container(
                            name=constants.MODEL_INITIALIZER,
                            image="model-initializer:latest",
                            command=["python", "-m", "model_initializer"],
                        )
                    ]
                ),
                status=models.IoK8sApiCoreV1PodStatus(phase="Running"),
            ),
            # Training node pod
            models.IoK8sApiCoreV1Pod(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="node-0-pod",
                    namespace=DEFAULT_NAMESPACE,
                    labels={
                        constants.JOBSET_NAME_LABEL: BASIC_TRAIN_JOB_NAME,
                        constants.JOBSET_RJOB_NAME_LABEL: constants.NODE,
                        constants.JOB_INDEX_LABEL: "0",
                    },
                ),
                spec=models.IoK8sApiCoreV1PodSpec(
                    containers=[
                        models.IoK8sApiCoreV1Container(
                            name=constants.NODE,
                            image="trainer:latest",
                            command=["python", "-m", "trainer"],
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


def get_custom_trainer(
    env: Optional[list[models.IoK8sApiCoreV1EnvVar]] = None,
) -> models.TrainerV1alpha1Trainer:
    """
    Get the custom trainer for the TrainJob.
    """

    return models.TrainerV1alpha1Trainer(
        command=[
            "bash",
            "-c",
            '\nif ! [ -x "$(command -v pip)" ]; then\n    python -m ensurepip '
            "|| python -m ensurepip --user || apt-get install python-pip"
            "\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet"
            "         --no-warn-script-location --index-url https://pypi.org/simple "
            "torch numpy \n\nread -r -d '' SCRIPT << EOM\n\nfunc=lambda: "
            'print("Hello World"),\n\n<lambda>('
            "{'learning_rate': 0.001, 'batch_size': 32})\n\nEOM\nprintf \"%s\" "
            '"$SCRIPT" > "backend_test.py"\ntorchrun "backend_test.py"',
        ],
        numNodes=2,
        env=env,
    )


def get_builtin_trainer() -> models.TrainerV1alpha1Trainer:
    """
    Get the builtin trainer for the TrainJob.
    """
    return models.TrainerV1alpha1Trainer(
        args=["batch_size=2", "epochs=2", "loss=Loss.CEWithChunkedOutputLoss"],
        command=["tune", "run"],
        numNodes=2,
    )


def get_train_job(
    runtime_name: str,
    train_job_name: str = BASIC_TRAIN_JOB_NAME,
    train_job_trainer: Optional[models.TrainerV1alpha1Trainer] = None,
) -> models.TrainerV1alpha1TrainJob:
    """
    Create a mock TrainJob object with optional trainer configurations.
    """
    train_job = models.TrainerV1alpha1TrainJob(
        apiVersion=constants.API_VERSION,
        kind=constants.TRAINJOB_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(name=train_job_name),
        spec=models.TrainerV1alpha1TrainJobSpec(
            runtimeRef=models.TrainerV1alpha1RuntimeRef(name=runtime_name),
            trainer=train_job_trainer,
        ),
    )

    return train_job


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
    """Return a mocked TrainJob object."""
    mock_thread = Mock()
    if args[2] == TIMEOUT or args[4] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME or args[4] == RUNTIME:
        raise RuntimeError()
    if args[3] == TRAIN_JOBS:  # TODO: review this.
        mock_thread.get.return_value = add_status(create_train_job(train_job_name=args[4]))

    return mock_thread


def add_status(
    train_job: models.TrainerV1alpha1TrainJob,
) -> models.TrainerV1alpha1TrainJob:
    """
    Add status information to the train job.
    """
    # Set initial status to Created
    status = models.TrainerV1alpha1TrainJobStatus(
        conditions=[
            models.IoK8sApimachineryPkgApisMetaV1Condition(
                type="Complete",
                status="True",
                lastTransitionTime=datetime.datetime.now(),
                reason="JobCompleted",
                message="Job completed successfully",
            )
        ]
    )
    train_job.status = status
    return train_job


def list_namespaced_custom_object_response(*args, **kwargs):
    """Return a list of mocked TrainJob objects."""
    mock_thread = Mock()
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME:
        raise RuntimeError()
    if args[3] == constants.TRAINJOB_PLURAL:
        items = [
            add_status(create_train_job(train_job_name="basic-job-1")),
            add_status(create_train_job(train_job_name="basic-job-2")),
        ]
        mock_thread.get.return_value = normalize_model(
            models.TrainerV1alpha1TrainJobList(items=items),
            models.TrainerV1alpha1TrainJobList,
        )

    return mock_thread


def list_cluster_custom_object(*args, **kwargs):
    """Return a generic mocked response for cluster object listing."""
    mock_thread = Mock()
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME:
        raise RuntimeError()
    if args[2] == constants.CLUSTER_TRAINING_RUNTIME_PLURAL:
        items = [
            create_cluster_training_runtime(name="runtime-1"),
            create_cluster_training_runtime(name="runtime-2"),
        ]
        mock_thread.get.return_value = normalize_model(
            models.TrainerV1alpha1ClusterTrainingRuntimeList(items=items),
            models.TrainerV1alpha1ClusterTrainingRuntimeList,
        )

    return mock_thread


def mock_read_namespaced_pod_log(*args, **kwargs):
    """Simulate log retrieval from a pod."""
    if kwargs.get("namespace") == FAIL_LOGS:
        raise Exception("Failed to read logs")
    return "test log content"


def mock_watch(*args, **kwargs):
    """Simulate watch event"""
    if kwargs.get("timeout_seconds") == 1:
        raise TimeoutError("Watch timeout")

    events = [
        {
            "type": "MODIFIED",
            "object": {
                "metadata": {
                    "name": f"{BASIC_TRAIN_JOB_NAME}-node-0",
                    "labels": {
                        constants.JOBSET_NAME_LABEL: BASIC_TRAIN_JOB_NAME,
                        constants.JOBSET_RJOB_NAME_LABEL: constants.NODE,
                        constants.JOB_INDEX_LABEL: "0",
                    },
                },
                "spec": {"containers": [{"name": constants.NODE}]},
                "status": {"phase": "Running"},
            },
        }
    ]

    return iter(events)


def normalize_model(model_obj, model_class):
    # Simulate real api behavior
    # Converts model to raw dictionary, like a real API response
    # Parses dict and ensures correct model instantiation and type validation
    return model_class.from_dict(model_obj.to_dict())


# --------------------------
# Object Creators
# --------------------------


def create_train_job(
    train_job_name: str = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11],
    namespace: str = "default",
    image: str = "pytorch/pytorch:latest",
    initializer: Optional[types.Initializer] = None,
    command: Optional[list] = None,
    args: Optional[list] = None,
) -> models.TrainerV1alpha1TrainJob:
    """Create a mock TrainJob object."""
    return models.TrainerV1alpha1TrainJob(
        apiVersion=constants.API_VERSION,
        kind=constants.TRAINJOB_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=train_job_name,
            namespace=namespace,
            creationTimestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        ),
        spec=models.TrainerV1alpha1TrainJobSpec(
            runtimeRef=models.TrainerV1alpha1RuntimeRef(name=TORCH_RUNTIME),
            trainer=None,
            initializer=(
                models.TrainerV1alpha1Initializer(
                    dataset=utils.get_dataset_initializer(initializer.dataset),
                    model=utils.get_model_initializer(initializer.model),
                )
                if initializer
                else None
            ),
        ),
    )


def create_cluster_training_runtime(
    name: str,
    namespace: str = "default",
) -> models.TrainerV1alpha1ClusterTrainingRuntime:
    """Create a mock ClusterTrainingRuntime object."""

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
                spec=models.JobsetV1alpha2JobSetSpec(replicatedJobs=[get_replicated_job()]),
            ),
        ),
    )


def get_replicated_job() -> models.JobsetV1alpha2ReplicatedJob:
    return models.JobsetV1alpha2ReplicatedJob(
        name="node",
        replicas=1,
        template=models.IoK8sApiBatchV1JobTemplateSpec(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                labels={"trainer.kubeflow.org/trainjob-ancestor-step": "trainer"}
            ),
            spec=models.IoK8sApiBatchV1JobSpec(
                template=models.IoK8sApiCoreV1PodTemplateSpec(
                    spec=models.IoK8sApiCoreV1PodSpec(containers=[get_container()])
                )
            ),
        ),
    )


def get_container() -> models.IoK8sApiCoreV1Container:
    return models.IoK8sApiCoreV1Container(
        name="node",
        image="image",
        command=["echo", "Hello World"],
        resources=get_resource_requirements(),
    )


def create_runtime_type(
    name: str,
) -> types.Runtime:
    """Create a mock Runtime object for testing."""
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=name,
        num_nodes=2,
        device="gpu",
        device_count=RUNTIME_DEVICES,
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return types.Runtime(
        name=name,
        pretrained_model=None,
        trainer=trainer,
    )


def get_train_job_data_type(
    runtime_name: str,
    train_job_name: str,
) -> types.TrainJob:
    """Create a mock TrainJob object with the expected structure for testing."""

    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=runtime_name,
        device="gpu",
        device_count=RUNTIME_DEVICES,
        num_nodes=2,
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return types.TrainJob(
        name=train_job_name,
        creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        runtime=types.Runtime(
            name=runtime_name,
            pretrained_model=None,
            trainer=trainer,
        ),
        steps=[
            types.Step(
                name="dataset-initializer",
                status="Running",
                pod_name="dataset-initializer-pod",
                device="Unknown",
                device_count="Unknown",
            ),
            types.Step(
                name="model-initializer",
                status="Running",
                pod_name="model-initializer-pod",
                device="Unknown",
                device_count="Unknown",
            ),
            types.Step(
                name="node-0",
                status="Running",
                pod_name="node-0-pod",
                device="gpu",
                device_count="1",
            ),
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
            config={"name": TORCH_RUNTIME},
            expected_output=create_runtime_type(name=TORCH_RUNTIME),
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
    """Test KubernetesBackend.get_runtime with basic success path."""
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
    """Test KubernetesBackend.list_runtimes with basic success path."""
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
            name="valid flow with custom trainer runtime",
            expected_status=SUCCESS,
            config={"runtime": create_runtime_type(name=TORCH_RUNTIME)},
        ),
        TestCase(
            name="value error with builtin trainer runtime",
            expected_status=FAILED,
            config={
                "runtime": types.Runtime(
                    name="torchtune-runtime",
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.BUILTIN_TRAINER,
                        framework="torchtune",
                        num_nodes=1,
                        device="cpu",
                        device_count="1",
                    ),
                )
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_runtime_packages(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_runtime_packages with basic success path."""
    print("Executing test:", test_case.name)

    try:
        kubernetes_backend.get_runtime_packages(**test_case.config)
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
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=BASIC_TRAIN_JOB_NAME,
            ),
        ),
        TestCase(
            name="valid flow with built in trainer",
            expected_status=SUCCESS,
            config={
                "trainer": types.BuiltinTrainer(
                    config=types.TorchTuneConfig(
                        num_nodes=2,
                        batch_size=2,
                        epochs=2,
                        loss=types.Loss.CEWithChunkedOutputLoss,
                    )
                ),
                "runtime": TORCH_TUNE_RUNTIME,
            },
            expected_output=get_train_job(
                runtime_name=TORCH_TUNE_RUNTIME,
                train_job_name=TRAIN_JOB_WITH_BUILT_IN_TRAINER,
                train_job_trainer=get_builtin_trainer(),
            ),
        ),
        TestCase(
            name="valid flow with custom trainer",
            expected_status=SUCCESS,
            config={
                "trainer": types.CustomTrainer(
                    func=lambda: print("Hello World"),
                    func_args={"learning_rate": 0.001, "batch_size": 32},
                    packages_to_install=["torch", "numpy"],
                    pip_index_url=constants.DEFAULT_PIP_INDEX_URL,
                    num_nodes=2,
                )
            },
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=TRAIN_JOB_WITH_CUSTOM_TRAINER,
                train_job_trainer=get_custom_trainer(),
            ),
        ),
        TestCase(
            name="valid flow with custom trainer and env vars",
            expected_status=SUCCESS,
            config={
                "trainer": types.CustomTrainer(
                    func=lambda: print("Hello World"),
                    func_args={"learning_rate": 0.001, "batch_size": 32},
                    packages_to_install=["torch", "numpy"],
                    pip_index_url=constants.DEFAULT_PIP_INDEX_URL,
                    num_nodes=2,
                    env={
                        "TEST_ENV": "test_value",
                        "ANOTHER_ENV": "another_value",
                    },
                )
            },
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=TRAIN_JOB_WITH_CUSTOM_TRAINER,
                train_job_trainer=get_custom_trainer(
                    env=[
                        models.IoK8sApiCoreV1EnvVar(name="TEST_ENV", value="test_value"),
                        models.IoK8sApiCoreV1EnvVar(name="ANOTHER_ENV", value="another_value"),
                    ],
                ),
            ),
        ),
        TestCase(
            name="timeout error when deleting job",
            expected_status=FAILED,
            config={
                "namespace": TIMEOUT,
            },
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when deleting job",
            expected_status=FAILED,
            config={
                "namespace": RUNTIME,
            },
            expected_error=RuntimeError,
        ),
        TestCase(
            name="value error when runtime doesn't support CustomTrainer",
            expected_status=FAILED,
            config={
                "trainer": types.CustomTrainer(
                    func=lambda: print("Hello World"),
                    num_nodes=2,
                ),
                "runtime": TORCH_TUNE_RUNTIME,
            },
            expected_error=ValueError,
        ),
    ],
)
def test_train(kubernetes_backend, test_case):
    """Test KubernetesBackend.train with basic success path."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        runtime = kubernetes_backend.get_runtime(test_case.config.get("runtime", TORCH_RUNTIME))

        train_job_name = kubernetes_backend.train(
            runtime=runtime, trainer=test_case.config.get("trainer", None)
        )

        assert test_case.expected_status == SUCCESS

        # This is to get around the fact that the train job name is dynamically generated
        # In the future name generation may be more deterministic, and we can revisit this approach
        expected_output = test_case.expected_output
        expected_output.metadata.name = train_job_name

        kubernetes_backend.custom_api.create_namespaced_custom_object.assert_called_with(
            constants.GROUP,
            constants.VERSION,
            DEFAULT_NAMESPACE,
            constants.TRAINJOB_PLURAL,
            expected_output.to_dict(),
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
            config={"name": BASIC_TRAIN_JOB_NAME},
            expected_output=get_train_job_data_type(
                runtime_name=TORCH_RUNTIME,
                train_job_name=BASIC_TRAIN_JOB_NAME,
            ),
        ),
        TestCase(
            name="timeout error when getting job",
            expected_status=FAILED,
            config={"name": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting job",
            expected_status=FAILED,
            config={"name": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_job(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_job with basic success path."""
    print("Executing test:", test_case.name)
    try:
        job = kubernetes_backend.get_job(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert asdict(job) == asdict(test_case.expected_output)

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
                get_train_job_data_type(
                    runtime_name=TORCH_RUNTIME,
                    train_job_name="basic-job-1",
                ),
                get_train_job_data_type(
                    runtime_name=TORCH_RUNTIME,
                    train_job_name="basic-job-2",
                ),
            ],
        ),
        TestCase(
            name="timeout error when listing jobs",
            expected_status=FAILED,
            config={"namespace": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when listing jobs",
            expected_status=FAILED,
            config={"namespace": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_list_jobs(kubernetes_backend, test_case):
    """Test KubernetesBackend.list_jobs with basic success path."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        jobs = kubernetes_backend.list_jobs()

        assert test_case.expected_status == SUCCESS
        assert isinstance(jobs, list)
        assert len(jobs) == 2
        assert [asdict(j) for j in jobs] == [asdict(r) for r in test_case.expected_output]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
            expected_output=["test log content"],
        ),
        TestCase(
            name="runtime error when getting logs",
            expected_status=FAILED,
            config={"name": BASIC_TRAIN_JOB_NAME, "namespace": FAIL_LOGS},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_job_logs(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_job_logs with basic success path."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        logs = kubernetes_backend.get_job_logs(test_case.config.get("name"))
        # Convert iterator to list for comparison.
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
            name="wait for complete status (default)",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
            expected_output=get_train_job_data_type(
                runtime_name=TORCH_RUNTIME,
                train_job_name=BASIC_TRAIN_JOB_NAME,
            ),
        ),
        TestCase(
            name="wait for multiple statuses",
            expected_status=SUCCESS,
            config={
                "name": BASIC_TRAIN_JOB_NAME,
                "status": {constants.TRAINJOB_RUNNING, constants.TRAINJOB_COMPLETE},
            },
            expected_output=get_train_job_data_type(
                runtime_name=TORCH_RUNTIME,
                train_job_name=BASIC_TRAIN_JOB_NAME,
            ),
        ),
        TestCase(
            name="invalid status set error",
            expected_status=FAILED,
            config={
                "name": BASIC_TRAIN_JOB_NAME,
                "status": {"InvalidStatus"},
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="polling interval is more than timeout error",
            expected_status=FAILED,
            config={
                "name": BASIC_TRAIN_JOB_NAME,
                "timeout": 1,
                "polling_interval": 2,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="job failed when not expected",
            expected_status=FAILED,
            config={
                "name": "failed-job",
                "status": {constants.TRAINJOB_RUNNING},
            },
            expected_error=RuntimeError,
        ),
        TestCase(
            name="timeout error to wait for failed status",
            expected_status=FAILED,
            config={
                "name": BASIC_TRAIN_JOB_NAME,
                "status": {constants.TRAINJOB_FAILED},
                "polling_interval": 1,
                "timeout": 2,
            },
            expected_error=TimeoutError,
        ),
    ],
)
def test_wait_for_job_status(kubernetes_backend, test_case):
    """Test KubernetesBackend.wait_for_job_status with various scenarios."""
    print("Executing test:", test_case.name)

    original_get_job = kubernetes_backend.get_job

    # TrainJob has unexpected failed status.
    def mock_get_job(name):
        job = original_get_job(name)
        if test_case.config.get("name") == "failed-job":
            job.status = constants.TRAINJOB_FAILED
        return job

    kubernetes_backend.get_job = mock_get_job

    try:
        job = kubernetes_backend.wait_for_job_status(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(job, types.TrainJob)
        # Job status should be in the expected set.
        assert job.status in test_case.config.get("status", {constants.TRAINJOB_COMPLETE})

    except Exception as e:
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
            expected_output=None,
        ),
        TestCase(
            name="timeout error when deleting job",
            expected_status=FAILED,
            config={"namespace": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when deleting job",
            expected_status=FAILED,
            config={"namespace": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_delete_job(kubernetes_backend, test_case):
    """Test KubernetesBackend.delete_job with basic success path."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        kubernetes_backend.delete_job(test_case.config.get("name"))
        assert test_case.expected_status == SUCCESS

        kubernetes_backend.custom_api.delete_namespaced_custom_object.assert_called_with(
            constants.GROUP,
            constants.VERSION,
            test_case.config.get("namespace", DEFAULT_NAMESPACE),
            constants.TRAINJOB_PLURAL,
            name=test_case.config.get("name"),
        )

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
