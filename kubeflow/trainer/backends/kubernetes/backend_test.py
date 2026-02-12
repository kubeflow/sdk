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

"""Unit tests for the KubernetesBackend class in the Kubeflow Trainer SDK.

This module uses pytest and unittest.mock to simulate Kubernetes API interactions.
It tests KubernetesBackend's behavior across job listing, resource creation etc.
"""

from dataclasses import asdict
import datetime
import logging
import multiprocessing
import random
import string
from typing import Optional
from unittest.mock import Mock, patch
import uuid

from kubeflow_trainer_api import models
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
import kubeflow.trainer.backends.kubernetes.utils as utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.options import (
    Annotations,
    Labels,
    SpecAnnotations,
    SpecLabels,
)
from kubeflow.trainer.test.common import (
    DEFAULT_NAMESPACE,
    FAILED,
    RUNTIME,
    SUCCESS,
    TIMEOUT,
    TestCase,
)
from kubeflow.trainer.types import types

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
                list_namespaced_event=Mock(side_effect=mock_list_namespaced_event),
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
    pip_index_urls: Optional[list[str]] = constants.DEFAULT_PIP_INDEX_URLS,
    packages_to_install: list[str] = ["torch", "numpy"],
    image: Optional[str] = None,
) -> models.TrainerV1alpha1Trainer:
    """
    Get the custom trainer for the TrainJob.
    """
    pip_command = [f"--index-url {pip_index_urls[0]}"]
    pip_command.extend([f"--extra-index-url {repo}" for repo in pip_index_urls[1:]])
    pip_command = " ".join(pip_command)

    packages_command = " ".join(packages_to_install)
    return models.TrainerV1alpha1Trainer(
        command=[
            "bash",
            "-c",
            '\nif ! [ -x "$(command -v pip)" ]; then\n    python -m ensurepip '
            "|| python -m ensurepip --user || apt-get install python-pip"
            "\nfi\n\n"
            "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet"
            f" --no-warn-script-location {pip_command} --user {packages_command}"
            " ||\nPIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet"
            f" --no-warn-script-location {pip_command} {packages_command}"
            "\n\nread -r -d '' SCRIPT << EOM\n\nfunc=lambda: "
            'print("Hello World"),\n\n<lambda>(**'
            "{'learning_rate': 0.001, 'batch_size': 32})\n\nEOM\nprintf \"%s\" "
            '"$SCRIPT" > "backend_test.py"\ntorchrun "backend_test.py"',
        ],
        numNodes=2,
        env=env,
        image=image,
    )


def get_custom_trainer_container(
    image: str,
    num_nodes: int,
    resources_per_node: models.IoK8sApiCoreV1ResourceRequirements,
    env: list[models.IoK8sApiCoreV1EnvVar],
) -> models.TrainerV1alpha1Trainer:
    """
    Get the custom trainer container for the TrainJob.
    """

    return models.TrainerV1alpha1Trainer(
        image=image,
        numNodes=num_nodes,
        resourcesPerNode=resources_per_node,
        env=env,
    )


def _build_core_api_mock(
    config_map_data: Optional[dict] = None,
    error: Optional[Exception] = None,
):
    """Helper to construct a CoreV1Api mock for version checks."""

    core_api = Mock()

    if error is not None:
        core_api.read_namespaced_config_map.side_effect = error
    else:
        core_api.read_namespaced_config_map.return_value = Mock(data=config_map_data)

    return core_api


def get_builtin_trainer(
    args: list[str],
) -> models.TrainerV1alpha1Trainer:
    """
    Get the builtin trainer for the TrainJob.
    """
    return models.TrainerV1alpha1Trainer(
        args=args,
        command=["tune", "run"],
        numNodes=2,
    )


def get_train_job(
    runtime_name: str,
    train_job_name: str = BASIC_TRAIN_JOB_NAME,
    train_job_trainer: Optional[models.TrainerV1alpha1Trainer] = None,
    labels: Optional[dict[str, str]] = None,
    annotations: Optional[dict[str, str]] = None,
    spec_labels: Optional[dict[str, str]] = None,
    spec_annotations: Optional[dict[str, str]] = None,
) -> models.TrainerV1alpha1TrainJob:
    """
    Create a mock TrainJob object with optional trainer configurations.
    """
    train_job = models.TrainerV1alpha1TrainJob(
        apiVersion=constants.API_VERSION,
        kind=constants.TRAINJOB_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=train_job_name,
            labels=labels,
            annotations=annotations,
        ),
        spec=models.TrainerV1alpha1TrainJobSpec(
            runtimeRef=models.TrainerV1alpha1RuntimeRef(name=runtime_name),
            trainer=train_job_trainer,
            labels=spec_labels,
            annotations=spec_annotations,
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


def mock_list_namespaced_event(*args, **kwargs):
    """Simulate event listing from namespace."""
    namespace = kwargs.get("namespace")

    # Errors occur at call time, not during .get()
    if namespace == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if namespace == RUNTIME:
        raise RuntimeError()

    mock_thread = Mock()
    mock_thread.get.return_value = models.IoK8sApiCoreV1EventList(
        items=[
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="test-event-1",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind=constants.TRAINJOB_KIND,
                    name=BASIC_TRAIN_JOB_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="TrainJob created successfully",
                reason="Created",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
            ),
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="test-event-2",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind="Pod",
                    name="node-0-pod",
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Pod scheduled successfully",
                reason="Scheduled",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 31, 0),
            ),
        ]
    )
    return mock_thread


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
        image="example.com/test-runtime",
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
        image="example.com/test-runtime",
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return types.Runtime(
        name=name,
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
        image="example.com/test-runtime",
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return types.TrainJob(
        name=train_job_name,
        creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        runtime=types.Runtime(
            name=runtime_name,
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


def _run_verify_backend_with_core_api(core_api: Mock) -> tuple[list[str], int]:
    """Helper to run verify_backend and capture warning logs."""

    logger_name = "kubeflow.trainer.backends.kubernetes.backend"
    logger_obj = logging.getLogger(logger_name)

    class _ListHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            self.records.append(record)

    handler = _ListHandler()
    logger_obj.addHandler(handler)
    previous_level = logger_obj.level
    logger_obj.setLevel(logging.WARNING)

    try:
        with (
            patch("kubernetes.config.load_kube_config", return_value=None),
            patch("kubeflow.common.utils.is_running_in_k8s", return_value=False),
            patch("kubernetes.client.ApiClient", return_value=Mock()),
            patch("kubernetes.client.CustomObjectsApi", return_value=Mock()),
            patch("kubernetes.client.CoreV1Api", return_value=core_api),
        ):
            KubernetesBackend(KubernetesBackendConfig())
    finally:
        logger_obj.removeHandler(handler)
        logger_obj.setLevel(previous_level)

    messages = [record.getMessage() for record in handler.records]
    call_count = core_api.read_namespaced_config_map.call_count
    return messages, call_count


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="version metadata present",
            expected_status=SUCCESS,
            config={
                "core_api": _build_core_api_mock({"kubeflow_trainer_version": "1.2.3"}),
                "expect_warning": False,
            },
        ),
        TestCase(
            name="ConfigMap read error logs warning",
            expected_status=SUCCESS,
            config={
                "core_api": _build_core_api_mock(None, Exception("ConfigMap not found")),
                "expect_warning": True,
                "must_contain": [
                    "Trainer control-plane version info is not available",
                    "kubeflow-trainer-public",
                    "ConfigMap not found",
                ],
            },
        ),
    ],
)
def test_verify_backend(test_case):
    """Test KubernetesBackend.verify_backend across version metadata scenarios."""

    print("Executing test:", test_case.name)

    core_api: Mock = test_case.config["core_api"]
    expect_warning: bool = test_case.config.get("expect_warning", False)
    must_contain: list[str] = test_case.config.get("must_contain", [])

    warnings, call_count = _run_verify_backend_with_core_api(core_api)
    combined = "\n".join(warnings)

    assert call_count >= 1

    if expect_warning:
        assert warnings, "Expected warning logs but found none"
        for text in must_contain:
            assert text in combined
    else:
        assert "Trainer control-plane version info is not available" not in combined

    print("test execution complete")


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
                        image="example.com/image",
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
                train_job_trainer=get_builtin_trainer(
                    args=["batch_size=2", "epochs=2", "loss=Loss.CEWithChunkedOutputLoss"],
                ),
            ),
        ),
        TestCase(
            name="valid flow with built in trainer and lora config",
            expected_status=SUCCESS,
            config={
                "trainer": types.BuiltinTrainer(
                    config=types.TorchTuneConfig(
                        num_nodes=2,
                        peft_config=types.LoraConfig(
                            apply_lora_to_mlp=True,
                            lora_rank=8,
                            lora_alpha=16,
                            lora_dropout=0.1,
                        ),
                    ),
                ),
                "runtime": TORCH_TUNE_RUNTIME,
            },
            expected_output=get_train_job(
                runtime_name=TORCH_TUNE_RUNTIME,
                train_job_name=TRAIN_JOB_WITH_BUILT_IN_TRAINER,
                train_job_trainer=get_builtin_trainer(
                    args=[
                        "model.apply_lora_to_mlp=True",
                        "model.lora_rank=8",
                        "model.lora_alpha=16",
                        "model.lora_dropout=0.1",
                        "model.lora_attn_modules=[q_proj,v_proj,output_proj]",
                    ],
                ),
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
                    pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
                    num_nodes=2,
                )
            },
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=TRAIN_JOB_WITH_CUSTOM_TRAINER,
                train_job_trainer=get_custom_trainer(
                    pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
                    packages_to_install=["torch", "numpy"],
                ),
            ),
        ),
        TestCase(
            name="valid flow with custom trainer that has env and image",
            expected_status=SUCCESS,
            config={
                "trainer": types.CustomTrainer(
                    func=lambda: print("Hello World"),
                    func_args={"learning_rate": 0.001, "batch_size": 32},
                    packages_to_install=["torch", "numpy"],
                    pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
                    num_nodes=2,
                    env={
                        "TEST_ENV": "test_value",
                        "ANOTHER_ENV": "another_value",
                    },
                    image="my-custom-image",
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
                    pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
                    packages_to_install=["torch", "numpy"],
                    image="my-custom-image",
                ),
            ),
        ),
        TestCase(
            name="valid flow with custom trainer container",
            expected_status=SUCCESS,
            config={
                "trainer": types.CustomTrainerContainer(
                    image="example.com/my-image",
                    num_nodes=2,
                    resources_per_node={"cpu": 5, "gpu": 3},
                    env={
                        "TEST_ENV": "test_value",
                        "ANOTHER_ENV": "another_value",
                    },
                )
            },
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=TRAIN_JOB_WITH_CUSTOM_TRAINER,
                train_job_trainer=get_custom_trainer_container(
                    image="example.com/my-image",
                    num_nodes=2,
                    resources_per_node=models.IoK8sApiCoreV1ResourceRequirements(
                        requests={
                            "cpu": models.IoK8sApimachineryPkgApiResourceQuantity(5),
                            "nvidia.com/gpu": models.IoK8sApimachineryPkgApiResourceQuantity(3),
                        },
                        limits={
                            "cpu": models.IoK8sApimachineryPkgApiResourceQuantity(5),
                            "nvidia.com/gpu": models.IoK8sApimachineryPkgApiResourceQuantity(3),
                        },
                    ),
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
        TestCase(
            name="train with metadata labels and annotations",
            expected_status=SUCCESS,
            config={
                "options": [
                    Labels({"team": "ml-platform"}),
                    Annotations({"created-by": "sdk"}),
                ],
            },
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=BASIC_TRAIN_JOB_NAME,
                labels={"team": "ml-platform"},
                annotations={"created-by": "sdk"},
            ),
        ),
        TestCase(
            name="train with spec labels and annotations",
            expected_status=SUCCESS,
            config={
                "options": [
                    SpecLabels({"app": "training", "version": "v1.0"}),
                    SpecAnnotations({"prometheus.io/scrape": "true"}),
                ],
            },
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=BASIC_TRAIN_JOB_NAME,
                spec_labels={"app": "training", "version": "v1.0"},
                spec_annotations={"prometheus.io/scrape": "true"},
            ),
        ),
        TestCase(
            name="train with both metadata and spec labels/annotations",
            expected_status=SUCCESS,
            config={
                "options": [
                    Labels({"owner": "ml-team"}),
                    Annotations({"description": "Fine-tuning job"}),
                    SpecLabels({"app": "training", "version": "v1.0"}),
                    SpecAnnotations({"prometheus.io/scrape": "true"}),
                ],
            },
            expected_output=get_train_job(
                runtime_name=TORCH_RUNTIME,
                train_job_name=BASIC_TRAIN_JOB_NAME,
                labels={"owner": "ml-team"},
                annotations={"description": "Fine-tuning job"},
                spec_labels={"app": "training", "version": "v1.0"},
                spec_annotations={"prometheus.io/scrape": "true"},
            ),
        ),
    ],
)
def test_train(kubernetes_backend, test_case):
    """Test KubernetesBackend.train with basic success path."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        runtime = kubernetes_backend.get_runtime(test_case.config.get("runtime", TORCH_RUNTIME))

        options = test_case.config.get("options", [])

        train_job_name = kubernetes_backend.train(
            runtime=runtime,
            trainer=test_case.config.get("trainer", None),
            options=options,
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

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get job events with valid trainjob",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
            expected_output=[
                types.Event(
                    involved_object_kind=constants.TRAINJOB_KIND,
                    involved_object_name=BASIC_TRAIN_JOB_NAME,
                    message="TrainJob created successfully",
                    reason="Created",
                    event_time=datetime.datetime(2025, 6, 1, 10, 30, 0),
                ),
                types.Event(
                    involved_object_kind="Pod",
                    involved_object_name="node-0-pod",
                    message="Pod scheduled successfully",
                    reason="Scheduled",
                    event_time=datetime.datetime(2025, 6, 1, 10, 31, 0),
                ),
            ],
        ),
        TestCase(
            name="timeout error when getting job events",
            expected_status=FAILED,
            config={"namespace": TIMEOUT, "name": BASIC_TRAIN_JOB_NAME},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting job events",
            expected_status=FAILED,
            config={"namespace": RUNTIME, "name": BASIC_TRAIN_JOB_NAME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_job_events(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_job_events with various scenarios."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        events = kubernetes_backend.get_job_events(test_case.config.get("name"))

        assert test_case.expected_status == SUCCESS
        assert isinstance(events, list)
        assert len(events) == len(test_case.expected_output)
        assert [asdict(e) for e in events] == [asdict(e) for e in test_case.expected_output]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
