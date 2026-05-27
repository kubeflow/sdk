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
It tests KubernetesBackend's behavior across job listing, resource creation, deletion,
log retrieval, event filtering, and status waiting.
"""

from dataclasses import asdict
import datetime
import multiprocessing
from typing import Any, TypeVar
from unittest.mock import Mock, patch

from kubeflow_katib_api import models
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.optimizer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Metric,
    Objective,
    OptimizationJob,
    Result,
    Trial,
    TrialConfig,
)
from kubeflow.optimizer.types.search_types import (
    ContinuousSearchSpace,
    Distribution,
    Search,
)
import kubeflow.trainer.constants.constants as trainer_constants
from kubeflow.trainer.test.common import (
    DEFAULT_NAMESPACE,
    FAILED,
    RUNTIME,
    SUCCESS,
    TIMEOUT,
    TestCase,
)
from kubeflow.trainer.types.types import (
    CustomTrainer,
    Event,
    Runtime,
    RuntimeTrainer,
    Step,
    TrainerType,
    TrainJob,
    TrainJobTemplate,
)

T = TypeVar("T")

BASIC_OPTIMIZATION_JOB_NAME = "basic-opt-job"
BASIC_TRIAL_NAME = "basic-trial"
BASIC_TRIAL_NAME_2 = "basic-trial-2"

# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def optimizer_backend():
    """Provide an optimizer KubernetesBackend with mocked Kubernetes APIs."""
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(side_effect=conditional_error_handler),
                delete_namespaced_custom_object=Mock(side_effect=conditional_error_handler),
                get_namespaced_custom_object=Mock(
                    side_effect=get_namespaced_custom_object_response
                ),
                list_namespaced_custom_object=Mock(
                    side_effect=list_namespaced_custom_object_response
                ),
            ),
        ),
        patch(
            "kubernetes.client.CoreV1Api",
            return_value=Mock(
                list_namespaced_event=Mock(side_effect=mock_list_namespaced_event),
            ),
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.verify_backend",
            return_value=None,
        ),
    ):
        backend = KubernetesBackend(KubernetesBackendConfig())
        backend.trainer_backend._get_trainjob_spec = Mock(
            return_value=Mock(to_dict=Mock(return_value={}))
        )
        backend.trainer_backend.get_job = Mock(side_effect=mock_trainer_get_job)
        backend.trainer_backend._read_pod_logs = Mock(return_value=iter(["test log content"]))
        yield backend


# --------------------------
# Mock Handlers
# --------------------------


def conditional_error_handler(*args: Any, **kwargs: Any) -> None:
    """Raise simulated errors based on namespace.

    Args:
        args: Positional args from the K8s API call.
            args[2] is the namespace for create/delete/list_namespaced_custom_object.
    """
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[2] == RUNTIME:
        raise RuntimeError()


def get_namespaced_custom_object_response(*args, **kwargs):
    """Return a mocked Experiment object.

    Args:
        args: Positional args from the K8s API call.
            args[4] is the resource name for get_namespaced_custom_object.
    """
    mock_thread = Mock()
    if args[4] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[4] == RUNTIME:
        raise RuntimeError()
    if args[3] == constants.EXPERIMENT_PLURAL:
        experiment = create_experiment_cr(name=args[4])
        mock_thread.get.return_value = normalize_model(experiment, models.V1beta1Experiment)
    return mock_thread


def list_namespaced_custom_object_response(*args, **kwargs):
    """Return a list of mocked Experiment or Trial objects.

    Args:
        args: Positional args from the K8s API call.
            args[2] is the namespace for list_namespaced_custom_object.
    """
    mock_thread = Mock()
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME:
        raise RuntimeError()
    if args[3] == constants.EXPERIMENT_PLURAL:
        items = [
            create_experiment_cr(name="opt-job-1"),
            create_experiment_cr(name="opt-job-2"),
        ]
        mock_thread.get.return_value = normalize_model(
            models.V1beta1ExperimentList(items=items),
            models.V1beta1ExperimentList,
        )
    elif args[3] == constants.TRIAL_PLURAL:
        items = [
            create_trial_cr(name=BASIC_TRIAL_NAME),
            create_trial_cr(name=BASIC_TRIAL_NAME_2),
        ]
        mock_thread.get.return_value = normalize_model(
            models.V1beta1TrialList(items=items),
            models.V1beta1TrialList,
        )
    return mock_thread


def mock_list_namespaced_event(*args, **kwargs):
    """Simulate event listing from namespace."""
    namespace = kwargs.get("namespace")

    if namespace == TIMEOUT:
        raise multiprocessing.TimeoutError()

    mock_thread = Mock()
    mock_thread.get.return_value = models.IoK8sApiCoreV1EventList(
        items=[
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="test-event-1",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind=constants.EXPERIMENT_KIND,
                    name=BASIC_OPTIMIZATION_JOB_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Experiment created successfully",
                reason="Created",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
            ),
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="test-event-2",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind=constants.TRIAL_KIND,
                    name=BASIC_TRIAL_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Trial started",
                reason="Running",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 31, 0),
            ),
            # Non-matching event (Pod kind) to test filtering
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="test-event-3",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind="Pod",
                    name="some-pod",
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Pod scheduled",
                reason="Scheduled",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 32, 0),
            ),
        ]
    )
    return mock_thread


def mock_trainer_get_job(name: str) -> TrainJob:
    """Return a mock TrainJob for the given trial name."""
    return create_mock_trainjob(name)


def normalize_model(model_obj: Any, model_class: type[T]) -> T:
    """Simulate real API behavior via dict round-trip."""
    return model_class.from_dict(model_obj.to_dict())


# --------------------------
# Object Creators
# --------------------------


def create_experiment_cr(
    name: str = BASIC_OPTIMIZATION_JOB_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    status_conditions: list[models.V1beta1ExperimentCondition] | None = None,
    best_trial: models.V1beta1OptimalTrial | None = None,
) -> models.V1beta1Experiment:
    """Create a mock Experiment CR."""
    status = None
    if status_conditions is not None or best_trial is not None:
        status = models.V1beta1ExperimentStatus(
            conditions=status_conditions,
            currentOptimalTrial=best_trial,
        )

    return models.V1beta1Experiment(
        apiVersion=constants.API_VERSION,
        kind=constants.EXPERIMENT_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
            creationTimestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        ),
        spec=models.V1beta1ExperimentSpec(
            parameters=[
                models.V1beta1ParameterSpec(
                    name="lr",
                    parameterType=constants.DOUBLE_PARAMETER,
                    feasibleSpace=models.V1beta1FeasibleSpace(
                        min="0.001",
                        max="0.1",
                        distribution=Distribution.UNIFORM.value,
                    ),
                ),
            ],
            objective=models.V1beta1ObjectiveSpec(
                objectiveMetricName="loss",
                type="minimize",
            ),
            algorithm=models.V1beta1AlgorithmSpec(
                algorithmName="random",
            ),
            maxTrialCount=10,
            parallelTrialCount=1,
        ),
        status=status,
    )


def create_trial_cr(
    name: str = BASIC_TRIAL_NAME,
    namespace: str = DEFAULT_NAMESPACE,
) -> models.V1beta1Trial:
    """Create a mock Trial CR."""
    return models.V1beta1Trial(
        apiVersion=constants.API_VERSION,
        kind=constants.TRIAL_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
        ),
        spec=models.V1beta1TrialSpec(
            parameterAssignments=[
                models.V1beta1ParameterAssignment(name="lr", value="0.01"),
            ],
            objective=models.V1beta1ObjectiveSpec(
                objectiveMetricName="loss",
                type="minimize",
            ),
            primaryContainerName=trainer_constants.NODE,
        ),
    )


def create_mock_trainjob(name: str) -> TrainJob:
    """Create a mock TrainJob object with the expected structure for testing."""
    trainer = RuntimeTrainer(
        trainer_type=TrainerType.CUSTOM_TRAINER,
        framework="torch",
        num_nodes=1,
        device="gpu",
        device_count="1",
        image="trainer:latest",
    )
    trainer.set_command(trainer_constants.TORCH_COMMAND)
    return TrainJob(
        name=name,
        creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        runtime=Runtime(
            name="torch",
            trainer=trainer,
        ),
        steps=[
            Step(
                name="node-0",
                status="Running",
                pod_name=f"{name}-node-0-pod",
                device="gpu",
                device_count="1",
            ),
        ],
        num_nodes=1,
        status=trainer_constants.TRAINJOB_COMPLETE,
    )


def get_optimization_job_data_type(
    name: str = BASIC_OPTIMIZATION_JOB_NAME,
    status: str = constants.OPTIMIZATION_JOB_CREATED,
    trials: list[Trial] | None = None,
) -> OptimizationJob:
    """Create the expected OptimizationJob output for assertion comparison."""
    if trials is None:
        trials = [
            Trial(
                name=BASIC_TRIAL_NAME,
                parameters={"lr": "0.01"},
                trainjob=create_mock_trainjob(BASIC_TRIAL_NAME),
            ),
            Trial(
                name=BASIC_TRIAL_NAME_2,
                parameters={"lr": "0.01"},
                trainjob=create_mock_trainjob(BASIC_TRIAL_NAME_2),
            ),
        ]

    return OptimizationJob(
        name=name,
        search_space={
            "lr": ContinuousSearchSpace(
                min=0.001,
                max=0.1,
                distribution=Distribution.UNIFORM,
            ),
        },
        objectives=[Objective(metric="loss")],
        algorithm=RandomSearch(),
        trial_config=TrialConfig(
            num_trials=10,
            parallel_trials=1,
            max_failed_trials=None,
        ),
        trials=trials,
        creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
        status=status,
    )


# --------------------------
# Tests
# --------------------------


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
        TestCase(
            name="empty search space raises ValueError",
            expected_status=FAILED,
            config={
                "search_space": {},
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="timeout error when creating job",
            expected_status=FAILED,
            config={
                "namespace": TIMEOUT,
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
            },
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when creating job",
            expected_status=FAILED,
            config={
                "namespace": RUNTIME,
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
            },
            expected_error=RuntimeError,
        ),
    ],
)
def test_optimize(optimizer_backend, test_case):
    """Test KubernetesBackend.optimize with success and error paths."""
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
        optimizer_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        job_name = optimizer_backend.optimize(
            trial_template=trial_template,
            search_space=search_space,
        )

        assert test_case.expected_status == SUCCESS
        assert isinstance(job_name, str) and len(job_name) > 0

        # Verify search_space param_spec.name values are unchanged.
        for param_name, param_spec in search_space.items():
            assert param_spec.name == original_names[param_name]

        # Verify trial_template.trainer.func_args is unchanged.
        assert trial_template.trainer.func_args == original_func_args

        # Verify the Experiment CR was created with expected payload.
        optimizer_backend.custom_api.create_namespaced_custom_object.assert_called_once()
        call_args = optimizer_backend.custom_api.create_namespaced_custom_object.call_args
        payload = call_args[0][4]
        assert payload["kind"] == constants.EXPERIMENT_KIND
        assert len(payload["spec"]["parameters"]) == len(search_space)
        assert payload["spec"]["objective"]["objectiveMetricName"] == "loss"
        assert payload["spec"]["algorithm"] is not None

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_output=get_optimization_job_data_type(
                name=BASIC_OPTIMIZATION_JOB_NAME,
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
def test_get_job(optimizer_backend, test_case):
    """Test KubernetesBackend.get_job with success and error paths."""
    print("Executing test:", test_case.name)
    try:
        job = optimizer_backend.get_job(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert asdict(job) == asdict(test_case.expected_output)

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="succeeded condition maps to OPTIMIZATION_JOB_COMPLETE",
            expected_status=SUCCESS,
            config={
                "name": "succeeded-job",
                "conditions": [
                    models.V1beta1ExperimentCondition(
                        type=constants.EXPERIMENT_SUCCEEDED, status="True"
                    ),
                ],
            },
            expected_output=get_optimization_job_data_type(
                name="succeeded-job",
                status=constants.OPTIMIZATION_JOB_COMPLETE,
            ),
        ),
        TestCase(
            name="failed condition maps to OPTIMIZATION_JOB_FAILED",
            expected_status=SUCCESS,
            config={
                "name": "failed-job",
                "conditions": [
                    models.V1beta1ExperimentCondition(
                        type=constants.OPTIMIZATION_JOB_FAILED, status="True"
                    ),
                ],
            },
            expected_output=get_optimization_job_data_type(
                name="failed-job",
                status=constants.OPTIMIZATION_JOB_FAILED,
            ),
        ),
        TestCase(
            name="running trial maps to OPTIMIZATION_JOB_RUNNING",
            expected_status=SUCCESS,
            config={
                "name": "running-job",
                "conditions": [
                    models.V1beta1ExperimentCondition(type="Running", status="True"),
                ],
                "trial_status": trainer_constants.TRAINJOB_RUNNING,
            },
            expected_output=get_optimization_job_data_type(
                name="running-job",
                status=constants.OPTIMIZATION_JOB_RUNNING,
            ),
        ),
        TestCase(
            name="no conditions maps to OPTIMIZATION_JOB_CREATED",
            expected_status=SUCCESS,
            config={
                "name": "created-job",
                "conditions": None,
            },
            expected_output=get_optimization_job_data_type(
                name="created-job",
                status=constants.OPTIMIZATION_JOB_CREATED,
            ),
        ),
    ],
)
def test_get_job_status_conditions(optimizer_backend, test_case):
    """Test status-mapping logic in __get_optimization_job_from_cr."""
    print("Executing test:", test_case.name)

    job_name = test_case.config["name"]
    conditions = test_case.config.get("conditions")
    trial_status = test_case.config.get("trial_status", trainer_constants.TRAINJOB_COMPLETE)

    experiment = create_experiment_cr(name=job_name, status_conditions=conditions)

    def patched_get(*args, **kwargs):
        mock_thread = Mock()
        if args[3] == constants.EXPERIMENT_PLURAL and args[4] == job_name:
            mock_thread.get.return_value = normalize_model(experiment, models.V1beta1Experiment)
        return mock_thread

    optimizer_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    def patched_trainer_get_job(name: str) -> TrainJob:
        job = create_mock_trainjob(name)
        job.status = trial_status
        return job

    optimizer_backend.trainer_backend.get_job = Mock(side_effect=patched_trainer_get_job)

    job = optimizer_backend.get_job(name=job_name)
    assert job.status == test_case.expected_output.status
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={},
            expected_output=[
                get_optimization_job_data_type(name="opt-job-1"),
                get_optimization_job_data_type(name="opt-job-2"),
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
def test_list_jobs(optimizer_backend, test_case):
    """Test KubernetesBackend.list_jobs with success and error paths."""
    print("Executing test:", test_case.name)
    try:
        optimizer_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        jobs = optimizer_backend.list_jobs()

        assert test_case.expected_status == SUCCESS
        assert isinstance(jobs, list)
        assert len(jobs) == 2
        assert [asdict(j) for j in jobs] == [asdict(r) for r in test_case.expected_output]

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_output=["test log content"],
        ),
        TestCase(
            name="explicit trial_name skips best-trial logic",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "trial_name": BASIC_TRIAL_NAME},
            expected_output=["test log content"],
        ),
        TestCase(
            name="follow=True passed through to read_pod_logs",
            expected_status=SUCCESS,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "trial_name": BASIC_TRIAL_NAME,
                "follow": True,
            },
            expected_output=["test log content"],
        ),
        TestCase(
            name="empty trials returns no logs",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "empty_trials": True},
            expected_output=[],
        ),
        TestCase(
            name="pod_name is None returns no logs",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "pending_pod": True},
            expected_output=[],
        ),
        TestCase(
            name="timeout error when getting job logs",
            expected_status=FAILED,
            config={"name": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting job logs",
            expected_status=FAILED,
            config={"name": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_job_logs(optimizer_backend, test_case):
    """Test KubernetesBackend.get_job_logs with success and error paths."""
    print("Executing test:", test_case.name)

    if test_case.config.get("empty_trials"):
        empty_job = OptimizationJob(
            name=BASIC_OPTIMIZATION_JOB_NAME,
            search_space={},
            objectives=[],
            algorithm=RandomSearch(),
            trial_config=TrialConfig(),
            trials=[],
            creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
            status=constants.OPTIMIZATION_JOB_CREATED,
        )
        optimizer_backend.get_job = Mock(return_value=empty_job)
        optimizer_backend._get_best_trial = Mock(return_value=None)

    if test_case.config.get("pending_pod"):
        pending_trainjob = TrainJob(
            name=BASIC_TRIAL_NAME,
            creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
            runtime=Runtime(
                name="torch",
                trainer=RuntimeTrainer(
                    trainer_type=TrainerType.CUSTOM_TRAINER,
                    framework="torch",
                    num_nodes=1,
                    device="gpu",
                    device_count="1",
                    image="trainer:latest",
                ),
            ),
            steps=[
                Step(
                    name="node-0",
                    status=trainer_constants.POD_PENDING,
                    pod_name=None,
                    device="gpu",
                    device_count="1",
                ),
            ],
            num_nodes=1,
            status=trainer_constants.TRAINJOB_RUNNING,
        )
        optimizer_backend.trainer_backend.get_job = Mock(return_value=pending_trainjob)

    try:
        optimizer_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        logs = optimizer_backend.get_job_logs(
            test_case.config.get("name"),
            trial_name=test_case.config.get("trial_name"),
            follow=test_case.config.get("follow", False),
        )
        logs_list = list(logs)
        assert test_case.expected_status == SUCCESS
        assert logs_list == test_case.expected_output

        if test_case.config.get("follow"):
            optimizer_backend.trainer_backend._read_pod_logs.assert_called_once_with(
                pod_name=f"{BASIC_TRIAL_NAME}-node-0-pod",
                container_name=constants.METRICS_COLLECTOR_CONTAINER,
                follow=True,
            )

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="best trial available",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_best_trial": True},
            expected_output=Result(
                parameters={"lr": "0.01"},
                metrics=[Metric(name="loss", latest="0.1", min="0.1", max="0.1")],
            ),
        ),
        TestCase(
            name="no best trial available",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_best_trial": False},
            expected_output=None,
        ),
        TestCase(
            name="timeout error when getting best results",
            expected_status=FAILED,
            config={"name": TIMEOUT, "has_best_trial": False},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting best results",
            expected_status=FAILED,
            config={"name": RUNTIME, "has_best_trial": False},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_best_results(optimizer_backend, test_case):
    """Test KubernetesBackend.get_best_results with success and error paths."""
    print("Executing test:", test_case.name)

    if test_case.config.get("has_best_trial"):
        best_trial = models.V1beta1OptimalTrial(
            bestTrialName=BASIC_TRIAL_NAME,
            parameterAssignments=[
                models.V1beta1ParameterAssignment(name="lr", value="0.01"),
            ],
            observation=models.V1beta1Observation(
                metrics=[
                    models.V1beta1Metric(name="loss", latest="0.1", min="0.1", max="0.1"),
                ],
            ),
        )
        experiment_with_best = create_experiment_cr(
            name=BASIC_OPTIMIZATION_JOB_NAME,
            best_trial=best_trial,
        )
        original_handler = optimizer_backend.custom_api.get_namespaced_custom_object.side_effect

        def patched_get(*args, **kwargs):
            if args[3] == constants.EXPERIMENT_PLURAL:
                mock_thread = Mock()
                mock_thread.get.return_value = normalize_model(
                    experiment_with_best, models.V1beta1Experiment
                )
                return mock_thread
            return original_handler(*args, **kwargs)

        optimizer_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    try:
        result = optimizer_backend.get_best_results(name=test_case.config["name"])

        assert test_case.expected_status == SUCCESS
        if test_case.expected_output is None:
            assert result is None
        else:
            assert asdict(result) == asdict(test_case.expected_output)

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait for complete status (default)",
            expected_status=SUCCESS,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "_conditions": [
                    models.V1beta1ExperimentCondition(
                        type=constants.EXPERIMENT_SUCCEEDED, status="True"
                    ),
                ],
            },
            expected_output=get_optimization_job_data_type(
                name=BASIC_OPTIMIZATION_JOB_NAME,
                status=constants.OPTIMIZATION_JOB_COMPLETE,
            ),
        ),
        TestCase(
            name="wait for multiple statuses",
            expected_status=SUCCESS,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "status": {
                    constants.OPTIMIZATION_JOB_RUNNING,
                    constants.OPTIMIZATION_JOB_COMPLETE,
                },
                "_conditions": [
                    models.V1beta1ExperimentCondition(
                        type=constants.EXPERIMENT_SUCCEEDED, status="True"
                    ),
                ],
            },
            expected_output=get_optimization_job_data_type(
                name=BASIC_OPTIMIZATION_JOB_NAME,
                status=constants.OPTIMIZATION_JOB_COMPLETE,
            ),
        ),
        TestCase(
            name="callback invoked on each poll iteration",
            expected_status=SUCCESS,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "_conditions": [
                    models.V1beta1ExperimentCondition(
                        type=constants.EXPERIMENT_SUCCEEDED, status="True"
                    ),
                ],
                "_has_callback": True,
            },
            expected_output=get_optimization_job_data_type(
                name=BASIC_OPTIMIZATION_JOB_NAME,
                status=constants.OPTIMIZATION_JOB_COMPLETE,
            ),
        ),
        TestCase(
            name="invalid status set error",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "status": {"InvalidStatus"},
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="polling interval is more than timeout error",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "timeout": 1,
                "polling_interval": 2,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="polling interval equal to timeout raises ValueError",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "timeout": 10,
                "polling_interval": 10,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="zero polling_interval raises ValueError",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "timeout": 10,
                "polling_interval": 0,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="negative polling_interval raises ValueError",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "timeout": 10,
                "polling_interval": -1,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="job failed when not expected",
            expected_status=FAILED,
            config={
                "name": "failed-job",
                "status": {constants.OPTIMIZATION_JOB_RUNNING},
                "_conditions": [
                    models.V1beta1ExperimentCondition(
                        type=constants.OPTIMIZATION_JOB_FAILED, status="True"
                    ),
                ],
            },
            expected_error=RuntimeError,
        ),
        TestCase(
            name="timeout error to wait for failed status",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "status": {constants.OPTIMIZATION_JOB_FAILED},
                "polling_interval": 1,
                "timeout": 2,
                "_conditions": [
                    models.V1beta1ExperimentCondition(
                        type=constants.EXPERIMENT_SUCCEEDED, status="True"
                    ),
                ],
            },
            expected_error=TimeoutError,
        ),
    ],
)
def test_wait_for_job_status(optimizer_backend, test_case):
    """Test KubernetesBackend.wait_for_job_status with various scenarios."""
    print("Executing test:", test_case.name)

    job_name = test_case.config.get("name", BASIC_OPTIMIZATION_JOB_NAME)
    status_conditions = test_case.config.get("_conditions")

    experiment = create_experiment_cr(name=job_name, status_conditions=status_conditions)

    def patched_get(*args, **kwargs):
        mock_thread = Mock()
        if args[3] == constants.EXPERIMENT_PLURAL:
            mock_thread.get.return_value = normalize_model(experiment, models.V1beta1Experiment)
        return mock_thread

    optimizer_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    trail_status = test_case.config.get("_trial_status", trainer_constants.TRAINJOB_COMPLETE)

    def patched_trainer_get_job(name: str) -> TrainJob:
        job = create_mock_trainjob(name)
        job.status = trail_status
        return job

    optimizer_backend.trainer_backend.get_job = Mock(side_effect=patched_trainer_get_job)

    mock_callback = Mock()

    wait_kwargs = {k: v for k, v in test_case.config.items() if not k.startswith("_")}

    if test_case.config.get("_has_callback"):
        wait_kwargs["callbacks"] = [mock_callback]

    try:
        with patch("time.sleep", return_value=None):
            job = optimizer_backend.wait_for_job_status(**wait_kwargs)

        assert test_case.expected_status == SUCCESS
        assert isinstance(job, OptimizationJob)
        assert job.status in test_case.config.get("status", {constants.OPTIMIZATION_JOB_COMPLETE})
        assert asdict(job) == asdict(test_case.expected_output)

        if test_case.config.get("_has_callback"):
            mock_callback.assert_called()
            for call_args in mock_callback.call_args_list:
                assert isinstance(call_args[0][0], OptimizationJob)

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with all defaults",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_output=None,
        ),
        TestCase(
            name="timeout error when deleting job",
            expected_status=FAILED,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "namespace": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when deleting job",
            expected_status=FAILED,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "namespace": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_delete_job(optimizer_backend, test_case):
    """Test KubernetesBackend.delete_job with success and error paths."""
    print("Executing test:", test_case.name)
    try:
        optimizer_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        optimizer_backend.delete_job(test_case.config.get("name"))
        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get job events with valid optimization job",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_output=[
                Event(
                    involved_object_kind=constants.EXPERIMENT_KIND,
                    involved_object_name=BASIC_OPTIMIZATION_JOB_NAME,
                    message="Experiment created successfully",
                    reason="Created",
                    event_time=datetime.datetime(2025, 6, 1, 10, 30, 0),
                ),
                Event(
                    involved_object_kind=constants.TRIAL_KIND,
                    involved_object_name=BASIC_TRIAL_NAME,
                    message="Trial started",
                    reason="Running",
                    event_time=datetime.datetime(2025, 6, 1, 10, 31, 0),
                ),
            ],
        ),
        TestCase(
            name="timeout error when getting job events",
            expected_status=FAILED,
            config={"namespace": TIMEOUT, "name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_error=TimeoutError,
        ),
    ],
)
def test_get_job_events(optimizer_backend, test_case):
    """Test KubernetesBackend.get_job_events with various scenarios."""
    print("Executing test:", test_case.name)
    try:
        optimizer_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        events = optimizer_backend.get_job_events(test_case.config.get("name"))

        assert test_case.expected_status == SUCCESS
        assert isinstance(events, list)
        assert len(events) == len(test_case.expected_output)
        assert [asdict(e) for e in events] == [asdict(e) for e in test_case.expected_output]

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error
    print("test execution complete")
