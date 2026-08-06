# Copyright 2026 The Kubeflow Authors.
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

"""Unit tests for the TrainerNativeBackend class in the Kubeflow Optimizer SDK.

This module uses pytest and unittest.mock to simulate Kubernetes API interactions.
It tests TrainerNativeBackend's behavior across OptimizationJob creation, listing,
retrieval, deletion, log retrieval, event filtering, and status waiting against
the trainer-native OptimizationJob CRD (trainer.kubeflow.org/v1alpha1).
"""

from dataclasses import asdict
import datetime
import multiprocessing
from typing import Any
from unittest.mock import Mock, patch

from kubernetes import client as k8s_client
import pytest

from kubeflow.optimizer.backends.trainer_native.backend import TrainerNativeBackend
from kubeflow.optimizer.backends.trainer_native.types import TrainerNativeBackendConfig
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import GridSearch, RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
    Result,
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
from kubeflow.trainer.types import types as trainer_types
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

BASIC_OPTIMIZATION_JOB_NAME = "basic-opt-job"
BASIC_TRIAL_NAME = "basic-trial"

# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def trainer_native_backend():
    """Provide a TrainerNativeBackend with mocked Kubernetes APIs."""
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
        backend = TrainerNativeBackend(TrainerNativeBackendConfig())
        backend.trainer_backend._get_trainjob_spec = Mock(
            return_value=Mock(to_dict=Mock(return_value={}))
        )
        backend.trainer_backend.get_job = Mock(side_effect=mock_trainer_get_job)
        backend.trainer_backend.get_job_logs = Mock(return_value=iter(["test log content"]))
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
    """Return a mocked OptimizationJob CR dict.

    Args:
        args: Positional args from the K8s API call.
            args[4] is the resource name for get_namespaced_custom_object.
    """
    mock_thread = Mock()
    if args[4] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[4] == RUNTIME:
        raise RuntimeError()
    if args[3] == constants.OPTIMIZATION_JOB_PLURAL:
        mock_thread.get.return_value = create_optimization_job_cr(name=args[4])
    return mock_thread


def list_namespaced_custom_object_response(*args, **kwargs):
    """Return a list of mocked OptimizationJob CR dicts.

    Args:
        args: Positional args from the K8s API call.
            args[2] is the namespace for list_namespaced_custom_object.
    """
    mock_thread = Mock()
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME:
        raise RuntimeError()
    if args[3] == constants.OPTIMIZATION_JOB_PLURAL:
        mock_thread.get.return_value = {
            "items": [
                create_optimization_job_cr(name="opt-job-1"),
                create_optimization_job_cr(name="opt-job-2"),
            ],
        }
    return mock_thread


def mock_list_namespaced_event(*args, **kwargs):
    """Simulate event listing from namespace."""
    namespace = kwargs.get("namespace")

    if namespace == TIMEOUT:
        raise multiprocessing.TimeoutError()

    mock_thread = Mock()
    mock_thread.get.return_value = k8s_client.CoreV1EventList(
        items=[
            k8s_client.CoreV1Event(
                metadata=k8s_client.V1ObjectMeta(
                    name="test-event-1",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involved_object=k8s_client.V1ObjectReference(
                    kind=constants.OPTIMIZATION_JOB_KIND,
                    name=BASIC_OPTIMIZATION_JOB_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="OptimizationJob created successfully",
                reason="Created",
                first_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
            ),
            k8s_client.CoreV1Event(
                metadata=k8s_client.V1ObjectMeta(
                    name="test-event-2",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involved_object=k8s_client.V1ObjectReference(
                    kind=trainer_constants.TRAINJOB_KIND,
                    name=BASIC_TRIAL_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="TrainJob started",
                reason="Running",
                first_timestamp=datetime.datetime(2025, 6, 1, 10, 31, 0),
            ),
            # Non-matching event (Pod kind) to test filtering
            k8s_client.CoreV1Event(
                metadata=k8s_client.V1ObjectMeta(
                    name="test-event-3",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involved_object=k8s_client.V1ObjectReference(
                    kind="Pod",
                    name="some-pod",
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Pod scheduled",
                reason="Scheduled",
                first_timestamp=datetime.datetime(2025, 6, 1, 10, 32, 0),
            ),
        ]
    )
    return mock_thread


def mock_trainer_get_job(name: str) -> TrainJob:
    """Return a mock TrainJob for the given trial name."""
    return create_mock_trainjob(name)


# --------------------------
# Object Creators
# --------------------------


def create_optimization_job_cr(
    name: str = BASIC_OPTIMIZATION_JOB_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    status_conditions: list[dict[str, Any]] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a mock trainer-native OptimizationJob CR dict."""
    optimization_job = {
        "apiVersion": trainer_constants.API_VERSION,
        "kind": constants.OPTIMIZATION_JOB_KIND,
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": "2025-06-01T10:30:00Z",
        },
        "spec": {
            "objectives": [{"metric": "loss", "direction": "Minimize"}],
            "searchAlgorithm": {"random": {}},
            "parameters": [
                {
                    "name": "lr",
                    "searchSpace": {
                        "uniform": {"min": "0.001", "max": "0.1", "type": "Float"},
                    },
                },
            ],
            "numTrials": 10,
            "parallelTrials": 1,
            "trainJobTemplate": {"spec": {}},
        },
    }

    status: dict[str, Any] = {}
    if status_conditions is not None:
        status["conditions"] = status_conditions
    if result is not None:
        status["result"] = result
    if status:
        optimization_job["status"] = status

    return optimization_job


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
            kind=trainer_types.RuntimeKind.TRAINING_RUNTIME,
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
) -> OptimizationJob:
    """Create the expected OptimizationJob output for assertion comparison."""
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
        trials=[],
        creation_timestamp=datetime.datetime(2025, 6, 1, 10, 30, 0, tzinfo=datetime.timezone.utc),
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
                "expected_parameters": [
                    {
                        "name": "lr",
                        "searchSpace": {
                            "uniform": {"min": "0.001", "max": "0.1", "type": "Float"},
                        },
                    },
                ],
            },
        ),
        TestCase(
            name="multiple search space parameters",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.loguniform(min=0.001, max=0.1),
                    "epochs": Search.choice([10, 20, 30]),
                },
                "expected_parameters": [
                    {
                        "name": "lr",
                        "searchSpace": {
                            "logUniform": {"min": "0.001", "max": "0.1", "type": "Float"},
                        },
                    },
                    {
                        "name": "epochs",
                        "searchSpace": {
                            "categorical": {"choices": ["10", "20", "30"]},
                        },
                    },
                ],
            },
        ),
        TestCase(
            name="random search algorithm with seed",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
                "algorithm": RandomSearch(random_state=42),
                "expected_search_algorithm": {"random": {"seed": 42}},
            },
        ),
        TestCase(
            name="grid search algorithm",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
                "algorithm": GridSearch(),
                "expected_search_algorithm": {"grid": {}},
            },
        ),
        TestCase(
            name="maximize objective direction",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
                "objectives": [Objective(metric="accuracy", direction="maximize")],
                "expected_objectives": [{"metric": "accuracy", "direction": "Maximize"}],
            },
        ),
        TestCase(
            name="custom trial config",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
                "trial_config": TrialConfig(num_trials=20, parallel_trials=4),
                "expected_num_trials": 20,
                "expected_parallel_trials": 4,
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
            name="multiple objectives raise ValueError",
            expected_status=FAILED,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
                "objectives": [Objective(metric="loss"), Objective(metric="accuracy")],
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="max_failed_trials raises ValueError",
            expected_status=FAILED,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
                "trial_config": TrialConfig(max_failed_trials=3),
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
def test_optimize(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.optimize with success and error paths."""
    print("Executing test:", test_case.name)

    search_space = test_case.config["search_space"]

    trial_template = TrainJobTemplate(
        trainer=CustomTrainer(
            func=lambda: None,
            func_args={"existing_arg": "original_value"},
            num_nodes=1,
        ),
    )
    original_func_args = dict(trial_template.trainer.func_args)

    try:
        trainer_native_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        job_name = trainer_native_backend.optimize(
            trial_template=trial_template,
            search_space=search_space,
            trial_config=test_case.config.get("trial_config"),
            objectives=test_case.config.get("objectives"),
            algorithm=test_case.config.get("algorithm"),
        )

        assert test_case.expected_status == SUCCESS
        assert isinstance(job_name, str) and len(job_name) > 0

        # The controller injects hyperparameters via environment variables, so
        # trial_template.trainer.func_args must be unchanged.
        assert trial_template.trainer.func_args == original_func_args

        # Verify the OptimizationJob CR was created with the expected payload.
        trainer_native_backend.custom_api.create_namespaced_custom_object.assert_called_once()
        call_args = trainer_native_backend.custom_api.create_namespaced_custom_object.call_args
        assert call_args[0][0] == trainer_constants.GROUP
        assert call_args[0][1] == trainer_constants.VERSION
        assert call_args[0][3] == constants.OPTIMIZATION_JOB_PLURAL

        payload = call_args[0][4]
        assert payload["apiVersion"] == trainer_constants.API_VERSION
        assert payload["kind"] == constants.OPTIMIZATION_JOB_KIND
        assert payload["spec"]["trainJobTemplate"] == {"spec": {}}
        assert payload["spec"]["numTrials"] == test_case.config.get("expected_num_trials", 10)
        assert payload["spec"]["parallelTrials"] == test_case.config.get(
            "expected_parallel_trials", 1
        )
        assert payload["spec"]["objectives"] == test_case.config.get(
            "expected_objectives", [{"metric": "loss", "direction": "Minimize"}]
        )
        assert payload["spec"]["searchAlgorithm"] == test_case.config.get(
            "expected_search_algorithm", {"random": {}}
        )
        if "expected_parameters" in test_case.config:
            assert payload["spec"]["parameters"] == test_case.config["expected_parameters"]

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
def test_get_job(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.get_job with success and error paths."""
    print("Executing test:", test_case.name)
    try:
        job = trainer_native_backend.get_job(**test_case.config)

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
            name="complete condition maps to OPTIMIZATION_JOB_COMPLETE",
            expected_status=SUCCESS,
            config={
                "name": "succeeded-job",
                "conditions": [
                    {"type": constants.OPTIMIZATION_JOB_COMPLETE, "status": "True"},
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
                    {"type": constants.OPTIMIZATION_JOB_FAILED, "status": "True"},
                ],
            },
            expected_output=get_optimization_job_data_type(
                name="failed-job",
                status=constants.OPTIMIZATION_JOB_FAILED,
            ),
        ),
        TestCase(
            name="created condition maps to OPTIMIZATION_JOB_CREATED",
            expected_status=SUCCESS,
            config={
                "name": "created-job",
                "conditions": [
                    {"type": constants.OPTIMIZATION_JOB_CREATED, "status": "True"},
                ],
            },
            expected_output=get_optimization_job_data_type(
                name="created-job",
                status=constants.OPTIMIZATION_JOB_CREATED,
            ),
        ),
        TestCase(
            name="condition with False status is ignored",
            expected_status=SUCCESS,
            config={
                "name": "pending-job",
                "conditions": [
                    {"type": constants.OPTIMIZATION_JOB_COMPLETE, "status": "False"},
                ],
            },
            expected_output=get_optimization_job_data_type(
                name="pending-job",
                status=constants.OPTIMIZATION_JOB_CREATED,
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
def test_get_job_status_conditions(trainer_native_backend, test_case):
    """Test status-mapping logic in __get_optimization_job_from_cr."""
    print("Executing test:", test_case.name)

    job_name = test_case.config["name"]
    conditions = test_case.config.get("conditions")

    optimization_job_cr = create_optimization_job_cr(name=job_name, status_conditions=conditions)

    def patched_get(*args, **kwargs):
        mock_thread = Mock()
        if args[3] == constants.OPTIMIZATION_JOB_PLURAL and args[4] == job_name:
            mock_thread.get.return_value = optimization_job_cr
        return mock_thread

    trainer_native_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    job = trainer_native_backend.get_job(name=job_name)
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
def test_list_jobs(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.list_jobs with success and error paths."""
    print("Executing test:", test_case.name)
    try:
        trainer_native_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        jobs = trainer_native_backend.list_jobs()

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
            name="best trial available in status result",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_result": True},
            expected_output=Result(
                parameters={"lr": "0.01"},
                metrics=[],
            ),
        ),
        TestCase(
            name="no best trial available",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_result": False},
            expected_output=None,
        ),
        TestCase(
            name="timeout error when getting best results",
            expected_status=FAILED,
            config={"name": TIMEOUT, "has_result": False},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting best results",
            expected_status=FAILED,
            config={"name": RUNTIME, "has_result": False},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_best_results(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.get_best_results with success and error paths."""
    print("Executing test:", test_case.name)

    if test_case.config.get("has_result"):
        optimization_job_cr = create_optimization_job_cr(
            name=BASIC_OPTIMIZATION_JOB_NAME,
            result={
                "trainJobName": BASIC_TRIAL_NAME,
                "parameters": [{"name": "lr", "value": "0.01"}],
            },
        )
        original_handler = (
            trainer_native_backend.custom_api.get_namespaced_custom_object.side_effect
        )

        def patched_get(*args, **kwargs):
            if args[3] == constants.OPTIMIZATION_JOB_PLURAL:
                mock_thread = Mock()
                mock_thread.get.return_value = optimization_job_cr
                return mock_thread
            return original_handler(*args, **kwargs)

        trainer_native_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    try:
        result = trainer_native_backend.get_best_results(name=test_case.config["name"])

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
            name="explicit trial_name delegates to trainer backend",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "trial_name": BASIC_TRIAL_NAME},
            expected_output=["test log content"],
        ),
        TestCase(
            name="logs from the best trial when trial_name is not set",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_result": True},
            expected_output=["test log content"],
        ),
        TestCase(
            name="no best trial returns no logs",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_result": False},
            expected_output=[],
        ),
        TestCase(
            name="timeout error when getting job logs",
            expected_status=FAILED,
            config={"name": TIMEOUT},
            expected_error=TimeoutError,
        ),
    ],
)
def test_get_job_logs(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.get_job_logs with success and error paths."""
    print("Executing test:", test_case.name)

    if test_case.config.get("has_result"):
        optimization_job_cr = create_optimization_job_cr(
            name=BASIC_OPTIMIZATION_JOB_NAME,
            result={
                "trainJobName": BASIC_TRIAL_NAME,
                "parameters": [{"name": "lr", "value": "0.01"}],
            },
        )

        def patched_get(*args, **kwargs):
            mock_thread = Mock()
            mock_thread.get.return_value = optimization_job_cr
            return mock_thread

        trainer_native_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    try:
        logs = trainer_native_backend.get_job_logs(
            test_case.config.get("name"),
            trial_name=test_case.config.get("trial_name"),
            follow=test_case.config.get("follow", False),
        )
        logs_list = list(logs)
        assert test_case.expected_status == SUCCESS
        assert logs_list == test_case.expected_output

        if logs_list:
            trainer_native_backend.trainer_backend.get_job_logs.assert_called_once_with(
                name=BASIC_TRIAL_NAME,
                follow=test_case.config.get("follow", False),
            )

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
                    {"type": constants.OPTIMIZATION_JOB_COMPLETE, "status": "True"},
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
                    {"type": constants.OPTIMIZATION_JOB_COMPLETE, "status": "True"},
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
                    {"type": constants.OPTIMIZATION_JOB_COMPLETE, "status": "True"},
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
            name="job failed when not expected",
            expected_status=FAILED,
            config={
                "name": "failed-job",
                "status": {constants.OPTIMIZATION_JOB_RUNNING},
                "_conditions": [
                    {"type": constants.OPTIMIZATION_JOB_FAILED, "status": "True"},
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
                    {"type": constants.OPTIMIZATION_JOB_COMPLETE, "status": "True"},
                ],
            },
            expected_error=TimeoutError,
        ),
    ],
)
def test_wait_for_job_status(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.wait_for_job_status with various scenarios."""
    print("Executing test:", test_case.name)

    job_name = test_case.config.get("name", BASIC_OPTIMIZATION_JOB_NAME)
    status_conditions = test_case.config.get("_conditions")

    optimization_job_cr = create_optimization_job_cr(
        name=job_name, status_conditions=status_conditions
    )

    def patched_get(*args, **kwargs):
        mock_thread = Mock()
        if args[3] == constants.OPTIMIZATION_JOB_PLURAL:
            mock_thread.get.return_value = optimization_job_cr
        return mock_thread

    trainer_native_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    mock_callback = Mock()

    wait_kwargs = {k: v for k, v in test_case.config.items() if not k.startswith("_")}

    if test_case.config.get("_has_callback"):
        wait_kwargs["callbacks"] = [mock_callback]

    try:
        with patch("time.sleep", return_value=None):
            job = trainer_native_backend.wait_for_job_status(**wait_kwargs)

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
def test_delete_job(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.delete_job with success and error paths."""
    print("Executing test:", test_case.name)
    try:
        trainer_native_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        trainer_native_backend.delete_job(test_case.config.get("name"))
        assert test_case.expected_status == SUCCESS

        call_args = trainer_native_backend.custom_api.delete_namespaced_custom_object.call_args
        assert call_args[0][0] == trainer_constants.GROUP
        assert call_args[0][3] == constants.OPTIMIZATION_JOB_PLURAL

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
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_result": True},
            expected_output=[
                Event(
                    involved_object_kind=constants.OPTIMIZATION_JOB_KIND,
                    involved_object_name=BASIC_OPTIMIZATION_JOB_NAME,
                    message="OptimizationJob created successfully",
                    reason="Created",
                    event_time=datetime.datetime(2025, 6, 1, 10, 30, 0),
                ),
                Event(
                    involved_object_kind=trainer_constants.TRAINJOB_KIND,
                    involved_object_name=BASIC_TRIAL_NAME,
                    message="TrainJob started",
                    reason="Running",
                    event_time=datetime.datetime(2025, 6, 1, 10, 31, 0),
                ),
            ],
        ),
        TestCase(
            name="events without best trial exclude TrainJob events",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "has_result": False},
            expected_output=[
                Event(
                    involved_object_kind=constants.OPTIMIZATION_JOB_KIND,
                    involved_object_name=BASIC_OPTIMIZATION_JOB_NAME,
                    message="OptimizationJob created successfully",
                    reason="Created",
                    event_time=datetime.datetime(2025, 6, 1, 10, 30, 0),
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
def test_get_job_events(trainer_native_backend, test_case):
    """Test TrainerNativeBackend.get_job_events with various scenarios."""
    print("Executing test:", test_case.name)

    if test_case.config.get("has_result"):
        optimization_job_cr = create_optimization_job_cr(
            name=BASIC_OPTIMIZATION_JOB_NAME,
            result={
                "trainJobName": BASIC_TRIAL_NAME,
                "parameters": [{"name": "lr", "value": "0.01"}],
            },
        )

        def patched_get(*args, **kwargs):
            mock_thread = Mock()
            mock_thread.get.return_value = optimization_job_cr
            return mock_thread

        trainer_native_backend.custom_api.get_namespaced_custom_object.side_effect = patched_get

    try:
        trainer_native_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        events = trainer_native_backend.get_job_events(test_case.config.get("name"))

        assert test_case.expected_status == SUCCESS
        assert isinstance(events, list)
        assert len(events) == len(test_case.expected_output)
        assert [asdict(e) for e in events] == [asdict(e) for e in test_case.expected_output]

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert type(e) is test_case.expected_error
    print("test execution complete")
