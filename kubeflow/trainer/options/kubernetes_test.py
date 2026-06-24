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

"""Unit tests for Kubernetes options."""

import pytest

from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.backends.localprocess.backend import LocalProcessBackend
from kubeflow.trainer.options import (
    Annotations,
    ContainerPatch,
    JobSetSpecPatch,
    JobSetTemplatePatch,
    JobSpecPatch,
    JobTemplatePatch,
    Labels,
    Name,
    PodSpecPatch,
    PodTemplatePatch,
    ReplicatedJobPatch,
    RuntimePatch,
    TrainerArgs,
    TrainerCommand,
    TrainingRuntimeSpecPatch,
)


@pytest.fixture
def mock_kubernetes_backend():
    """Mock Kubernetes backend for testing."""
    from unittest.mock import Mock

    backend = Mock(spec=KubernetesBackend)
    backend.__class__ = KubernetesBackend
    return backend


@pytest.fixture
def mock_localprocess_backend():
    """Mock LocalProcess backend for testing."""
    from unittest.mock import MagicMock

    # Create a proper mock that isinstance checks will work with
    backend = MagicMock(spec=LocalProcessBackend)
    # Make type(backend).__name__ return the correct class name
    type(backend).__name__ = "LocalProcessBackend"
    return backend


class TestKubernetesOptionBackendValidation:
    """Test that Kubernetes options validate backend compatibility."""

    @pytest.mark.parametrize(
        "option_class,option_args",
        [
            (Labels, {"app": "test", "version": "v1"}),
            (Annotations, {"description": "test job"}),
            (TrainerCommand, ["python", "train.py"]),
            (TrainerArgs, ["--epochs", "10"]),
        ],
    )
    def test_kubernetes_options_reject_wrong_backend(
        self, option_class, option_args, mock_localprocess_backend
    ):
        """Test Kubernetes-specific options reject non-Kubernetes backends."""
        if option_class == TrainerCommand:
            option = option_class(command=option_args)
        elif option_class == TrainerArgs:
            option = option_class(args=option_args)
        else:
            option = option_class(option_args)

        job_spec = {}

        with pytest.raises(ValueError) as exc_info:
            option(job_spec, None, mock_localprocess_backend)

        assert "not compatible with" in str(exc_info.value)
        assert "LocalProcessBackend" in str(exc_info.value)

    def test_runtime_patch_rejects_wrong_backend(self, mock_localprocess_backend):
        """Test RuntimePatch rejects non-Kubernetes backends."""
        patch = RuntimePatch()

        job_spec = {}

        with pytest.raises(ValueError) as exc_info:
            patch(job_spec, None, mock_localprocess_backend)

        assert "not compatible with" in str(exc_info.value)


class TestKubernetesOptionApplication:
    """Test Kubernetes option application behavior."""

    @pytest.mark.parametrize(
        "option_class,option_args,expected_spec",
        [
            (
                Labels,
                {"app": "test", "version": "v1"},
                {"metadata": {"labels": {"app": "test", "version": "v1"}}},
            ),
            (
                Annotations,
                {"description": "test job"},
                {"metadata": {"annotations": {"description": "test job"}}},
            ),
            (Name, "custom-job-name", {"metadata": {"name": "custom-job-name"}}),
            (
                TrainerCommand,
                ["python", "train.py"],
                {"spec": {"trainer": {"command": ["python", "train.py"]}}},
            ),
            (
                TrainerArgs,
                ["--epochs", "10"],
                {"spec": {"trainer": {"args": ["--epochs", "10"]}}},
            ),
        ],
    )
    def test_option_application(
        self, option_class, option_args, expected_spec, mock_kubernetes_backend
    ):
        """Test each option applies correctly to job spec with Kubernetes backend."""
        if option_class == TrainerCommand:
            option = option_class(command=option_args)
        elif option_class == TrainerArgs:
            option = option_class(args=option_args)
        else:
            option = option_class(option_args)

        job_spec = {}
        option(job_spec, None, mock_kubernetes_backend)

        assert job_spec == expected_spec


class TestTrainerOptionValidation:
    """Test validation of trainer-specific options."""

    @pytest.mark.parametrize(
        "option_class,option_args,trainer_type,should_fail",
        [
            # Validation failures
            (TrainerCommand, ["python", "train.py"], "CustomTrainer", True),
            (TrainerArgs, ["--epochs", "10"], "CustomTrainer", True),
            (TrainerCommand, ["python", "train.py"], "BuiltinTrainer", True),
            (TrainerArgs, ["--epochs", "10"], "BuiltinTrainer", True),
            # Successful applications
            (TrainerCommand, ["python", "train.py"], "CustomTrainerContainer", False),
            (TrainerArgs, ["--epochs", "10"], "CustomTrainerContainer", False),
        ],
    )
    def test_trainer_option_validation(
        self, option_class, option_args, trainer_type, should_fail, mock_kubernetes_backend
    ):
        """Test trainer option validation with different trainer types."""
        from kubeflow.trainer.types.types import (
            BuiltinTrainer,
            CustomTrainer,
            CustomTrainerContainer,
            TorchTuneConfig,
        )

        # Create appropriate trainer instance
        if trainer_type == "CustomTrainer":

            def dummy_func():
                pass

            trainer = CustomTrainer(func=dummy_func)
        elif trainer_type == "BuiltinTrainer":
            trainer = BuiltinTrainer(config=TorchTuneConfig())
        else:  # CustomTrainerContainer
            trainer = CustomTrainerContainer(image="custom-image:latest")

        # Create option
        if option_class == TrainerCommand:
            option = option_class(command=option_args)
        else:  # TrainerArgs
            option = option_class(args=option_args)

        job_spec = {}

        if should_fail:
            with pytest.raises(ValueError) as exc_info:
                option(job_spec, trainer, mock_kubernetes_backend)
            assert "TrainerCommand can only be used with CustomTrainerContainer" in str(
                exc_info.value
            ) or "TrainerArgs can only be used with CustomTrainerContainer" in str(exc_info.value)
        else:
            option(job_spec, trainer, mock_kubernetes_backend)
            if option_class == TrainerCommand:
                assert job_spec["spec"]["trainer"]["command"] == option_args
            else:
                assert job_spec["spec"]["trainer"]["args"] == option_args


class TestContainerPatch:
    """Test ContainerPatch validation."""

    @pytest.mark.parametrize(
        "kwargs,expected_error",
        [
            ({"name": ""}, "Container name must be a non-empty string"),
            (
                {"name": "trainer", "env": [{"invalid": "structure"}]},
                "Each env entry must have a 'name' key",
            ),
            (
                {"name": "trainer", "volume_mounts": [{"name": "vol"}]},
                "Each volume_mounts entry must have a 'mountPath' key",
            ),
        ],
    )
    def test_container_patch_validation(self, kwargs, expected_error):
        """Test ContainerPatch validates inputs correctly."""
        with pytest.raises(ValueError) as exc_info:
            ContainerPatch(**kwargs)
        assert expected_error in str(exc_info.value)


class TestRuntimePatch:
    """Test RuntimePatch validation."""

    def test_runtime_patch_auto_sets_manager(self):
        """Test RuntimePatch automatically sets manager."""
        patch = RuntimePatch()
        assert patch.manager == "trainer.kubeflow.org/kubeflow-sdk"

    def test_runtime_patch_with_training_runtime_spec(self):
        """Test RuntimePatch with training runtime spec."""
        patch = RuntimePatch(
            training_runtime_spec=TrainingRuntimeSpecPatch(
                template=JobSetTemplatePatch(
                    metadata={"labels": {"app": "training"}},
                ),
            ),
        )
        assert patch.manager == "trainer.kubeflow.org/kubeflow-sdk"
        assert patch.training_runtime_spec is not None


class TestRuntimePatchApplication:
    """Test RuntimePatch application functionality."""

    def test_runtime_patch_basic(self, mock_kubernetes_backend):
        """Test basic RuntimePatch application with manager only."""
        patch = RuntimePatch()

        job_spec = {}
        patch(job_spec, None, mock_kubernetes_backend)

        assert "spec" in job_spec
        assert "runtimePatches" in job_spec["spec"]
        assert len(job_spec["spec"]["runtimePatches"]) == 1
        assert job_spec["spec"]["runtimePatches"][0] == {
            "manager": "trainer.kubeflow.org/kubeflow-sdk"
        }

    def test_runtime_patch_with_node_selector(self, mock_kubernetes_backend):
        """Test RuntimePatch with node selector configuration."""
        patch = RuntimePatch(
            training_runtime_spec=TrainingRuntimeSpecPatch(
                template=JobSetTemplatePatch(
                    spec=JobSetSpecPatch(
                        replicated_jobs=[
                            ReplicatedJobPatch(
                                name="node",
                                template=JobTemplatePatch(
                                    spec=JobSpecPatch(
                                        template=PodTemplatePatch(
                                            spec=PodSpecPatch(
                                                node_selector={"node-type": "gpu-a100"},
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        )

        job_spec = {}
        patch(job_spec, None, mock_kubernetes_backend)

        expected = {
            "spec": {
                "runtimePatches": [
                    {
                        "manager": "trainer.kubeflow.org/kubeflow-sdk",
                        "trainingRuntimeSpec": {
                            "template": {
                                "spec": {
                                    "replicatedJobs": [
                                        {
                                            "name": "node",
                                            "template": {
                                                "spec": {
                                                    "template": {
                                                        "spec": {
                                                            "nodeSelector": {
                                                                "node-type": "gpu-a100"
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    ],
                                },
                            },
                        },
                    },
                ],
            },
        }

        assert job_spec == expected

    def test_runtime_patch_with_volume_and_container(self, mock_kubernetes_backend):
        """Test RuntimePatch with volumes and container patches."""
        patch = RuntimePatch(
            training_runtime_spec=TrainingRuntimeSpecPatch(
                template=JobSetTemplatePatch(
                    spec=JobSetSpecPatch(
                        replicated_jobs=[
                            ReplicatedJobPatch(
                                name="node",
                                template=JobTemplatePatch(
                                    spec=JobSpecPatch(
                                        template=PodTemplatePatch(
                                            spec=PodSpecPatch(
                                                volumes=[
                                                    {
                                                        "name": "user-volume",
                                                        "persistentVolumeClaim": {
                                                            "claimName": "user-pvc"
                                                        },
                                                    }
                                                ],
                                                containers=[
                                                    ContainerPatch(
                                                        name="trainer",
                                                        volume_mounts=[
                                                            {
                                                                "name": "user-volume",
                                                                "mountPath": "/workspace",
                                                            }
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        )

        job_spec = {}
        patch(job_spec, None, mock_kubernetes_backend)

        runtime_patch = job_spec["spec"]["runtimePatches"][0]
        pod_spec = runtime_patch["trainingRuntimeSpec"]["template"]["spec"]["replicatedJobs"][0][
            "template"
        ]["spec"]["template"]["spec"]

        assert pod_spec["volumes"] == [
            {"name": "user-volume", "persistentVolumeClaim": {"claimName": "user-pvc"}}
        ]
        assert pod_spec["containers"] == [
            {
                "name": "trainer",
                "volumeMounts": [{"name": "user-volume", "mountPath": "/workspace"}],
            }
        ]

    def test_runtime_patch_with_jobset_metadata(self, mock_kubernetes_backend):
        """Test RuntimePatch with JobSet-level metadata."""
        patch = RuntimePatch(
            training_runtime_spec=TrainingRuntimeSpecPatch(
                template=JobSetTemplatePatch(
                    metadata={"labels": {"app": "training"}},
                ),
            ),
        )

        job_spec = {}
        patch(job_spec, None, mock_kubernetes_backend)

        assert job_spec["spec"]["runtimePatches"][0] == {
            "manager": "trainer.kubeflow.org/kubeflow-sdk",
            "trainingRuntimeSpec": {
                "template": {
                    "metadata": {"labels": {"app": "training"}},
                },
            },
        }
