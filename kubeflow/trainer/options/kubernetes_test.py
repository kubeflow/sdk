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
    ContainerOverride,
    Labels,
    Name,
    PodTemplateOverride,
    PodTemplateOverrides,
    SpecAnnotations,
    SpecLabels,
    TrainerArgs,
    TrainerCommand,
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
            (SpecLabels, {"app": "training"}),
            (SpecAnnotations, {"prometheus.io/scrape": "true"}),
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

    def test_pod_template_overrides_rejects_wrong_backend(self, mock_localprocess_backend):
        """Test PodTemplateOverrides rejects non-Kubernetes backends."""
        override = PodTemplateOverride(target_jobs=["node"])
        option = PodTemplateOverrides(override)

        job_spec = {}

        with pytest.raises(ValueError) as exc_info:
            option(job_spec, None, mock_localprocess_backend)

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
            (
                SpecLabels,
                {"app": "training", "version": "v1.0"},
                {"spec": {"labels": {"app": "training", "version": "v1.0"}}},
            ),
            (
                SpecAnnotations,
                {"prometheus.io/scrape": "true"},
                {"spec": {"annotations": {"prometheus.io/scrape": "true"}}},
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


class TestContainerOverride:
    """Test ContainerOverride validation."""

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
    def test_container_override_validation(self, kwargs, expected_error):
        """Test ContainerOverride validates inputs correctly."""
        with pytest.raises(ValueError) as exc_info:
            ContainerOverride(**kwargs)
        assert expected_error in str(exc_info.value)


class TestPodTemplateOverrides:
    """Test PodTemplateOverrides functionality."""

    def test_pod_template_overrides_basic(self, mock_kubernetes_backend):
        """Test basic PodTemplateOverrides application."""

        override = PodTemplateOverride(target_jobs=["node"])
        option = PodTemplateOverrides(override)

        job_spec = {}
        option(job_spec, None, mock_kubernetes_backend)

        assert "spec" in job_spec
        assert "podTemplateOverrides" in job_spec["spec"]
        assert len(job_spec["spec"]["podTemplateOverrides"]) == 1
        assert job_spec["spec"]["podTemplateOverrides"][0]["targetJobs"] == [{"name": "node"}]
