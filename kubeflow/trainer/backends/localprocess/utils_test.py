# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the localprocess utils module in the Kubeflow Trainer SDK."""

from unittest.mock import patch

import pytest

from kubeflow.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.backends.localprocess import utils
from kubeflow.trainer.backends.localprocess.constants import (
    DEFAULT_COMMAND,
    LOCAL_RUNTIME_IMAGE,
    TORCH_COMMAND,
    TORCH_FRAMEWORK_TYPE,
)
from kubeflow.trainer.backends.localprocess.types import LocalRuntimeTrainer
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


def dummy_training_function():
    """Dummy training function used to exercise real inspect.getsource()."""
    print("Training started")


def dummy_training_function_with_args(parameters):
    """Dummy training function with parameters, used to exercise real inspect.getsource()."""
    print(f"lr={parameters['lr']} epochs={parameters['epochs']}")


def _make_runtime(command=("python",), has_trainer=True):
    """A Runtime backed by a plain RuntimeTrainer (not LocalRuntimeTrainer)."""
    if not has_trainer:
        return types.Runtime(
            name="test-runtime", trainer=None, kind=types.RuntimeKind.TRAINING_RUNTIME
        )
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=TORCH_FRAMEWORK_TYPE,
        image=LOCAL_RUNTIME_IMAGE,
    )
    trainer.set_command(command)
    return types.Runtime(
        name="test-runtime", trainer=trainer, kind=types.RuntimeKind.TRAINING_RUNTIME
    )


def _make_local_runtime(packages=None, command=("python",)):
    """A Runtime backed by a real LocalRuntimeTrainer."""
    trainer = LocalRuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=TORCH_FRAMEWORK_TYPE,
        image=LOCAL_RUNTIME_IMAGE,
        packages=packages or ["torch"],
    )
    trainer.set_command(command)
    return types.Runtime(
        name="test-runtime", trainer=trainer, kind=types.RuntimeKind.TRAINING_RUNTIME
    )


def _make_custom_trainer(packages_to_install=None):
    return types.CustomTrainer(
        func=dummy_training_function,
        packages_to_install=packages_to_install,
    )


# ---------------------------------------------------------------------------
# _extract_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="plain_name",
            expected_status=SUCCESS,
            config={"requirement": "torch"},
            expected_output="torch",
        ),
        TestCase(
            name="name_with_extras",
            expected_status=SUCCESS,
            config={"requirement": "torch[cuda,extra]"},
            expected_output="torch",
        ),
        TestCase(
            name="name_with_version_specifier",
            expected_status=SUCCESS,
            config={"requirement": "torch==2.1.0"},
            expected_output="torch",
        ),
        TestCase(
            name="name_with_range_specifier",
            expected_status=SUCCESS,
            config={"requirement": "numpy>=1.0,<2.0"},
            expected_output="numpy",
        ),
        TestCase(
            name="name_with_url",
            expected_status=SUCCESS,
            config={"requirement": "torch @ https://example.com/torch.whl"},
            expected_output="torch",
        ),
        TestCase(
            name="name_with_marker",
            expected_status=SUCCESS,
            config={"requirement": "torch; python_version >= '3.10'"},
            expected_output="torch",
        ),
        TestCase(
            name="name_with_leading_whitespace",
            expected_status=SUCCESS,
            config={"requirement": "   torch"},
            expected_output="torch",
        ),
    ],
)
def test_extract_name(test_case):
    assert utils._extract_name(test_case.config["requirement"]) == test_case.expected_output


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="none_requirement",
            expected_status=FAILED,
            config={"requirement": None},
            expected_error=ValueError,
        ),
        TestCase(
            name="empty_requirement",
            expected_status=FAILED,
            config={"requirement": ""},
            expected_error=ValueError,
        ),
        TestCase(
            name="whitespace_only_requirement",
            expected_status=FAILED,
            config={"requirement": "   "},
            expected_error=ValueError,
        ),
        TestCase(
            name="unparseable_requirement",
            expected_status=FAILED,
            config={"requirement": "===not-a-name"},
            expected_error=ValueError,
        ),
    ],
)
def test_extract_name_errors(test_case):
    with pytest.raises(test_case.expected_error):
        utils._extract_name(test_case.config["requirement"])


# ---------------------------------------------------------------------------
# _canonicalize_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="already_canonical",
            expected_status=SUCCESS,
            config={"name": "torch"},
            expected_output="torch",
        ),
        TestCase(
            name="mixed_case",
            expected_status=SUCCESS,
            config={"name": "TorchVision"},
            expected_output="torchvision",
        ),
        TestCase(
            name="collapses_separators",
            expected_status=SUCCESS,
            config={"name": "Py_Torch.Lightning--Extra"},
            expected_output="py-torch-lightning-extra",
        ),
    ],
)
def test_canonicalize_name(test_case):
    assert utils._canonicalize_name(test_case.config["name"]) == test_case.expected_output


# ---------------------------------------------------------------------------
# get_install_packages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="no_trainer_packages_returns_runtime_as_is",
            expected_status=SUCCESS,
            config={"runtime_packages": ["torch", "numpy"], "trainer_packages": None},
            expected_output=["torch", "numpy"],
        ),
        TestCase(
            name="empty_trainer_packages_returns_runtime_as_is",
            expected_status=SUCCESS,
            config={"runtime_packages": ["torch"], "trainer_packages": []},
            expected_output=["torch"],
        ),
        TestCase(
            name="trainer_overwrites_matching_runtime_package",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch", "numpy"],
                "trainer_packages": ["torch==2.1.0"],
            },
            expected_output=["numpy", "torch==2.1.0"],
        ),
        TestCase(
            name="case_insensitive_overwrite",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["NumPy"],
                "trainer_packages": ["numpy==1.26.0"],
            },
            expected_output=["numpy==1.26.0"],
        ),
        TestCase(
            name="runtime_duplicates_last_occurrence_wins",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch==1.9", "numpy", "torch==2.0"],
                "trainer_packages": ["scipy"],
            },
            expected_output=["numpy", "torch==2.0", "scipy"],
        ),
        TestCase(
            name="trainer_only_packages_appended_in_order",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch"],
                "trainer_packages": ["scipy", "pandas"],
            },
            expected_output=["torch", "scipy", "pandas"],
        ),
    ],
)
def test_get_install_packages(test_case):
    result = utils.get_install_packages(
        runtime_packages=test_case.config["runtime_packages"],
        trainer_packages=test_case.config["trainer_packages"],
    )
    assert result == test_case.expected_output


def test_get_install_packages_duplicate_trainer_package_raises():
    with pytest.raises(ValueError, match="Duplicate dependency in trainer_packages"):
        utils.get_install_packages(
            runtime_packages=["torch"],
            trainer_packages=["torch==1.0", "Torch==2.0"],
        )


# ---------------------------------------------------------------------------
# get_local_runtime_trainer
# ---------------------------------------------------------------------------


def test_get_local_runtime_trainer_unknown_runtime_raises():
    with pytest.raises(ValueError, match="Runtime nonexistent-runtime not found"):
        utils.get_local_runtime_trainer(
            runtime_name="nonexistent-runtime",
            venv_dir="/tmp/venv",
            framework=TORCH_FRAMEWORK_TYPE,
        )


def test_get_local_runtime_trainer_torch_framework_uses_torchrun():
    trainer = utils.get_local_runtime_trainer(
        runtime_name=constants.DEFAULT_TRAINING_RUNTIME,
        venv_dir="/tmp/venv",
        framework=TORCH_FRAMEWORK_TYPE,
    )
    assert trainer.image == LOCAL_RUNTIME_IMAGE
    assert trainer.packages == ["torch"]
    assert trainer.command == (f"/tmp/venv/bin/{TORCH_COMMAND}",)


def test_get_local_runtime_trainer_non_torch_framework_uses_default_command():
    trainer = utils.get_local_runtime_trainer(
        runtime_name=constants.DEFAULT_TRAINING_RUNTIME,
        venv_dir="/tmp/venv",
        framework="custom-framework",
    )
    assert trainer.command == (f"/tmp/venv/bin/{DEFAULT_COMMAND}",)


# ---------------------------------------------------------------------------
# get_dependencies_command
# ---------------------------------------------------------------------------


def test_get_dependencies_command_single_index_url_quiet():
    result = utils.get_dependencies_command(
        runtime_packages=["torch"],
        pip_index_urls=["https://pypi.org/simple"],
        trainer_packages=None,
    )
    assert "--quiet" in result
    assert "--index-url https://pypi.org/simple" in result
    assert "--extra-index-url" not in result
    assert '"torch"' in result


def test_get_dependencies_command_multiple_index_urls_not_quiet():
    result = utils.get_dependencies_command(
        runtime_packages=["torch"],
        pip_index_urls=["https://pypi.org/simple", "https://extra.example.com"],
        trainer_packages=["numpy"],
        quiet=False,
    )
    assert "--quiet" not in result
    assert "--index-url https://pypi.org/simple" in result
    assert "--extra-index-url https://extra.example.com" in result
    assert '"torch" "numpy"' in result


# ---------------------------------------------------------------------------
# get_command_using_train_func
# ---------------------------------------------------------------------------


def test_get_command_using_train_func_no_trainer_raises():
    runtime = _make_runtime(has_trainer=False)
    with pytest.raises(ValueError, match="Runtime must have a trainer"):
        utils.get_command_using_train_func(
            runtime=runtime,
            train_func=dummy_training_function,
            train_func_parameters=None,
            venv_dir="/tmp/venv",
            train_job_name="job",
        )


def test_get_command_using_train_func_without_parameters(tmp_path):
    runtime = _make_runtime()
    entrypoint = utils.get_command_using_train_func(
        runtime=runtime,
        train_func=dummy_training_function,
        train_func_parameters=None,
        venv_dir=str(tmp_path),
        train_job_name="job1",
    )
    func_file = tmp_path / "train_job1.py"
    assert func_file.exists()
    content = func_file.read_text()
    assert "def dummy_training_function" in content
    assert content.strip().endswith("dummy_training_function()")
    assert str(func_file) in entrypoint
    assert "python" in entrypoint


def test_get_command_using_train_func_with_parameters(tmp_path):
    runtime = _make_runtime()
    utils.get_command_using_train_func(
        runtime=runtime,
        train_func=dummy_training_function_with_args,
        train_func_parameters={"lr": 0.01, "epochs": 3},
        venv_dir=str(tmp_path),
        train_job_name="job2",
    )
    func_file = tmp_path / "train_job2.py"
    content = func_file.read_text()
    assert "dummy_training_function_with_args({'lr': 0.01, 'epochs': 3})" in content


# ---------------------------------------------------------------------------
# get_cleanup_venv_script
# ---------------------------------------------------------------------------


def test_get_cleanup_venv_script_disabled_returns_blank():
    assert utils.get_cleanup_venv_script(venv_dir="/tmp/venv", cleanup_venv=False) == "\n"


def test_get_cleanup_venv_script_enabled_includes_venv_dir():
    result = utils.get_cleanup_venv_script(venv_dir="/tmp/venv", cleanup_venv=True)
    assert "/tmp/venv" in result
    assert "rm -rf" in result


# ---------------------------------------------------------------------------
# get_local_train_job_script
# ---------------------------------------------------------------------------


def test_get_local_train_job_script_no_python_binary_raises():
    runtime = _make_local_runtime()
    trainer = _make_custom_trainer()
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(ValueError, match="No python executable found"),
    ):
        utils.get_local_train_job_script(
            train_job_name="job",
            venv_dir="/tmp/venv",
            trainer=trainer,
            runtime=runtime,
        )


def test_get_local_train_job_script_falls_back_to_python3():
    runtime = _make_local_runtime()
    trainer = _make_custom_trainer()
    with (
        patch("shutil.which", side_effect=[None, "/usr/bin/python3"]),
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_command_using_train_func",
            return_value="ENTRYPOINT_SCRIPT",
        ),
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_cleanup_venv_script",
            return_value="CLEANUP_SCRIPT",
        ),
    ):
        shell, flag, script = utils.get_local_train_job_script(
            train_job_name="job",
            venv_dir="/tmp/venv",
            trainer=trainer,
            runtime=runtime,
        )
    assert shell == "bash"
    assert flag == "-c"
    assert "/usr/bin/python3" in script


def test_get_local_train_job_script_invalid_trainer_type_raises():
    bad_trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=TORCH_FRAMEWORK_TYPE,
        image=LOCAL_RUNTIME_IMAGE,
    )
    runtime = types.Runtime(
        name="test-runtime", trainer=bad_trainer, kind=types.RuntimeKind.TRAINING_RUNTIME
    )
    trainer = _make_custom_trainer()
    with (
        patch("shutil.which", return_value="/usr/bin/python"),
        pytest.raises(ValueError, match="Invalid Runtime Trainer type"),
    ):
        utils.get_local_train_job_script(
            train_job_name="job",
            venv_dir="/tmp/venv",
            trainer=trainer,
            runtime=runtime,
        )


def test_get_local_train_job_script_without_packages_to_install():
    runtime = _make_local_runtime()
    trainer = _make_custom_trainer(packages_to_install=None)
    with (
        patch("shutil.which", return_value="/usr/bin/python"),
        patch("kubeflow.trainer.backends.localprocess.utils.get_dependencies_command") as mock_deps,
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_command_using_train_func",
            return_value="ENTRYPOINT_SCRIPT",
        ),
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_cleanup_venv_script",
            return_value="CLEANUP_SCRIPT",
        ),
    ):
        shell, flag, script = utils.get_local_train_job_script(
            train_job_name="job",
            venv_dir="/tmp/venv",
            trainer=trainer,
            runtime=runtime,
        )
    mock_deps.assert_not_called()
    assert "ENTRYPOINT_SCRIPT" in script
    assert "CLEANUP_SCRIPT" in script
    assert "/tmp/venv" in script


def test_get_local_train_job_script_with_packages_to_install():
    runtime = _make_local_runtime(packages=["torch"])
    trainer = _make_custom_trainer(packages_to_install=["scipy"])
    with (
        patch("shutil.which", return_value="/usr/bin/python"),
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_dependencies_command",
            return_value="DEPENDENCIES_SCRIPT",
        ) as mock_deps,
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_command_using_train_func",
            return_value="ENTRYPOINT_SCRIPT",
        ),
        patch(
            "kubeflow.trainer.backends.localprocess.utils.get_cleanup_venv_script",
            return_value="CLEANUP_SCRIPT",
        ),
    ):
        shell, flag, script = utils.get_local_train_job_script(
            train_job_name="job",
            venv_dir="/tmp/venv",
            trainer=trainer,
            runtime=runtime,
        )
    mock_deps.assert_called_once_with(
        pip_index_urls=trainer.pip_index_urls,
        runtime_packages=["torch"],
        trainer_packages=["scipy"],
        quiet=False,
    )
    assert "DEPENDENCIES_SCRIPT" in script
