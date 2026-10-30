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

import pytest

from kubeflow.trainer.backends.localprocess import utils
from kubeflow.trainer.backends.localprocess.constants import LOCAL_RUNTIME_IMAGE
from kubeflow.trainer.backends.localprocess.types import LocalRuntimeTrainer
from kubeflow.trainer.test.common import SUCCESS, TestCase
from kubeflow.trainer.types import types


def _build_runtime() -> types.Runtime:
    return types.Runtime(
        name="test-runtime",
        trainer=LocalRuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image=LOCAL_RUNTIME_IMAGE,
        ),
        kind=types.RuntimeKind.TRAINING_RUNTIME,
    )


def train_with_kwargs(lr: float, num_epochs: int):
    print(f"Training with lr={lr}, num_epochs={num_epochs}")


def train_with_no_args():
    print("Training with no arguments")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train func with parameters is called with kwargs unpacking",
            expected_status=SUCCESS,
            config={
                "train_func": train_with_kwargs,
                "train_func_parameters": {"lr": 0.01, "num_epochs": 5},
            },
            expected_output=f"{train_with_kwargs.__name__}(**{{'lr': 0.01, 'num_epochs': 5}})",
        ),
        TestCase(
            name="train func without parameters is called with no arguments",
            expected_status=SUCCESS,
            config={
                "train_func": train_with_no_args,
                "train_func_parameters": None,
            },
            expected_output=f"{train_with_no_args.__name__}()",
        ),
    ],
)
def test_get_command_using_train_func_generates_valid_call(test_case, tmp_path):
    runtime = _build_runtime()

    utils.get_command_using_train_func(
        runtime=runtime,
        train_func=test_case.config["train_func"],
        train_func_parameters=test_case.config["train_func_parameters"],
        venv_dir=str(tmp_path),
        train_job_name="test-job",
    )

    func_file = tmp_path / "train_test-job.py"
    generated_code = func_file.read_text()

    assert test_case.expected_output in generated_code
    # The generated call must be valid Python that we can actually compile and execute,
    # proving the training function receives its arguments as named parameters rather
    # than as a single positional dict.
    namespace: dict = {}
    exec(generated_code, namespace)
