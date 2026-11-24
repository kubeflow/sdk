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

import ast

from kubeflow_katib_api import models
import pytest

from kubeflow.optimizer.backends.kubernetes.utils import (
    convert_value,
    get_trial_parameter_func_arg,
)
from kubeflow.optimizer.types.search_types import Search
from kubeflow.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.backends.kubernetes import utils as trainer_utils
from kubeflow.trainer.constants import constants as trainer_constants
from kubeflow.trainer.types.types import Runtime, RuntimeKind, RuntimeTrainer, TrainerType


def train_func(value):
    pass


def get_value_passed_to_train_func(
    parameter: models.V1beta1ParameterSpec,
    katib_value: str,
) -> object:
    """Render the training script for the parameter and read the value a Trial passes."""
    runtime_trainer = RuntimeTrainer(
        trainer_type=TrainerType.CUSTOM_TRAINER,
        framework="torch",
        image="image",
    )
    runtime_trainer.set_command(trainer_constants.DEFAULT_COMMAND)
    runtime = Runtime(
        name="runtime",
        trainer=runtime_trainer,
        kind=RuntimeKind.CLUSTER_TRAINING_RUNTIME,
    )

    command = trainer_utils.get_command_using_train_func(
        runtime,
        train_func,
        {"value": get_trial_parameter_func_arg("value", parameter)},
        [],
        None,
    )

    # Katib replaces the placeholder in the Trial spec with the assigned value as plain text.
    script = "\n".join(command).replace("${trialParameters.value}", katib_value)
    func_call = script[script.rindex("train_func(**") :].splitlines()[0]
    func_kwargs = func_call.removeprefix("train_func(**").removesuffix(")")

    return ast.literal_eval(func_kwargs)["value"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="int", config={"args": ("42", int)}, expected_output=42),
        TestCase(name="float", config={"args": ("3.14", float)}, expected_output=3.14),
        TestCase(name="bool-1", config={"args": ("1", bool)}, expected_output=True),
        TestCase(name="bool-0", config={"args": ("0", bool)}, expected_output=False),
        TestCase(name="bool-true", config={"args": ("true", bool)}, expected_output=True),
        TestCase(name="bool-false", config={"args": ("false", bool)}, expected_output=False),
        TestCase(name="unhandled-type", config={"args": ("hello", list)}, expected_output="hello"),
        TestCase(name="single-type-union", config={"args": ("42", int | None)}, expected_output=42),
        TestCase(
            name="multi-type-union", config={"args": ("42", int | str | None)}, expected_output="42"
        ),
    ],
)
def test_convert_value(test_case: TestCase):
    """Test convert_value handles both basic types and T | None syntax."""
    print("Executing test:", test_case.name)
    try:
        result = convert_value(*test_case.config["args"])
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert isinstance(e, test_case.expected_error)
    else:
        assert test_case.expected_status == SUCCESS
        assert result == test_case.expected_output
        assert isinstance(result, type(test_case.expected_output))
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="uniform value is passed as float",
            config={"parameter": Search.uniform(0.1, 0.9), "katib_value": "0.4567"},
            expected_output=0.4567,
        ),
        TestCase(
            name="loguniform value in scientific notation is passed as float",
            config={"parameter": Search.loguniform(1e-5, 1e-1), "katib_value": "3.2e-05"},
            expected_output=3.2e-05,
        ),
        TestCase(
            name="int choice is passed as int",
            config={"parameter": Search.choice([16, 32]), "katib_value": "16"},
            expected_output=16,
        ),
        TestCase(
            name="float choice is passed as float",
            config={"parameter": Search.choice([0.1, 0.5]), "katib_value": "0.5"},
            expected_output=0.5,
        ),
        TestCase(
            name="bool choice is passed as bool",
            config={"parameter": Search.choice([True, False]), "katib_value": "False"},
            expected_output=False,
        ),
        TestCase(
            name="string choice is passed as string",
            config={"parameter": Search.choice(["adam", "sgd"]), "katib_value": "adam"},
            expected_output="adam",
        ),
        TestCase(
            name="numeric string choice stays a string",
            config={"parameter": Search.choice(["1", "2"]), "katib_value": "1"},
            expected_output="1",
        ),
        TestCase(
            name="string choice with leading zeros stays a string",
            config={"parameter": Search.choice(["007", "010"]), "katib_value": "007"},
            expected_output="007",
        ),
        TestCase(
            name="string choice that looks like a float name stays a string",
            config={"parameter": Search.choice(["nan", "inf"]), "katib_value": "nan"},
            expected_output="nan",
        ),
        TestCase(
            name="mixed choice is passed as string",
            config={"parameter": Search.choice([16, "adam"]), "katib_value": "16"},
            expected_output="16",
        ),
        TestCase(
            name="non-finite float choice is passed as string",
            config={"parameter": Search.choice([float("inf"), 1.0]), "katib_value": "inf"},
            expected_output="inf",
        ),
        TestCase(
            name="categorical spec created without Search.choice is passed as string",
            config={
                "parameter": models.V1beta1ParameterSpec(
                    parameterType="categorical",
                    feasibleSpace=models.V1beta1FeasibleSpace(list=["16", "32"]),
                ),
                "katib_value": "16",
            },
            expected_output="16",
        ),
        TestCase(
            name="string choice with a single quote raises ValueError",
            expected_status=FAILED,
            config={"parameter": Search.choice(["it's"]), "katib_value": "it's"},
            expected_error=ValueError,
        ),
        TestCase(
            name="string choice with a backslash raises ValueError",
            expected_status=FAILED,
            config={"parameter": Search.choice(["a\\b"]), "katib_value": "a\\b"},
            expected_error=ValueError,
        ),
        TestCase(
            name="string choice with a line break raises ValueError",
            expected_status=FAILED,
            config={"parameter": Search.choice(["a\nb"]), "katib_value": "a\nb"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_trial_parameter_func_arg(test_case: TestCase):
    """Test the value and type that a Trial passes to the training function."""
    print("Executing test:", test_case.name)
    try:
        result = get_value_passed_to_train_func(**test_case.config)
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    else:
        assert test_case.expected_status == SUCCESS
        assert result == test_case.expected_output
        assert type(result) is type(test_case.expected_output)
    print("test execution complete")
