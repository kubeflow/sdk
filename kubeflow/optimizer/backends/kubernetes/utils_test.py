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

from kubeflow_katib_api import models
import pytest

from kubeflow.optimizer.backends.kubernetes.utils import convert_value
from kubeflow.test.common import FAILED, SUCCESS, TestCase


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


def test_trial_parameter_placeholder_repr_unquoted():
    """Bare placeholders omit quotes for numeric Katib substitution."""
    placeholder = TrialParameterPlaceholder("learning_rate", quoted=False)
    assert repr(placeholder) == "${trialParameters.learning_rate}"
    assert f"**{{'learning_rate': {placeholder!r}}}" == (
        "**{'learning_rate': ${trialParameters.learning_rate}}"
    )


def test_trial_parameter_placeholder_repr_quoted():
    """Quoted placeholders keep string quotes for categorical substitution."""
    placeholder = TrialParameterPlaceholder("optimizer", quoted=True)
    assert repr(placeholder) == "'${trialParameters.optimizer}'"
    rendered = f"**{{'optimizer': {placeholder!r}}}"
    assert rendered == "**{'optimizer': '${trialParameters.optimizer}'}"


@pytest.mark.parametrize(
    "param_spec, expected_quoted",
    [
        (Search.uniform(0.001, 0.1), False),
        (Search.loguniform(1e-5, 1e-1), False),
        (Search.choice([10, 20, 30]), False),
        (Search.choice([0.1, 0.2]), False),
        (Search.choice(["adam", "sgd"]), True),
        (Search.choice([16, "adam"]), True),
        (
            models.V1beta1ParameterSpec(
                parameterType="unknown",
                feasibleSpace=models.V1beta1FeasibleSpace(min="1", max="2"),
            ),
            True,
        ),
    ],
)
def test_should_quote_trial_parameter(param_spec, expected_quoted):
    """Quoting follows double vs numeric/string categorical rules."""
    assert should_quote_trial_parameter(param_spec) is expected_quoted


def test_search_choice_rejects_unsafe_string_values():
    """String choices with quotes/backslashes/newlines are rejected early."""
    with pytest.raises(ValueError, match="break generated trial scripts"):
        Search.choice(["it's"])
    with pytest.raises(ValueError, match="break generated trial scripts"):
        Search.choice(["a\\b"])
    with pytest.raises(ValueError, match="break generated trial scripts"):
        Search.choice(["line1\nline2"])


def test_search_choice_allows_safe_values():
    """Numeric and safe string choices remain valid."""
    numeric = Search.choice([10, 20, 30])
    assert numeric.parameter_type == constants.CATEGORICAL_PARAMETERS
    assert numeric.feasible_space.list == ["10", "20", "30"]

    strings = Search.choice(["adam", "sgd"])
    assert strings.feasible_space.list == ["adam", "sgd"]
