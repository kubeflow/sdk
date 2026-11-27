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

"""Unit tests for Kubeflow Optimizer search types."""

import pytest

from kubeflow.optimizer.types.search_types import Search


def test_uniform_valid() -> None:
    spec = Search.uniform(0.0, 1.0)
    assert spec.feasible_space.min == "0.0"
    assert spec.feasible_space.max == "1.0"


@pytest.mark.parametrize("min_val,max_val", [(1.0, 0.1), (0.5, 0.5), (2.0, 1.0)])
def test_uniform_invalid_range_raises(min_val: float, max_val: float) -> None:
    with pytest.raises(ValueError, match="min must be strictly less than max"):
        Search.uniform(min_val, max_val)


def test_loguniform_valid() -> None:
    spec = Search.loguniform(0.1, 10.0)
    assert spec.feasible_space.min == "0.1"
    assert spec.feasible_space.max == "10.0"


@pytest.mark.parametrize("min_val", [0, -1.0, 0.0])
def test_loguniform_non_positive_min_raises(min_val: float) -> None:
    with pytest.raises(ValueError, match="min must be strictly greater than 0"):
        Search.loguniform(min_val, 10.0)


@pytest.mark.parametrize("min_val,max_val", [(1.0, 0.1), (5.0, 5.0)])
def test_loguniform_invalid_range_raises(min_val: float, max_val: float) -> None:
    with pytest.raises(ValueError, match="min must be strictly less than max"):
        Search.loguniform(min_val, max_val)


def test_choice_valid() -> None:
    spec = Search.choice(["a", "b"])
    assert spec.feasible_space.list == ["a", "b"]


def test_choice_empty_raises() -> None:
    with pytest.raises(ValueError, match="must have at least 1 items"):
        Search.choice([])


def test_choice_single_value_works() -> None:
    spec = Search.choice(["only"])
    assert spec.feasible_space.list == ["only"]
