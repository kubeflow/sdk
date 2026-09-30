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

"""
Unit tests for the LocalProcessBackend utils in the Kubeflow Trainer SDK.
"""

import pytest

from kubeflow.trainer.backends.localprocess.utils import get_install_packages
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="trainer_package_overrides_runtime_package",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch==2.0", "numpy"],
                "trainer_packages": ["Torch>=2.1"],
            },
            expected_output=["numpy", "Torch>=2.1"],
        ),
        TestCase(
            name="multiple_git_urls_are_different_packages",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch"],
                "trainer_packages": [
                    "git+https://github.com/huggingface/transformers",
                    "git+https://github.com/huggingface/peft",
                ],
            },
            expected_output=[
                "torch",
                "git+https://github.com/huggingface/transformers",
                "git+https://github.com/huggingface/peft",
            ],
        ),
        TestCase(
            name="git_url_does_not_override_package_named_git",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["git"],
                "trainer_packages": ["git+https://github.com/huggingface/peft"],
            },
            expected_output=["git", "git+https://github.com/huggingface/peft"],
        ),
        TestCase(
            name="duplicate_trainer_package",
            expected_status=FAILED,
            config={
                "runtime_packages": [],
                "trainer_packages": ["numpy", "NumPy==1.26"],
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_install_packages(test_case):
    """Test get_install_packages()."""
    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            get_install_packages(**test_case.config)
    else:
        assert get_install_packages(**test_case.config) == test_case.expected_output
