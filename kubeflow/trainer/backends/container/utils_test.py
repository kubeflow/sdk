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

from kubeflow.trainer.backends.container import utils as container_utils
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


def simple_train_func():
    print("Training...")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="no packages returns empty string",
            expected_status=SUCCESS,
            config={"packages": None},
            expected_output="",
        ),
        TestCase(
            name="single package with defaults",
            expected_status=SUCCESS,
            config={"packages": ["torchvision"]},
            expected_output=(
                "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1"
                " python -m pip install --no-warn-script-location"
                " --index-url https://pypi.org/simple "
                " --user torchvision >/tmp/pip_install.log 2>&1; then"
                ' echo "Successfully installed Python packages (user): torchvision";'
                " elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1"
                " python -m pip install --no-warn-script-location"
                " --index-url https://pypi.org/simple "
                " torchvision >>/tmp/pip_install.log 2>&1; then"
                ' echo "Successfully installed Python packages (system-wide): torchvision";'
                " else"
                ' echo "ERROR: Failed to install Python packages: torchvision" >&2;'
                " cat /tmp/pip_install.log >&2; exit 1;"
                " fi && "
            ),
        ),
        TestCase(
            name="multiple packages with custom index",
            expected_status=SUCCESS,
            config={
                "packages": ["torchvision", "transformers[torch]"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://download.pytorch.org/whl/cpu",
                ],
            },
            expected_output=(
                "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1"
                " python -m pip install --no-warn-script-location"
                " --index-url https://pypi.org/simple"
                " --extra-index-url https://download.pytorch.org/whl/cpu"
                " --user torchvision 'transformers[torch]'"
                " >/tmp/pip_install.log 2>&1; then"
                ' echo "Successfully installed Python packages (user):'
                " torchvision 'transformers[torch]'\";"
                " elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1"
                " python -m pip install --no-warn-script-location"
                " --index-url https://pypi.org/simple"
                " --extra-index-url https://download.pytorch.org/whl/cpu"
                " torchvision 'transformers[torch]'"
                " >>/tmp/pip_install.log 2>&1; then"
                ' echo "Successfully installed Python packages (system-wide):'
                " torchvision 'transformers[torch]'\";"
                " else"
                ' echo "ERROR: Failed to install Python packages:'
                " torchvision 'transformers[torch]'\" >&2;"
                " cat /tmp/pip_install.log >&2; exit 1;"
                " fi && "
            ),
        ),
    ],
)
def test_build_pip_install_cmd(test_case: TestCase):
    """Test pip install command generation."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(
            func=simple_train_func,
            packages_to_install=test_case.config.get("packages"),
            pip_index_urls=test_case.config.get("pip_index_urls"),
        )
        result = container_utils.build_pip_install_cmd(trainer)

        assert test_case.expected_status == SUCCESS
        assert result == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")
