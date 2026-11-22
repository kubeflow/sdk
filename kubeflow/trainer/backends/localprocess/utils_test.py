# Copyright The Kubeflow Authors.
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

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from kubeflow.trainer.backends.localprocess import constants, utils
from kubeflow.trainer.test.common import TestCase
from kubeflow.trainer.types import types


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="omitted extras", config={"extras": None}, expected_output=["torch"]),
        TestCase(name="empty extras", config={"extras": []}, expected_output=["torch"]),
        TestCase(
            name="additional package",
            config={"extras": ["numpy"]},
            expected_output=["torch", "numpy"],
        ),
        TestCase(
            name="runtime override",
            config={"extras": ["torch==2.5.0"]},
            expected_output=["torch==2.5.0"],
        ),
        TestCase(
            name="only trainer packages",
            config={"runtime_packages": [], "extras": ["numpy"]},
            expected_output=["numpy"],
        ),
        TestCase(
            name="no packages",
            config={"runtime_packages": [], "extras": None},
            expected_output=[],
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_local_script_dependencies(test_case: TestCase, tmp_path: Path) -> None:
    runtime = deepcopy(constants.local_runtimes[0])
    runtime.trainer = utils.get_local_runtime_trainer(
        runtime.name, str(tmp_path), runtime.trainer.framework
    )
    runtime.trainer.packages = test_case.config.get("runtime_packages", ["torch"])
    trainer = types.CustomTrainer(func=lambda: None, packages_to_install=test_case.config["extras"])

    with patch.object(utils, "get_command_using_train_func", return_value="true"):
        script = utils.get_local_train_job_script(
            "test-dependencies", str(tmp_path), trainer, runtime, cleanup_venv=False
        )[2]

    install_lines = [line for line in script.splitlines() if "pip install" in line]
    if test_case.expected_output:
        assert len(install_lines) == 1
        for package in test_case.expected_output:
            assert f'"{package}"' in install_lines[0]
        if "torch" not in test_case.expected_output:
            assert '"torch"' not in install_lines[0]
    else:
        assert not install_lines
