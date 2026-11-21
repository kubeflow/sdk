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

import os
import shutil
import subprocess

import pytest

from kubeflow.test.common import SUCCESS, TestCase
from kubeflow.trainer.backends.localprocess.utils import get_dependencies_command


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="plain packages",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch"],
                "pip_index_urls": ["https://pypi.org/simple"],
                "trainer_packages": ["numpy==2.0.0"],
            },
            expected_output={
                "index_urls": ["https://pypi.org/simple"],
                "packages": ["torch", "numpy==2.0.0"],
            },
        ),
        TestCase(
            name="package with environment marker keeps its quotes",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch"],
                "pip_index_urls": ["https://pypi.org/simple"],
                "trainer_packages": ['six; python_version >= "3.8"'],
            },
            expected_output={
                "index_urls": ["https://pypi.org/simple"],
                "packages": ["torch", 'six; python_version >= "3.8"'],
            },
        ),
        TestCase(
            name="shell special characters are not expanded",
            expected_status=SUCCESS,
            config={
                "runtime_packages": [],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://example.com/simple?token=a&b=$HOME",
                ],
                "trainer_packages": ["transformers[torch]", "pkg @ https://example.com/$(id).whl"],
            },
            expected_output={
                "index_urls": [
                    "https://pypi.org/simple",
                    "https://example.com/simple?token=a&b=$HOME",
                ],
                "packages": ["transformers[torch]", "pkg @ https://example.com/$(id).whl"],
            },
        ),
    ],
)
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_get_dependencies_command(test_case, tmp_path):
    """Test that packages and index URLs reach pip unchanged when bash runs the command."""
    print("Executing test:", test_case.name)

    # Fake pip that prints every argument it receives on its own line.
    fake_pip = tmp_path / "pip"
    fake_pip.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_pip.chmod(0o755)

    command = get_dependencies_command(**test_case.config)
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        check=True,
    )
    args = result.stdout.splitlines()

    index_urls = [
        args[i + 1] for i, a in enumerate(args) if a in ("--index-url", "--extra-index-url")
    ]
    assert index_urls == test_case.expected_output["index_urls"]
    num_packages = len(test_case.expected_output["packages"])
    assert args[-num_packages:] == test_case.expected_output["packages"]

    print("test execution complete")
