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

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from kubeflow.trainer.utils import utils
from kubeflow.trainer.constants import constants


@dataclass
class TestCase:
    name: str
    config: Dict[str, Any]
    expected_output: Dict[str, Any]
    __test__ = False


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="multiple pip index URLs",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://private.repo.com/simple",
                    "https://internal.company.com/simple"
                ],
                "is_mpi": False
            },
            expected_output={
                "contains": [
                    "--index-url https://pypi.org/simple",
                    "--extra-index-url https://private.repo.com/simple",
                    "--extra-index-url https://internal.company.com/simple",
                    "torch numpy custom-package"
                ],
                "not_contains": []
            }
        ),
        TestCase(
            name="single pip index URL (backward compatibility)",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": ["https://pypi.org/simple"],
                "is_mpi": False
            },
            expected_output={
                "contains": [
                    "--index-url https://pypi.org/simple",
                    "torch numpy custom-package"
                ],
                "not_contains": [
                    "--extra-index-url"
                ]
            }
        ),
        TestCase(
            name="multiple pip index URLs with MPI",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://private.repo.com/simple",
                    "https://internal.company.com/simple"
                ],
                "is_mpi": True
            },
            expected_output={
                "contains": [
                    "--index-url https://pypi.org/simple",
                    "--extra-index-url https://private.repo.com/simple",
                    "--extra-index-url https://internal.company.com/simple",
                    "--user",
                    "torch numpy custom-package"
                ],
                "not_contains": []
            }
        ),
        TestCase(
            name="default pip index URLs",
            config={
                "packages_to_install": ["torch", "numpy"],
                "pip_index_urls": constants.DEFAULT_PIP_INDEX_URLS,
                "is_mpi": False
            },
            expected_output={
                "contains": [
                    f"--index-url {constants.DEFAULT_PIP_INDEX_URLS[0]}",
                    "torch numpy"
                ],
                "not_contains": []
            }
        ),
    ],
)
def test_get_script_for_python_packages(test_case):
    """Test get_script_for_python_packages with various configurations."""

    script = utils.get_script_for_python_packages(
        packages_to_install=test_case.config["packages_to_install"],
        pip_index_urls=test_case.config["pip_index_urls"],
        is_mpi=test_case.config["is_mpi"]
    )

    # Verify the script contains expected elements
    for expected_text in test_case.expected_output["contains"]:
        assert expected_text in script, (
            f"Expected '{expected_text}' to be in script"
        )

    # Verify the script does not contain unexpected elements
    for unexpected_text in test_case.expected_output["not_contains"]:
        assert unexpected_text not in script, (
            f"Expected '{unexpected_text}' to NOT be in script"
        )
