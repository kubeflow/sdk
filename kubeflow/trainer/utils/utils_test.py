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

from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types
from kubeflow.trainer.utils import utils


def _build_runtime() -> types.Runtime:
    runtime_trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="torch",
        device="cpu",
        device_count="1",
    )
    runtime_trainer.set_command(constants.DEFAULT_COMMAND)
    return types.Runtime(name="test-runtime", trainer=runtime_trainer)


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
                    "https://internal.company.com/simple",
                ],
                "is_mpi": False,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                "--no-warn-script-location --index-url https://pypi.org/simple "
                "--extra-index-url https://private.repo.com/simple "
                "--extra-index-url https://internal.company.com/simple "
                "torch numpy custom-package\n"
            ),
        ),
        TestCase(
            name="single pip index URL (backward compatibility)",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": ["https://pypi.org/simple"],
                "is_mpi": False,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                "--no-warn-script-location --index-url https://pypi.org/simple "
                "torch numpy custom-package\n"
            ),
        ),
        TestCase(
            name="multiple pip index URLs with MPI",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://private.repo.com/simple",
                    "https://internal.company.com/simple",
                ],
                "is_mpi": True,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                "--no-warn-script-location --index-url https://pypi.org/simple "
                "--extra-index-url https://private.repo.com/simple "
                "--extra-index-url https://internal.company.com/simple "
                "--user torch numpy custom-package\n"
            ),
        ),
        TestCase(
            name="default pip index URLs",
            config={
                "packages_to_install": ["torch", "numpy"],
                "pip_index_urls": constants.DEFAULT_PIP_INDEX_URLS,
                "is_mpi": False,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                f"--no-warn-script-location --index-url "
                f"{constants.DEFAULT_PIP_INDEX_URLS[0]} torch numpy\n"
            ),
        ),
    ],
)
def test_get_script_for_python_packages(test_case):
    """Test get_script_for_python_packages with various configurations."""

    script = utils.get_script_for_python_packages(
        packages_to_install=test_case.config["packages_to_install"],
        pip_index_urls=test_case.config["pip_index_urls"],
        is_mpi=test_case.config["is_mpi"],
    )

    assert test_case.expected_output == script


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="with args dict always unpacks kwargs",
            expected_status=SUCCESS,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": {"batch_size": 128, "learning_rate": 0.001, "epochs": 20},
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda: print("Hello World")),\n\n'
                    "<lambda>(**{'batch_size': 128, 'learning_rate': 0.001, 'epochs': 20})\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
        TestCase(
            name="without args calls function with no params",
            expected_status=SUCCESS,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": None,
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda: print("Hello World")),\n\n'
                    "<lambda>()\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
        TestCase(
            name="raises when runtime has no trainer",
            expected_status=FAILED,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": None,
                "runtime": types.Runtime(name="no-trainer", trainer=None),
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="raises when train_func is not callable",
            expected_status=FAILED,
            config={
                "func": "not callable",
                "func_args": None,
                "runtime": _build_runtime(),
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="single dict param also unpacks kwargs",
            expected_status=SUCCESS,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": {"a": 1, "b": 2},
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda: print("Hello World")),\n\n'
                    "<lambda>(**{'a': 1, 'b': 2})\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
        TestCase(
            name="multi-param function uses kwargs-unpacking",
            expected_status=SUCCESS,
            config={
                "func": (lambda **kwargs: "ok"),
                "func_args": {"a": 3, "b": "hi", "c": 0.2},
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda **kwargs: "ok"),\n\n'
                    "<lambda>(**{'a': 3, 'b': 'hi', 'c': 0.2})\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
    ],
)
def test_get_command_using_train_func(test_case: TestCase):
    print("Executing test:", test_case.name)

    try:
        command = utils.get_command_using_train_func(
            runtime=test_case.config["runtime"],
            train_func=test_case.config.get("func"),
            train_func_parameters=test_case.config.get("func_args"),
            pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
            packages_to_install=[],
        )

        assert test_case.expected_status == SUCCESS
        assert command == test_case.expected_output

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_get_dataset_initializer():
    """Test get_dataset_initializer uses DataCacheInitializer optional fields as env vars."""
    datacache_initializer = types.DataCacheInitializer(
        storage_uri="cache://test_schema/test_table",
        num_data_nodes=3,
        metadata_loc="s3://bucket/metadata",
        head_cpu="1",
        head_mem="1Gi",
        worker_cpu="2",
        worker_mem="2Gi",
        iam_role="arn:aws:iam::123456789012:role/test-role",
    )

    dataset_initializer = utils.get_dataset_initializer(datacache_initializer)

    assert dataset_initializer is not None
    assert dataset_initializer.env is not None
    env_dict = {env_var.name: env_var.value for env_var in dataset_initializer.env}

    # Check CLUSTER_SIZE is present from num_data_nodes
    assert env_dict["CLUSTER_SIZE"] == "4"

    # Check METADATA_LOC is present from metadata_loc
    assert env_dict["METADATA_LOC"] == "s3://bucket/metadata"

    # Check all optional fields are present as uppercase env vars
    assert env_dict["HEAD_CPU"] == "1"
    assert env_dict["HEAD_MEM"] == "1Gi"
    assert env_dict["WORKER_CPU"] == "2"
    assert env_dict["WORKER_MEM"] == "2Gi"
    assert env_dict["IAM_ROLE"] == "arn:aws:iam::123456789012:role/test-role"
