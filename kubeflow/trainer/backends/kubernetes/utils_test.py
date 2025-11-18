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

import kubeflow.trainer.backends.kubernetes.utils as utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


def _build_runtime() -> types.Runtime:
    runtime_trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="torch",
        device="cpu",
        device_count="1",
        image="example.com/image",
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
                "--user torch numpy custom-package ||\n"
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
                "--user torch numpy custom-package ||\n"
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
                "--user torch numpy custom-package ||\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                "--no-warn-script-location --index-url https://pypi.org/simple "
                "--extra-index-url https://private.repo.com/simple "
                "--extra-index-url https://internal.company.com/simple "
                "torch numpy custom-package\n"
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
                f"{constants.DEFAULT_PIP_INDEX_URLS[0]} --user torch numpy ||\n"
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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="DataCacheInitializer with all optional fields",
            expected_status=SUCCESS,
            config={
                "initializer": types.DataCacheInitializer(
                    storage_uri="cache://test_schema/test_table",
                    num_data_nodes=3,
                    metadata_loc="s3://bucket/metadata",
                    head_cpu="1",
                    head_mem="1Gi",
                    worker_cpu="2",
                    worker_mem="2Gi",
                    iam_role="arn:aws:iam::123456789012:role/test-role",
                ),
            },
            expected_output={
                "storage_uri": "cache://test_schema/test_table",
                "env": {
                    "CLUSTER_SIZE": "4",
                    "METADATA_LOC": "s3://bucket/metadata",
                    "HEAD_CPU": "1",
                    "HEAD_MEM": "1Gi",
                    "WORKER_CPU": "2",
                    "WORKER_MEM": "2Gi",
                    "IAM_ROLE": "arn:aws:iam::123456789012:role/test-role",
                },
            },
        ),
        TestCase(
            name="DataCacheInitializer with only required fields",
            expected_status=SUCCESS,
            config={
                "initializer": types.DataCacheInitializer(
                    storage_uri="cache://schema/table",
                    num_data_nodes=2,
                    metadata_loc="s3://bucket/metadata.json",
                ),
            },
            expected_output={
                "storage_uri": "cache://schema/table",
                "env": {
                    "CLUSTER_SIZE": "3",
                    "METADATA_LOC": "s3://bucket/metadata.json",
                },
            },
        ),
        TestCase(
            name="HuggingFaceDatasetInitializer without access token",
            expected_status=SUCCESS,
            config={
                "initializer": types.HuggingFaceDatasetInitializer(
                    storage_uri="hf://datasets/public-dataset",
                ),
            },
            expected_output={
                "storage_uri": "hf://datasets/public-dataset",
                "env": {},
            },
        ),
        TestCase(
            name="S3DatasetInitializer with all optional fields",
            expected_status=SUCCESS,
            config={
                "initializer": types.S3DatasetInitializer(
                    storage_uri="s3://my-bucket/datasets/train",
                    endpoint="https://s3.custom.com",
                    access_key_id="test-access-key",
                    secret_access_key="test-secret-key",
                    region="us-west-2",
                    role_arn="arn:aws:iam::123456789012:role/test-role",
                ),
            },
            expected_output={
                "storage_uri": "s3://my-bucket/datasets/train",
                "env": {
                    "ENDPOINT": "https://s3.custom.com",
                    "ACCESS_KEY_ID": "test-access-key",
                    "SECRET_ACCESS_KEY": "test-secret-key",
                    "REGION": "us-west-2",
                    "ROLE_ARN": "arn:aws:iam::123456789012:role/test-role",
                },
            },
        ),
        TestCase(
            name="Invalid dataset type",
            expected_status=FAILED,
            config={
                "initializer": "invalid_type",
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_dataset_initializer(test_case):
    """Test get_dataset_initializer with various dataset initializer types."""
    print("Executing test:", test_case.name)
    try:
        dataset_initializer = utils.get_dataset_initializer(test_case.config["initializer"])

        assert test_case.expected_status == SUCCESS
        assert dataset_initializer is not None
        assert dataset_initializer.storage_uri == test_case.expected_output["storage_uri"]

        # Check env vars if expected
        expected_env = test_case.expected_output.get("env", {})
        env_dict = {
            env_var.name: env_var.value for env_var in getattr(dataset_initializer, "env", [])
        }
        assert env_dict == expected_env, f"Expected env {expected_env}, got {env_dict}"

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="HuggingFaceModelInitializer with access token and ignore patterns",
            expected_status=SUCCESS,
            config={
                "initializer": types.HuggingFaceModelInitializer(
                    storage_uri="hf://username/my-model",
                    access_token="hf_test_token_789",
                    ignore_patterns=["*.bin", "*.safetensors"],
                ),
            },
            expected_output={
                "storage_uri": "hf://username/my-model",
                "env": {
                    "ACCESS_TOKEN": "hf_test_token_789",
                    "IGNORE_PATTERNS": "*.bin,*.safetensors",
                },
            },
        ),
        TestCase(
            name="HuggingFaceModelInitializer without access token",
            expected_status=SUCCESS,
            config={
                "initializer": types.HuggingFaceModelInitializer(
                    storage_uri="hf://username/public-model",
                ),
            },
            expected_output={
                "storage_uri": "hf://username/public-model",
                "env": {
                    "IGNORE_PATTERNS": ",".join(constants.INITIALIZER_DEFAULT_IGNORE_PATTERNS),
                },
            },
        ),
        TestCase(
            name="S3ModelInitializer with all optional fields",
            expected_status=SUCCESS,
            config={
                "initializer": types.S3ModelInitializer(
                    storage_uri="s3://my-bucket/models/trained-model",
                    endpoint="https://s3.custom.com",
                    access_key_id="test-access-key",
                    secret_access_key="test-secret-key",
                    region="us-east-1",
                    role_arn="arn:aws:iam::123456789012:role/test-role",
                    ignore_patterns=["*.txt", "*.log"],
                ),
            },
            expected_output={
                "storage_uri": "s3://my-bucket/models/trained-model",
                "env": {
                    "ENDPOINT": "https://s3.custom.com",
                    "ACCESS_KEY_ID": "test-access-key",
                    "SECRET_ACCESS_KEY": "test-secret-key",
                    "REGION": "us-east-1",
                    "ROLE_ARN": "arn:aws:iam::123456789012:role/test-role",
                    "IGNORE_PATTERNS": "*.txt,*.log",
                },
            },
        ),
        TestCase(
            name="Invalid model type",
            expected_status=FAILED,
            config={
                "initializer": "invalid_type",
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_model_initializer(test_case):
    """Test get_model_initializer with various model initializer types."""
    print("Executing test:", test_case.name)
    try:
        model_initializer = utils.get_model_initializer(test_case.config["initializer"])

        assert test_case.expected_status == SUCCESS
        assert model_initializer is not None
        assert model_initializer.storage_uri == test_case.expected_output["storage_uri"]

        # Check env vars if expected
        expected_env = test_case.expected_output.get("env", {})
        env_dict = {
            env_var.name: env_var.value for env_var in getattr(model_initializer, "env", [])
        }
        assert env_dict == expected_env, f"Expected env {expected_env}, got {env_dict}"

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
