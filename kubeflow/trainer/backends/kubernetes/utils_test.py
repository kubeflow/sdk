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

from kubeflow_trainer_api import models
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
            name="single MIG limit returns device and count",
            expected_status=SUCCESS,
            config={
                "resources": models.IoK8sApiCoreV1ResourceRequirements(
                    limits={
                        "nvidia.com/mig-1g.5gb": models.IoK8sApimachineryPkgApiResourceQuantity(2),
                    }
                )
            },
            expected_output=("mig-1g.5gb", "2.0"),
        ),
        TestCase(
            name="multiple MIG limits are not supported",
            expected_status=FAILED,
            config={
                "resources": models.IoK8sApiCoreV1ResourceRequirements(
                    limits={
                        "nvidia.com/mig-1g.5gb": models.IoK8sApimachineryPkgApiResourceQuantity(1),
                        "nvidia.com/mig-2g.10gb": models.IoK8sApimachineryPkgApiResourceQuantity(1),
                    }
                )
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_container_devices(test_case: TestCase):
    print("Executing test:", test_case.name)
    try:
        device = utils.get_container_devices(test_case.config["resources"])

        assert test_case.expected_status == SUCCESS
        assert device == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="mig alias expands to fully qualified key",
            expected_status=SUCCESS,
            config={
                "resources_per_node": {
                    "MiG-1G.5GB": 2,
                    "cpu": "500m",
                }
            },
            expected_output=models.IoK8sApiCoreV1ResourceRequirements(
                limits={
                    "cpu": models.IoK8sApimachineryPkgApiResourceQuantity("500m"),
                    "nvidia.com/mig-1g.5gb": models.IoK8sApimachineryPkgApiResourceQuantity(2),
                },
                requests={
                    "cpu": models.IoK8sApimachineryPkgApiResourceQuantity("500m"),
                    "nvidia.com/mig-1g.5gb": models.IoK8sApimachineryPkgApiResourceQuantity(2),
                },
            ),
        ),
        TestCase(
            name="gpu and mig together raises error",
            expected_status=FAILED,
            config={"resources_per_node": {"gpu": 1, "mig-1g.5gb": 1}},
            expected_error=ValueError,
        ),
        TestCase(
            name="multiple mig resource types raises error",
            expected_status=FAILED,
            config={
                "resources_per_node": {
                    "mig-1g.5gb": 1,
                    "nvidia.com/mig-2g.10gb": 1,
                }
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="extended resource preserves case",
            expected_status=SUCCESS,
            config={
                "resources_per_node": {
                    "example.com/Capitalized": 1,
                    "CPU": 2,
                    "Memory": "16Gi",
                    "EPHEMERAL-STORAGE": "100Gi",
                }
            },
            expected_output=models.IoK8sApiCoreV1ResourceRequirements(
                limits={
                    "example.com/Capitalized": models.IoK8sApimachineryPkgApiResourceQuantity(1),
                    "cpu": models.IoK8sApimachineryPkgApiResourceQuantity(2),
                    "memory": models.IoK8sApimachineryPkgApiResourceQuantity("16Gi"),
                    "ephemeral-storage": models.IoK8sApimachineryPkgApiResourceQuantity("100Gi"),
                },
                requests={
                    "example.com/Capitalized": models.IoK8sApimachineryPkgApiResourceQuantity(1),
                    "cpu": models.IoK8sApimachineryPkgApiResourceQuantity(2),
                    "memory": models.IoK8sApimachineryPkgApiResourceQuantity("16Gi"),
                    "ephemeral-storage": models.IoK8sApimachineryPkgApiResourceQuantity("100Gi"),
                },
            ),
        ),
        TestCase(
            name="diverse resource types and mixed case standard keys",
            expected_status=SUCCESS,
            config={
                "resources_per_node": {
                    "example.com/test": 1,
                    "Example.com/Custom-NPU": 2,
                    "mEmOrY": "8Gi",
                    "STORAGE": "100Gi",
                }
            },
            expected_output=models.IoK8sApiCoreV1ResourceRequirements(
                limits={
                    "example.com/test": models.IoK8sApimachineryPkgApiResourceQuantity(1),
                    "Example.com/Custom-NPU": models.IoK8sApimachineryPkgApiResourceQuantity(2),
                    "memory": models.IoK8sApimachineryPkgApiResourceQuantity("8Gi"),
                    "ephemeral-storage": models.IoK8sApimachineryPkgApiResourceQuantity("100Gi"),
                },
                requests={
                    "example.com/test": models.IoK8sApimachineryPkgApiResourceQuantity(1),
                    "Example.com/Custom-NPU": models.IoK8sApimachineryPkgApiResourceQuantity(2),
                    "memory": models.IoK8sApimachineryPkgApiResourceQuantity("8Gi"),
                    "ephemeral-storage": models.IoK8sApimachineryPkgApiResourceQuantity("100Gi"),
                },
            ),
        ),
    ],
)
def test_get_resources_per_node(test_case: TestCase):
    print("Executing test:", test_case.name)
    try:
        resources = utils.get_resources_per_node(test_case.config["resources_per_node"])

        assert test_case.expected_status == SUCCESS
        assert resources == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


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
                "fi\n\n\n"
                "PACKAGES=(torch numpy custom-package)\n"
                "PIP_OPTS=(--index-url https://pypi.org/simple --extra-index-url https://private.repo.com/simple --extra-index-url https://internal.company.com/simple)\n"
                'LOG_FILE="pip_install.log"\n'
                'rm -f "$LOG_FILE"\n'
                "\n"
                "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" --user "${PACKAGES[@]}" >"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" "${PACKAGES[@]}" >>"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "else\n"
                '    echo "ERROR: Failed to install Python packages: ${PACKAGES[*]}" >&2\n'
                '    cat "$LOG_FILE" >&2\n'
                "    exit 1\n"
                "fi\n\n"
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
                "fi\n\n\n"
                "PACKAGES=(torch numpy custom-package)\n"
                "PIP_OPTS=(--index-url https://pypi.org/simple)\n"
                'LOG_FILE="pip_install.log"\n'
                'rm -f "$LOG_FILE"\n'
                "\n"
                "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" --user "${PACKAGES[@]}" >"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" "${PACKAGES[@]}" >>"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "else\n"
                '    echo "ERROR: Failed to install Python packages: ${PACKAGES[*]}" >&2\n'
                '    cat "$LOG_FILE" >&2\n'
                "    exit 1\n"
                "fi\n\n"
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
                "fi\n\n\n"
                "PACKAGES=(torch numpy custom-package)\n"
                "PIP_OPTS=(--index-url https://pypi.org/simple --extra-index-url https://private.repo.com/simple --extra-index-url https://internal.company.com/simple)\n"
                'LOG_FILE="pip_install.log"\n'
                'rm -f "$LOG_FILE"\n'
                "\n"
                "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" --user "${PACKAGES[@]}" >"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" "${PACKAGES[@]}" >>"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "else\n"
                '    echo "ERROR: Failed to install Python packages: ${PACKAGES[*]}" >&2\n'
                '    cat "$LOG_FILE" >&2\n'
                "    exit 1\n"
                "fi\n\n"
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
                "fi\n\n\n"
                "PACKAGES=(torch numpy)\n"
                "PIP_OPTS=(--index-url https://pypi.org/simple)\n"
                'LOG_FILE="pip_install.log"\n'
                'rm -f "$LOG_FILE"\n'
                "\n"
                "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" --user "${PACKAGES[@]}" >"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" "${PACKAGES[@]}" >>"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "else\n"
                '    echo "ERROR: Failed to install Python packages: ${PACKAGES[*]}" >&2\n'
                '    cat "$LOG_FILE" >&2\n'
                "    exit 1\n"
                "fi\n\n"
            ),
        ),
        TestCase(
            name="packages with extras notation",
            config={
                "packages_to_install": ["datasets", "transformers[torch]", "cloudpathlib[all]"],
                "pip_index_urls": constants.DEFAULT_PIP_INDEX_URLS,
                "is_mpi": False,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n\n"
                "PACKAGES=(datasets 'transformers[torch]' 'cloudpathlib[all]')\n"
                "PIP_OPTS=(--index-url https://pypi.org/simple)\n"
                'LOG_FILE="pip_install.log"\n'
                'rm -f "$LOG_FILE"\n'
                "\n"
                "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" --user "${PACKAGES[@]}" >"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                '    --no-warn-script-location "${PIP_OPTS[@]}" "${PACKAGES[@]}" >>"$LOG_FILE" 2>&1; then\n'
                '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                "else\n"
                '    echo "ERROR: Failed to install Python packages: ${PACKAGES[*]}" >&2\n'
                '    cat "$LOG_FILE" >&2\n'
                "    exit 1\n"
                "fi\n\n"
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
        TestCase(
            name="with packages to install",
            expected_status=SUCCESS,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": None,
                "runtime": _build_runtime(),
                "packages_to_install": ["requests"],
            },
            expected_output=[
                "bash",
                "-c",
                (
                    '\nif ! [ -x "$(command -v pip)" ]; then\n'
                    "    python -m ensurepip || python -m ensurepip --user || "
                    "apt-get install python-pip\n"
                    "fi\n\n\n"
                    "PACKAGES=(requests)\n"
                    "PIP_OPTS=(--index-url https://pypi.org/simple)\n"
                    'LOG_FILE="pip_install.log"\n'
                    'rm -f "$LOG_FILE"\n'
                    "\n"
                    "if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                    '    --no-warn-script-location "${PIP_OPTS[@]}" --user "${PACKAGES[@]}" >"$LOG_FILE" 2>&1; then\n'
                    '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                    "elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --quiet \\\n"
                    '    --no-warn-script-location "${PIP_OPTS[@]}" "${PACKAGES[@]}" >>"$LOG_FILE" 2>&1; then\n'
                    '    echo "Successfully installed Python packages: ${PACKAGES[*]}"\n'
                    "else\n"
                    '    echo "ERROR: Failed to install Python packages: ${PACKAGES[*]}" >&2\n'
                    '    cat "$LOG_FILE" >&2\n'
                    "    exit 1\n"
                    "fi\n\n"
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda: print("Hello World")),\n\n'
                    "<lambda>()\n\n"
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
            packages_to_install=test_case.config.get("packages_to_install", []),
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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="lora_dropout=0.0 is not silently dropped",
            expected_status=SUCCESS,
            config={
                "peft_config": types.LoraConfig(lora_dropout=0.0),
            },
            expected_output=[
                "model.lora_dropout=0.0",
                "model.lora_attn_modules=[q_proj,v_proj,output_proj]",
            ],
        ),
        TestCase(
            name="apply_lora_to_mlp=False is not silently dropped",
            expected_status=SUCCESS,
            config={
                "peft_config": types.LoraConfig(apply_lora_to_mlp=False),
            },
            expected_output=[
                "model.apply_lora_to_mlp=False",
                "model.lora_attn_modules=[q_proj,v_proj,output_proj]",
            ],
        ),
        TestCase(
            name="standard lora config with positive values",
            expected_status=SUCCESS,
            config={
                "peft_config": types.LoraConfig(lora_rank=8, lora_alpha=16, lora_dropout=0.1),
            },
            expected_output=[
                "model.lora_rank=8",
                "model.lora_alpha=16",
                "model.lora_dropout=0.1",
                "model.lora_attn_modules=[q_proj,v_proj,output_proj]",
            ],
        ),
        TestCase(
            name="invalid peft config type raises ValueError",
            expected_status=FAILED,
            config={
                "peft_config": "invalid",
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_args_from_peft_config(test_case: TestCase):
    print("Executing test:", test_case.name)
    try:
        args = utils.get_args_from_peft_config(test_case.config["peft_config"])

        assert test_case.expected_status == SUCCESS
        assert args == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train_on_input=False is not silently dropped",
            expected_status=SUCCESS,
            config={
                "dataset_preprocess_config": types.TorchTuneInstructDataset(
                    train_on_input=False,
                ),
            },
            expected_output=[
                f"dataset={constants.TORCH_TUNE_INSTRUCT_DATASET}",
                "dataset.train_on_input=False",
            ],
        ),
        TestCase(
            name="train_on_input=True is included",
            expected_status=SUCCESS,
            config={
                "dataset_preprocess_config": types.TorchTuneInstructDataset(
                    train_on_input=True,
                ),
            },
            expected_output=[
                f"dataset={constants.TORCH_TUNE_INSTRUCT_DATASET}",
                "dataset.train_on_input=True",
            ],
        ),
    ],
)
def test_get_args_from_dataset_preprocess_config(test_case: TestCase):
    print("Executing test:", test_case.name)
    try:
        args = utils.get_args_from_dataset_preprocess_config(
            test_case.config["dataset_preprocess_config"]
        )

        assert test_case.expected_status == SUCCESS
        assert args == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")
