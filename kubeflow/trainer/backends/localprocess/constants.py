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

import re
import textwrap

import kubeflow.common.constants as common_constants
from kubeflow.trainer.backends.localprocess import types
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types as base_types

TORCH_FRAMEWORK_TYPE = "torch"

# Image name for the local runtime.
LOCAL_RUNTIME_IMAGE = "local"

local_runtimes = [
    base_types.Runtime(
        name=constants.TORCH_RUNTIME,
        trainer=types.LocalRuntimeTrainer(
            trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
            framework=TORCH_FRAMEWORK_TYPE,
            num_nodes=1,
            device_count=common_constants.UNKNOWN,
            device=common_constants.UNKNOWN,
            packages=["torch"],
            image=LOCAL_RUNTIME_IMAGE,
        ),
    )
]


# Create venv script


# The exec script to embed training function into container command.
DEPENDENCIES_SCRIPT = textwrap.dedent(
    """
        PIP_DISABLE_PIP_VERSION_CHECK=1 pip install $QUIET \
    --no-warn-script-location $PIP_INDEX $PACKAGE_STR
    """
)

# activate virtualenv, then run the entrypoint from the virtualenv bin
LOCAL_EXEC_ENTRYPOINT = textwrap.dedent(
    """
    $ENTRYPOINT "$FUNC_FILE" "$PARAMETERS"
    """
)

TORCH_COMMAND = "torchrun"

# default command, will run from within the virtualenv
DEFAULT_COMMAND = "python"

# remove virtualenv after training is completed.
LOCAL_EXEC_JOB_CLEANUP_SCRIPT = textwrap.dedent(
    """
    rm -rf $PYENV_LOCATION
    """
)


LOCAL_EXEC_JOB_TEMPLATE = textwrap.dedent(
    """
    set -e
    $OS_PYTHON_BIN -m venv --without-pip $PYENV_LOCATION
    echo "Operating inside $PYENV_LOCATION"
    source $PYENV_LOCATION/bin/activate
    $PYENV_LOCATION/bin/python -m ensurepip --upgrade --default-pip
    $DEPENDENCIES_SCRIPT
    $ENTRYPOINT
    $CLEANUP_SCRIPT
    """
)

LOCAL_EXEC_FILENAME = "train_{}.py"

PYTHON_PACKAGE_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
