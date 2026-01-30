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
        name=constants.DEFAULT_TRAINING_RUNTIME,
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


TORCH_COMMAND = "torchrun"
DEFAULT_COMMAND = "python"

# Create venv script


RUNNER_TEMPLATE = textwrap.dedent(
    """
import os
import shutil
import subprocess
import sys

def main():
    venv_dir = r"${pyenv_location}"
    requirements = ${packages_list}
    pip_index_urls = ${pip_index_urls}
    command = ${command}
    cleanup_venv = ${cleanup_venv}

    # 1. Create venv
    print(f"Creating venv at {venv_dir}")
    # Use sys.executable to ensure we use the same python interpreter kind
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

    venv_python = os.path.join(venv_dir, "bin", "python")
    # Upgrade pip
    subprocess.run([venv_python, "-m", "ensurepip", "--upgrade", "--default-pip"], check=True)

    # 2. Install dependencies
    if requirements:
        pip_cmd = [venv_python, "-m", "pip", "install"]
        if pip_index_urls:
            pip_cmd.extend(["--index-url", pip_index_urls[0]])
            for url in pip_index_urls[1:]:
                pip_cmd.extend(["--extra-index-url", url])
        pip_cmd.extend(requirements)
        print(f"Installing dependencies: {requirements}")
        subprocess.run(pip_cmd, check=True)

    # 3. Run Training Command
    print(f"Running command: {command}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    finally:
        # 4. Cleanup
        if cleanup_venv:
            print(f"Cleaning up venv at {venv_dir}")
            try:
                # We are running inside venv_dir, so we can't delete it fully on some OSs.
                # But typically on Unix it's allowed.
                # If we encounter errors, we ignore them to avoid failing the job status.
                shutil.rmtree(venv_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to cleanup venv: {e}")

if __name__ == "__main__":
    main()
"""
)

LOCAL_EXEC_FILENAME = "train_{}.py"

PYTHON_PACKAGE_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
