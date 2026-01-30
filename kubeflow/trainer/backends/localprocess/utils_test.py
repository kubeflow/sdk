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

from pathlib import Path
import tempfile

from kubeflow.trainer.backends.localprocess import constants as local_exec_constants, utils
from kubeflow.trainer.backends.localprocess.types import LocalRuntimeTrainer
from kubeflow.trainer.types import types


def dummy_func(a: int = 1):
    print(a)


def test_get_local_train_job_script():
    with tempfile.TemporaryDirectory() as venv_dir:
        train_job_name = "test-job"
        trainer = types.CustomTrainer(
            func=dummy_func,
            packages_to_install=["numpy"],
            pip_index_urls=["https://pypi.org/simple"],
        )
        runtime = types.Runtime(
            name="test-runtime",
            trainer=LocalRuntimeTrainer(
                trainer_type=types.TrainerType.CUSTOM_TRAINER,
                framework="torch",
                num_nodes=1,
                packages=["torch"],
                image="local",
            ),
        )
        # Mock command to be just python
        runtime.trainer.set_command(("python",))

        command = utils.get_local_train_job_script(
            train_job_name=train_job_name,
            venv_dir=venv_dir,
            trainer=trainer,
            runtime=runtime,
            cleanup_venv=True,
        )

        # Check command structure
        assert isinstance(command, list)
        assert len(command) == 2
        assert "python" in command[0].lower()  # OS specific, but should contain python
        assert command[1].endswith("runner.py")
        assert Path(command[1]).exists()

        # Check runner.py content
        with open(command[1]) as f:
            content = f.read()
            assert "import subprocess" in content
            assert f'venv_dir = r"{venv_dir}"' in content
            assert "numpy" in content
            assert "torch" in content
            assert "https://pypi.org/simple" in content
            assert "python" in content  # command part

        # Check train function file creation
        func_file = Path(venv_dir) / local_exec_constants.LOCAL_EXEC_FILENAME.format(train_job_name)
        assert func_file.exists()
        with open(func_file) as f:
            func_content = f.read()
            assert "def dummy_func" in func_content
            assert "dummy_func()" in func_content  # parameters match
