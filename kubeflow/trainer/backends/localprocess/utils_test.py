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

"""Unit tests for LocalProcessBackend script-building utilities."""

from kubeflow.trainer.backends.localprocess.utils import (
    get_local_runtime_trainer,
    get_local_train_job_script,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

TORCH_RUNTIME = constants.DEFAULT_TRAINING_RUNTIME


def dummy_training_function():
    """Dummy training function used to build a real training script."""
    print("Training started")


def _build_script(tmp_path, cleanup_venv: bool) -> str:
    venv_dir = str(tmp_path)
    runtime_trainer = get_local_runtime_trainer(
        runtime_name=TORCH_RUNTIME,
        venv_dir=venv_dir,
        framework="torch",
    )
    runtime = types.Runtime(
        name=TORCH_RUNTIME,
        trainer=runtime_trainer,
        kind=types.RuntimeKind.TRAINING_RUNTIME,
    )
    trainer = types.CustomTrainer(func=dummy_training_function)

    _, _, script = get_local_train_job_script(
        train_job_name="test-job",
        venv_dir=venv_dir,
        trainer=trainer,
        runtime=runtime,
        cleanup_venv=cleanup_venv,
    )
    return script


def test_cleanup_runs_on_exit_trap_not_as_a_trailing_statement(tmp_path):
    """Regression test: cleanup must run even when the entrypoint fails.

    Before the fix, the cleanup command (`rm -rf $PYENV_LOCATION`) was the last line of a
    `set -e` script, after the entrypoint. If the entrypoint (the user's training function)
    or any earlier setup step exited non-zero, `set -e` aborted the script immediately and
    the cleanup line was never reached, permanently leaking the per-job virtualenv. Cleanup
    must instead run unconditionally via an EXIT trap, registered before any step that can
    fail.
    """
    script = _build_script(tmp_path, cleanup_venv=True)

    assert "trap '" in script
    assert "rm -rf" in script
    assert str(tmp_path) in script

    # The trap must be registered before the venv is created, so cleanup also runs if venv
    # creation, dependency installation, or the entrypoint itself fails.
    trap_index = script.index("trap '")
    venv_create_index = script.index("-m venv")
    assert trap_index < venv_create_index

    # Cleanup must not be a bare trailing statement after the entrypoint, since `set -e`
    # would skip it whenever the entrypoint or an earlier step fails. The entrypoint line
    # runs the generated training script file, named after the TrainJob.
    entrypoint_index = script.rindex("train_test-job.py")
    trailing_script = script[entrypoint_index:]
    assert "rm -rf" not in trailing_script


def test_cleanup_disabled_still_registers_a_harmless_trap(tmp_path):
    """When cleanup_venv=False, the trap must still be valid (a no-op), not break the script."""
    script = _build_script(tmp_path, cleanup_venv=False)

    assert "trap '" in script
    assert "rm -rf" not in script
