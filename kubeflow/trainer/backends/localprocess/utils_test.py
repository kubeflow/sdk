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

"""Unit tests for LocalProcessBackend utils — focusing on training-function call generation."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock

import pytest

from kubeflow.test.common import SUCCESS, TestCase
from kubeflow.trainer.backends.localprocess import utils as local_utils
from kubeflow.trainer.backends.localprocess.constants import LOCAL_EXEC_FILENAME
from kubeflow.trainer.types import types


# ---------------------------------------------------------------------------
# Helper: build a minimal Runtime with a real trainer command so that
# get_command_using_train_func does not fail before reaching the call-gen code.
# ---------------------------------------------------------------------------
def _make_runtime(command: list[str] | None = None) -> types.Runtime:
    trainer = MagicMock(spec=types.RuntimeTrainer)
    trainer.command = command or ["python"]
    runtime = MagicMock(spec=types.Runtime)
    runtime.trainer = trainer
    return runtime


def _read_generated_call(venv_dir: str, job_name: str) -> str:
    """Return the last non-empty line of the generated script file (the call line)."""
    script_path = Path(venv_dir) / LOCAL_EXEC_FILENAME.format(job_name)
    text = script_path.read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1]


# ---------------------------------------------------------------------------
# Regression tests: get_command_using_train_func call generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        # ---------------------------------------------------------------
        # No func_args: call must be fn() with no arguments.
        # ---------------------------------------------------------------
        TestCase(
            name="no func_args generates empty call",
            expected_status=SUCCESS,
            config={"func_args": None},
            expected_output="dummy_no_args()",
        ),
        # ---------------------------------------------------------------
        # Named required parameters: train(lr, epochs)
        # The bug generated train({'lr':…}) → TypeError at runtime.
        # The fix generates train(**{'lr':…}) which passes each kwarg.
        # ---------------------------------------------------------------
        TestCase(
            name="func_args with named required parameters uses kwargs unpacking",
            expected_status=SUCCESS,
            config={"func_args": {"lr": 0.01, "epochs": 5}},
            expected_output="dummy_named(**{'lr': 0.01, 'epochs': 5})",
        ),
        # ---------------------------------------------------------------
        # Keyword-only parameters: train(*, lr, epochs)
        # The bug generated train({'lr':…}) → TypeError (0 positional args).
        # ---------------------------------------------------------------
        TestCase(
            name="func_args with keyword-only parameters uses kwargs unpacking",
            expected_status=SUCCESS,
            config={"func_args": {"lr": 0.01, "epochs": 5}},
            expected_output="dummy_kwonly(**{'lr': 0.01, 'epochs': 5})",
        ),
        # ---------------------------------------------------------------
        # **kwargs collector: train(**kwargs)
        # The bug generated train({'lr':…}) → TypeError (0 positional args).
        # ---------------------------------------------------------------
        TestCase(
            name="func_args with var-kwargs collector uses kwargs unpacking",
            expected_status=SUCCESS,
            config={"func_args": {"lr": 0.01, "epochs": 5}},
            expected_output="dummy_var_kwargs(**{'lr': 0.01, 'epochs': 5})",
        ),
    ],
)
def test_get_command_using_train_func_call_generation(test_case: TestCase):
    """Generated training-function call must unpack func_args as **kwargs."""
    print("Executing test:", test_case.name)

    # Four target functions — one per test case (matched by expected_output prefix).
    def dummy_no_args():
        pass

    def dummy_named(lr, epochs):
        pass

    def dummy_kwonly(*, lr, epochs):
        pass

    def dummy_var_kwargs(**kwargs):
        pass

    # Pick the function whose name matches the expected call prefix.
    fn_map = {
        "dummy_no_args": dummy_no_args,
        "dummy_named": dummy_named,
        "dummy_kwonly": dummy_kwonly,
        "dummy_var_kwargs": dummy_var_kwargs,
    }
    fn_name = test_case.expected_output.split("(")[0]
    train_func = fn_map[fn_name]

    runtime = _make_runtime()
    func_args = test_case.config["func_args"]

    with tempfile.TemporaryDirectory() as venv_dir:
        local_utils.get_command_using_train_func(
            runtime=runtime,
            train_func=train_func,
            train_func_parameters=func_args,
            venv_dir=venv_dir,
            train_job_name="test-job",
        )
        call_line = _read_generated_call(venv_dir, "test-job")

    assert call_line == test_case.expected_output, (
        f"Generated call {call_line!r} does not match expected {test_case.expected_output!r}"
    )
    print("test execution complete")


# ---------------------------------------------------------------------------
# Smoke test: the generated call is valid Python that actually executes.
# This confirms the fix produces correct kwargs unpacking, not just the right
# string representation.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="generated script executes correctly for named params",
            expected_status=SUCCESS,
            config={
                "func_args": {"lr": 0.01, "epochs": 5},
                "fn_name": "dummy_named",
            },
        ),
        TestCase(
            name="generated script executes correctly for keyword-only params",
            expected_status=SUCCESS,
            config={
                "func_args": {"lr": 0.01, "epochs": 5},
                "fn_name": "dummy_kwonly",
            },
        ),
        TestCase(
            name="generated script executes correctly for var-kwargs",
            expected_status=SUCCESS,
            config={
                "func_args": {"lr": 0.01, "epochs": 5},
                "fn_name": "dummy_var_kwargs",
            },
        ),
    ],
)
def test_generated_call_executes(test_case: TestCase):
    """The generated script file must be syntactically valid and run without TypeError."""
    print("Executing test:", test_case.name)

    def dummy_named(lr, epochs):
        assert lr == 0.01
        assert epochs == 5

    def dummy_kwonly(*, lr, epochs):
        assert lr == 0.01
        assert epochs == 5

    def dummy_var_kwargs(**kwargs):
        assert kwargs == {"lr": 0.01, "epochs": 5}

    fn_map = {
        "dummy_named": dummy_named,
        "dummy_kwonly": dummy_kwonly,
        "dummy_var_kwargs": dummy_var_kwargs,
    }
    train_func = fn_map[test_case.config["fn_name"]]
    runtime = _make_runtime()

    with tempfile.TemporaryDirectory() as venv_dir:
        local_utils.get_command_using_train_func(
            runtime=runtime,
            train_func=train_func,
            train_func_parameters=test_case.config["func_args"],
            venv_dir=venv_dir,
            train_job_name="exec-test-job",
        )
        script_path = Path(venv_dir) / LOCAL_EXEC_FILENAME.format("exec-test-job")
        script_text = script_path.read_text()

    # Execute the generated script in an isolated namespace.
    # Any TypeError from incorrect call generation will surface here.
    exec(script_text, {})
    print("test execution complete")
