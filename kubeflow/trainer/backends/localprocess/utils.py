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
from __future__ import annotations

from collections.abc import Callable
import inspect
import os
from pathlib import Path
import re
import shutil
from string import Template
import textwrap
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

if TYPE_CHECKING:
    # Only imported when a type checker runs — never at runtime.
    # This keeps the runtime import clean (API only) while giving
    # editors and mypy the correct type information.
    from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from kubeflow.trainer.backends.localprocess import constants as local_exec_constants
from kubeflow.trainer.backends.localprocess.types import LocalRuntimeTrainer
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


def _extract_name(requirement: str) -> str:
    if requirement is None:
        raise ValueError("Requirement string cannot be None")
    s = requirement.strip()
    if not s:
        raise ValueError("Empty requirement string")
    m = local_exec_constants.PYTHON_PACKAGE_NAME_RE.match(s)
    if not m:
        raise ValueError(f"Could not parse package name from requirement: {requirement!r}")
    return m.group(1)


def _canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def get_install_packages(
    runtime_packages: list[str],
    trainer_packages: list[str] | None = None,
) -> list[str]:
    # Pure CPU merge — called from get_dependencies_command which is already
    # spanned. No span here avoids noise without losing any signal.
    if not trainer_packages:
        return runtime_packages

    runtime_parsed: list[tuple[str, str]] = []
    last_runtime_index_by_name: dict[str, int] = {}

    for i, orig in enumerate(runtime_packages):
        raw_name = _extract_name(orig)
        canon = _canonicalize_name(raw_name)
        runtime_parsed.append((orig, canon))
        last_runtime_index_by_name[canon] = i

    trainer_parsed: list[tuple[str, str]] = []
    seen_trainer: set[str] = set()
    for orig in trainer_packages:
        raw_name = _extract_name(orig)
        canon = _canonicalize_name(raw_name)
        if canon in seen_trainer:
            raise ValueError(
                f"Duplicate dependency in trainer_packages: '{raw_name}' (canonical: '{canon}')"
            )
        seen_trainer.add(canon)
        trainer_parsed.append((orig, canon))

    trainer_names: set[str] = {canon for _, canon in trainer_parsed}
    merged: list[str] = []
    emitted: set[str] = set()

    for idx, (orig, canon) in enumerate(runtime_parsed):
        if canon in trainer_names:
            continue
        if last_runtime_index_by_name[canon] == idx and canon not in emitted:
            merged.append(orig)
            emitted.add(canon)

    for orig, _ in trainer_parsed:
        merged.append(orig)

    return merged


def get_local_runtime_trainer(
    runtime_name: str,
    venv_dir: str,
    framework: str,
) -> LocalRuntimeTrainer:
    # Pure in-memory construction — no I/O. Spanned at the call site in
    # backend.train so no span needed here.
    local_runtime = next(
        (rt for rt in local_exec_constants.local_runtimes if rt.name == runtime_name),
        None,
    )
    if not local_runtime:
        raise ValueError(f"Runtime {runtime_name} not found")

    trainer = LocalRuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=framework,
        packages=local_runtime.trainer.packages,
        image=local_exec_constants.LOCAL_RUNTIME_IMAGE,
    )

    venv_bin_dir = str(Path(venv_dir) / "bin")
    default_cmd = [str(Path(venv_bin_dir) / local_exec_constants.DEFAULT_COMMAND)]
    if framework == local_exec_constants.TORCH_FRAMEWORK_TYPE:
        trainer.set_command((os.path.join(venv_bin_dir, local_exec_constants.TORCH_COMMAND),))
    else:
        trainer.set_command(tuple(default_cmd))

    return trainer


def get_dependencies_command(
    runtime_packages: list[str],
    pip_index_urls: list[str],
    trainer_packages: list[str],
    quiet: bool = True,
    tracer_provider: TracerProvider | None = None,
) -> str:
    _provider = tracer_provider or trace.get_tracer_provider()
    _tracer = _provider.get_tracer(__name__)

    with _tracer.start_as_current_span("localprocess.deps.resolve") as span:
        packages = get_install_packages(
            runtime_packages=runtime_packages,
            trainer_packages=trainer_packages,
        )
        span.set_attribute("deps.total_packages", len(packages))
        span.set_attribute("deps.trainer_packages", str(trainer_packages))
        span.set_attribute("deps.pip_index_urls", str(pip_index_urls))

        options = [f"--index-url {pip_index_urls[0]}"]
        options.extend(f"--extra-index-url {u}" for u in pip_index_urls[1:])

        mapping = {
            "QUIET": "--quiet" if quiet else "",
            "PIP_INDEX": " ".join(options),
            "PACKAGE_STR": '"{}"'.format('" "'.join(packages)),
        }
        return Template(local_exec_constants.DEPENDENCIES_SCRIPT).substitute(**mapping)


def get_command_using_train_func(
    runtime: types.Runtime,
    train_func: Callable,
    train_func_parameters: dict[str, Any] | None,
    venv_dir: str,
    train_job_name: str,
    tracer_provider: TracerProvider | None = None,
) -> str:
    _provider = tracer_provider or trace.get_tracer_provider()
    _tracer = _provider.get_tracer(__name__)

    with _tracer.start_as_current_span("localprocess.script.write_func") as span:
        if not runtime.trainer:
            raise ValueError(f"Runtime must have a trainer: {runtime}")
        if not callable(train_func):
            raise ValueError(f"Training function must be callable, got: {type(train_func)}")

        func_code = textwrap.dedent(inspect.getsource(train_func))
        func_file = Path(venv_dir) / local_exec_constants.LOCAL_EXEC_FILENAME.format(train_job_name)

        if train_func_parameters is None:
            func_code = f"{func_code}\n{train_func.__name__}()\n"
        else:
            func_code = f"{func_code}\n{train_func.__name__}({train_func_parameters})\n"

        with open(func_file, "w") as f:
            f.write(func_code)

        span.set_attribute("script.func_name", train_func.__name__)
        span.set_attribute("script.file_path", str(func_file))
        span.set_attribute("script.has_parameters", train_func_parameters is not None)

        mapping = {
            "PARAMETERS": "",
            "PYENV_LOCATION": venv_dir,
            "ENTRYPOINT": " ".join(runtime.trainer.command),
            "FUNC_FILE": func_file,
        }
        return Template(local_exec_constants.LOCAL_EXEC_ENTRYPOINT).safe_substitute(**mapping)


def get_cleanup_venv_script(venv_dir: str, cleanup_venv: bool = True) -> str:
    if not cleanup_venv:
        return "\n"
    return Template(local_exec_constants.LOCAL_EXEC_JOB_CLEANUP_SCRIPT).substitute(
        PYENV_LOCATION=venv_dir
    )


def get_local_train_job_script(
    train_job_name: str,
    venv_dir: str,
    trainer: types.CustomTrainer,
    runtime: types.Runtime,
    cleanup_venv: bool = True,
    tracer_provider: TracerProvider | None = None,
) -> tuple:
    _provider = tracer_provider or trace.get_tracer_provider()
    _tracer = _provider.get_tracer(__name__)

    with _tracer.start_as_current_span("localprocess.script.build") as span:
        span.set_attribute("trainjob.name", train_job_name)
        span.set_attribute("trainjob.venv_dir", venv_dir)
        span.set_attribute("trainjob.cleanup_venv", cleanup_venv)

        python_bin = shutil.which("python") or shutil.which("python3")
        if not python_bin:
            span.set_status(StatusCode.ERROR, "No python executable found")
            raise ValueError("No python executable found")

        span.set_attribute("script.python_bin", python_bin)

        if not isinstance(runtime.trainer, LocalRuntimeTrainer):
            raise ValueError(f"Invalid Runtime Trainer type: {type(runtime.trainer)}")

        dependency_script = "\n"
        if trainer.packages_to_install:
            # Opens child span: localprocess.deps.resolve
            dependency_script = get_dependencies_command(
                pip_index_urls=(trainer.pip_index_urls or constants.DEFAULT_PIP_INDEX_URLS),
                runtime_packages=runtime.trainer.packages,
                trainer_packages=trainer.packages_to_install,
                quiet=False,
                tracer_provider=_provider,
            )

        # Opens child span: localprocess.script.write_func
        entrypoint = get_command_using_train_func(
            venv_dir=venv_dir,
            runtime=runtime,
            train_func=trainer.func,
            train_func_parameters=trainer.func_args,
            train_job_name=train_job_name,
            tracer_provider=_provider,
        )

        cleanup_script = get_cleanup_venv_script(cleanup_venv=cleanup_venv, venv_dir=venv_dir)

        command = Template(local_exec_constants.LOCAL_EXEC_JOB_TEMPLATE).safe_substitute(
            OS_PYTHON_BIN=python_bin,
            PYENV_LOCATION=venv_dir,
            DEPENDENCIES_SCRIPT=dependency_script,
            ENTRYPOINT=entrypoint,
            CLEANUP_SCRIPT=cleanup_script,
        )
        return "bash", "-c", command
