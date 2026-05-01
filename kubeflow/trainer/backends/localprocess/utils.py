from collections.abc import Callable
import inspect
import os
from pathlib import Path
import re
import shutil
from string import Template
import textwrap
from typing import Any

from kubeflow.trainer.backends.localprocess import constants as local_exec_constants
from kubeflow.trainer.backends.localprocess.types import LocalRuntimeTrainer
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


def _extract_name(requirement: str) -> str:
    """
    Extract the base distribution name from a requirement string without external deps.

    Supports common PEP 508 patterns:
      - 'package'
      - 'package[extra1,extra2]'
      - 'package==1.2.3', 'package>=1.0', 'package~=1.4', etc.
      - 'package @ https://...'
      - markers after ';' are irrelevant for name extraction.

    Returns the *raw* (un-normalized) name as it appears.
    Raises ValueError if a name cannot be parsed.
    """
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
    """
    PEP 503-style normalization: case-insensitive, and collapse runs of -, _, . into '-'.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def get_install_packages(
    runtime_packages: list[str],
    trainer_packages: list[str] | None = None,
) -> list[str]:
    """
    Merge two requirement lists into a single list of strings.

    Rules implemented:
    1) If a package appears in trainer_packages, it overwrites the one in runtime_packages.
       We keep the *trainer string verbatim* (specifier, markers, extras, spacing).
    2) Case-insensitive matching of package names (PEP 503-style normalization).
    3) Output is a list of strings.
    4) If trainer_packages contains the same dependency multiple times (case-insensitive),
       raise ValueError.
    5) If runtime_packages contains duplicates, the last one among *runtime* wins there
       (no error), but any trainer entry still overwrites it. Runtime packages shouldn't
        have any duplicates.
    6) Ordering: keep runtime-only packages in their original order (emitting only their
       last occurrence), then append all trainer packages in their original order.
    """
    if not trainer_packages:
        return runtime_packages

    # --- Parse + normalize runtime ---
    runtime_parsed: list[tuple[str, str]] = []  # (orig, canonical_name)
    last_runtime_index_by_name: dict[str, int] = {}

    for i, orig in enumerate(runtime_packages):
        raw_name = _extract_name(orig)
        canon = _canonicalize_name(raw_name)
        runtime_parsed.append((orig, canon))
        last_runtime_index_by_name[canon] = i  # last occurrence index wins among runtime

    # --- Parse + validate trainer (detect duplicates) ---
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

    # --- Build merged list respecting order semantics ---
    merged: list[str] = []

    # 1) Runtime-only packages (only emit the last occurrence for each name)
    emitted_runtime_names: set[str] = set()
    for idx, (orig, canon) in enumerate(runtime_parsed):
        if canon in trainer_names:
            continue  # overwritten by trainer
        if last_runtime_index_by_name[canon] == idx and canon not in emitted_runtime_names:
            merged.append(orig)
            emitted_runtime_names.add(canon)

    # 2) Trainer packages (overwrite and preserve trainer's exact strings, original order)
    for orig, _ in trainer_parsed:
        merged.append(orig)

    return merged


def get_local_runtime_trainer(
    runtime_name: str,
    venv_dir: str,
    framework: str,
) -> LocalRuntimeTrainer:
    """
    Get the LocalRuntimeTrainer object.
    """
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

    # set command to run from venv
    venv_bin_dir = str(Path(venv_dir) / "bin")
    default_cmd = [str(Path(venv_bin_dir) / local_exec_constants.DEFAULT_COMMAND)]
    # Set the Trainer entrypoint.
    if framework == local_exec_constants.TORCH_FRAMEWORK_TYPE:
        _c = [os.path.join(venv_bin_dir, local_exec_constants.TORCH_COMMAND)]
        trainer.set_command(tuple(_c))
    else:
        trainer.set_command(tuple(default_cmd))

    return trainer


def get_dependencies_command(
    runtime_packages: list[str],
    pip_index_urls: list[str],
    trainer_packages: list[str],
    quiet: bool = True,
) -> str:
    # resolve runtime dependencies and trainer dependencies.
    packages = get_install_packages(
        runtime_packages=runtime_packages,
        trainer_packages=trainer_packages,
    )

    options = [f"--index-url {pip_index_urls[0]}"]
    options.extend(f"--extra-index-url {extra_index_url}" for extra_index_url in pip_index_urls[1:])

    """
           PIP_DISABLE_PIP_VERSION_CHECK=1 pip install $QUIET $AS_USER \
       --no-warn-script-location $PIP_INDEX $PACKAGE_STR
       """
    mapping = {
        "QUIET": "--quiet" if quiet else "",
        "PIP_INDEX": " ".join(options),
        "PACKAGE_STR": '"{}"'.format('" "'.join(packages)),  # quote deps
    }
    t = Template(local_exec_constants.DEPENDENCIES_SCRIPT)
    result = t.substitute(**mapping)
    return result


def get_command_using_train_func(
    runtime: types.Runtime,
    train_func: Callable,
    train_func_parameters: dict[str, Any] | None,
    venv_dir: str,
    train_job_name: str,
) -> str:
    """
    Get the Trainer container command from the given training function and parameters.
    """
    # Check if the runtime has a Trainer.
    if not runtime.trainer:
        raise ValueError(f"Runtime must have a trainer: {runtime}")

    # Check if training function is callable.
    if not callable(train_func):
        raise ValueError(
            f"Training function must be callable, got function type: {type(train_func)}"
        )

    # Extract the function implementation.
    func_code = inspect.getsource(train_func)

    # Extract the file name where the function is defined and move it the venv directory.
    func_file = Path(venv_dir) / local_exec_constants.LOCAL_EXEC_FILENAME.format(train_job_name)

    # Function might be defined in some indented scope (e.g. in another function).
    # We need to dedent the function code.
    func_code = textwrap.dedent(func_code)

    # Wrap function code to execute it from the file. For example:
    # TODO (andreyvelich): Find a better way to run users' scripts.
    # def train(parameters):
    #     print('Start Training...')
    # train({'lr': 0.01})
    if train_func_parameters is None:
        func_code = f"{func_code}\n{train_func.__name__}()\n"
    else:
        func_code = f"{func_code}\n{train_func.__name__}({train_func_parameters})\n"

    with open(func_file, "w") as f:
        f.write(func_code)
    f.close()

    t = Template(local_exec_constants.LOCAL_EXEC_ENTRYPOINT)
    mapping = {
        "PARAMETERS": "",  ## Torch Parameters if any
        "PYENV_LOCATION": venv_dir,
        "ENTRYPOINT": " ".join(runtime.trainer.command),
        "FUNC_FILE": func_file,
    }
    entrypoint = t.safe_substitute(**mapping)

    return entrypoint


def get_cleanup_venv_script(venv_dir: str, cleanup_venv: bool = True) -> str:
    script = "\n"
    if not cleanup_venv:
        return script

    t = Template(local_exec_constants.LOCAL_EXEC_JOB_CLEANUP_SCRIPT)
    mapping = {
        "PYENV_LOCATION": venv_dir,
    }
    return t.substitute(**mapping)


def get_local_train_job_script(
    train_job_name: str,
    venv_dir: str,
    trainer: types.CustomTrainer,
    runtime: types.Runtime,
    cleanup_venv: bool = True,
) -> tuple:
    # use local-exec train job template
    t = Template(local_exec_constants.LOCAL_EXEC_JOB_TEMPLATE)
    # find os python binary to create venv
    python_bin = shutil.which("python")
    if not python_bin:
        python_bin = shutil.which("python3")
    if not python_bin:
        raise ValueError("No python executable found")

    # workout if dependencies needs to be installed
    if isinstance(runtime.trainer, LocalRuntimeTrainer):
        runtime_trainer: LocalRuntimeTrainer = runtime.trainer
    else:
        raise ValueError("Invalid Runtime Trainer type: {type(runtime.trainer)}")
    dependency_script = "\n"
    if trainer.packages_to_install:
        dependency_script = get_dependencies_command(
            pip_index_urls=(
                trainer.pip_index_urls
                if trainer.pip_index_urls
                else constants.DEFAULT_PIP_INDEX_URLS
            ),
            runtime_packages=runtime_trainer.packages,
            trainer_packages=trainer.packages_to_install,
            quiet=False,
        )

    entrypoint = get_command_using_train_func(
        venv_dir=venv_dir,
        runtime=runtime,
        train_func=trainer.func,
        train_func_parameters=trainer.func_args,
        train_job_name=train_job_name,
    )

    cleanup_script = get_cleanup_venv_script(cleanup_venv=cleanup_venv, venv_dir=venv_dir)

    mapping = {
        "OS_PYTHON_BIN": python_bin,
        "PYENV_LOCATION": venv_dir,
        "DEPENDENCIES_SCRIPT": dependency_script,
        "ENTRYPOINT": entrypoint,
        "CLEANUP_SCRIPT": cleanup_script,
    }

    command = t.safe_substitute(**mapping)

    return "bash", "-c", command
