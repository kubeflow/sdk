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
import ast
from collections.abc import Callable
import inspect
import math
import os
import textwrap
from typing import Any

from kubernetes import config

from kubeflow.common import constants


def is_running_in_k8s() -> bool:
    return os.path.isdir("/var/run/secrets/kubernetes.io/")


def get_default_target_namespace(context: str | None = None) -> str:
    if not is_running_in_k8s():
        try:
            all_contexts, current_context = config.list_kube_config_contexts()
            # If context is set, we should get namespace from it.
            if context:
                for c in all_contexts:
                    if isinstance(c, dict) and c.get("name") == context:
                        return c["context"]["namespace"]
            # Otherwise, try to get namespace from the current context.
            return current_context["context"]["namespace"]
        except Exception:
            return constants.DEFAULT_NAMESPACE
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        return f.readline()


def validate_wait_for_job_status(polling_interval: int, timeout: int) -> None:
    """Validate polling_interval and timeout values used by wait_for_job_status methods.

    Args:
        polling_interval: The polling interval in seconds.
        timeout: The timeout in seconds.

    Raises:
        ValueError: If polling_interval or timeout are not positive, or if polling_interval
            is not strictly less than timeout.
    """
    if timeout <= 0:
        raise ValueError(f"Timeout must be a positive number, got timeout={timeout}")
    if polling_interval <= 0:
        raise ValueError(
            f"Polling interval must be a positive number, got polling_interval={polling_interval}"
        )
    if polling_interval >= timeout:
        raise ValueError(
            "Polling interval must be strictly less than timeout. "
            f"Received polling_interval={polling_interval}, timeout={timeout}"
        )


def validate_python_function(func: Callable) -> None:
    """Validate a Python function.

    Args:
        func: The Python function to validate.

    Raises:
        ValueError: If the function is not a Python function, is asynchronous,
            is a lambda function, is decorated, or its source cannot be
            inspected or parsed.
    """
    if not inspect.isfunction(func):
        raise ValueError("Function must be a Python function.")

    if inspect.iscoroutinefunction(func):
        raise ValueError("Async functions are not supported.")

    if func.__name__ == "<lambda>":
        raise ValueError("Lambda functions are not supported.")

    try:
        func_source = textwrap.dedent(inspect.getsource(func))
    except TypeError as e:
        raise ValueError(
            "Function must be a pure-Python function; built-in or "
            "C-implemented callables are not supported."
        ) from e
    except OSError as e:
        raise ValueError(
            "Function source could not be read. Functions defined interactively are not supported."
        ) from e

    try:
        func_tree = ast.parse(func_source)
    except SyntaxError as e:
        raise ValueError("Function source could not be parsed.") from e

    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list
        for node in func_tree.body
    ):
        raise ValueError("Decorated functions are not supported.")


def is_literal_func_arg(value: Any) -> bool:
    """Check whether a function argument can be written into a generated script as a literal.

    The SDK renders `func_args` into the generated training script with `repr()`. Only values
    whose `repr()` is a valid Python literal survive that round trip. Other values, such as
    `pathlib.Path`, NumPy scalars, or non finite floats, render as names that are not defined
    in the training container, for example `PosixPath('/data')`, `np.float64(0.1)`, or `inf`.

    Args:
        value: The function argument value.

    Returns:
        True if the value is None, a str, int, bool, finite float, or a list, tuple, or dict
        that only contains such values (dict keys must be str).
    """
    # Use type() rather than isinstance() so that subclasses are rejected: their repr() is
    # usually not a plain literal, e.g. enum.IntEnum members or numpy.float64 under NumPy 2.
    if value is None or type(value) in (str, int, bool):
        return True

    if type(value) is float:
        return math.isfinite(value)

    if type(value) in (list, tuple):
        return all(is_literal_func_arg(v) for v in value)

    if type(value) is dict:
        return all(isinstance(k, str) and is_literal_func_arg(v) for k, v in value.items())

    return False


def validate_func_args(func_args: dict | None) -> None:
    """Validate the arguments that are passed to a user function in a generated script.

    Args:
        func_args: The keyword arguments for the function.

    Raises:
        ValueError: If `func_args` is not a dict, has a key that is not a str, or has a value
            that cannot be written into the generated script as a Python literal.
    """
    if func_args is None:
        return

    if not isinstance(func_args, dict):
        raise ValueError(f"func_args must be a dict, got: {type(func_args).__name__}")

    for key, value in func_args.items():
        if not isinstance(key, str):
            raise ValueError(f"func_args keys must be str, got: {key!r}")
        if not is_literal_func_arg(value):
            raise ValueError(
                f"func_args[{key!r}] has unsupported value {value!r} of type "
                f"{type(value).__name__}. Use None, str, int, bool, a finite float, or a "
                "list, tuple, or dict of these values. For example, pass a path as str(path) "
                "and a NumPy scalar as value.item()."
            )
