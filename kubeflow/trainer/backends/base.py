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

"""Abstract base class defining the runtime backend interface."""

import abc
from collections.abc import Callable, Iterator

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class RuntimeBackend(abc.ABC):
    """Base class for runtime backends.

    Options self-validate by checking the backend instance type in their __call__ method.
    """

    @abc.abstractmethod
    def list_runtimes(self) -> list[types.Runtime]:
        """List the available runtimes."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_runtime(self, name: str) -> types.Runtime:
        """Get the runtime with the given name."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_runtime_packages(self, runtime: types.Runtime) -> list[str] | None:
        """Report the Python version and packages available in the given runtime.

        Backends may either return the list of package requirement strings (as the
        local-process backend does) or report them as a side effect and return
        ``None`` (as the Kubernetes backend does).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train(
        self,
        runtime: str | types.Runtime | None = None,
        initializer: types.Initializer | None = None,
        trainer: types.CustomTrainer
        | types.CustomTrainerContainer
        | types.BuiltinTrainer
        | None = None,
        options: list | None = None,
    ) -> str:
        """Create a TrainJob and return its name."""
        raise NotImplementedError()

    @abc.abstractmethod
    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        """List TrainJobs, optionally filtered by runtime."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job(self, name: str) -> types.TrainJob:
        """Get the TrainJob with the given name."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        """Stream the logs for the given TrainJob step."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job_events(self, name: str) -> list[types.Event]:
        """List the events related to the given TrainJob."""
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: list[Callable[[types.TrainJob], None]] | None = None,
    ) -> types.TrainJob:
        """Wait until the TrainJob reaches one of the expected statuses."""
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_job(self, name: str) -> None:
        """Delete the TrainJob with the given name."""
        raise NotImplementedError()
