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

import abc
from collections.abc import Iterator
from typing import Optional, Union

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class RuntimeBackend(abc.ABC):
    """Base class for runtime backends.

    Options self-validate by checking the backend instance type in their __call__ method.
    """

    @abc.abstractmethod
    def list_runtimes(self) -> list[types.Runtime]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_runtime(self, name: str) -> types.Runtime:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_runtime_packages(self, runtime: types.Runtime):
        raise NotImplementedError()

    @abc.abstractmethod
    def train(
        self,
        runtime: Optional[types.Runtime] = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[
            Union[types.CustomTrainer, types.CustomTrainerContainer, types.BuiltinTrainer]
        ] = None,
        options: Optional[list] = None,
    ) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> list[types.TrainJob]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job(self, name: str) -> types.TrainJob:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJob:
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_job(self, name: str):
        raise NotImplementedError()
