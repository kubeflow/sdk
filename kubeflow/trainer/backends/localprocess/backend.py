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

from collections.abc import Callable, Iterator
from datetime import datetime
import logging
import random
import string
import tempfile
import time
from typing import TYPE_CHECKING
import uuid

from opentelemetry import trace

if TYPE_CHECKING:
    # Only imported when a type checker runs — never at runtime.
    # This keeps the runtime import clean (API only) while giving
    # editors and mypy the correct type information.
    from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from kubeflow.trainer.backends.base import RuntimeBackend
from kubeflow.trainer.backends.localprocess import utils as local_utils
from kubeflow.trainer.backends.localprocess.constants import local_runtimes
from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.backends.localprocess.types import (
    LocalBackendJobs,
    LocalBackendStep,
    LocalProcessBackendConfig,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class LocalProcessBackend(RuntimeBackend):
    def __init__(
        self,
        cfg: LocalProcessBackendConfig,
        tracer_provider: TracerProvider | None = None,
    ):
        self.__local_jobs: list[LocalBackendJobs] = []
        self.cfg = cfg

        # Use the provider passed down from TrainerClient.
        # If none, fall back to the global (which may be no-op).
        # This module never installs or changes any provider.
        _provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = _provider.get_tracer(__name__)
        self._tracer_provider = _provider

    def list_runtimes(self) -> list[types.Runtime]:
        with self._tracer.start_as_current_span("localprocess.list_runtimes") as span:
            runtimes = [self.__convert_local_runtime_to_runtime(rt) for rt in local_runtimes]
            span.set_attribute("runtimes.count", len(runtimes))
            return runtimes

    def get_runtime(self, name: str) -> types.Runtime:
        with self._tracer.start_as_current_span("localprocess.get_runtime") as span:
            span.set_attribute("runtime.name", name)
            runtime = next(
                (
                    self.__convert_local_runtime_to_runtime(rt)
                    for rt in local_runtimes
                    if rt.name == name
                ),
                None,
            )
            if not runtime:
                span.set_status(StatusCode.ERROR, f"Runtime '{name}' not found")
                raise ValueError(f"Runtime '{name}' not found.")
            return runtime

    def get_runtime_packages(self, runtime: types.Runtime):
        with self._tracer.start_as_current_span("localprocess.get_runtime_packages") as span:
            span.set_attribute("runtime.name", runtime.name)
            local_runtime = next((rt for rt in local_runtimes if rt.name == runtime.name), None)
            if not local_runtime:
                span.set_status(StatusCode.ERROR, f"Runtime '{runtime.name}' not found")
                raise ValueError(f"Runtime '{runtime.name}' not found.")
            packages = local_runtime.trainer.packages
            span.set_attribute("packages.count", len(packages))
            return packages

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
        with self._tracer.start_as_current_span("localprocess.train") as span:
            try:
                if runtime is None:
                    raise ValueError("Runtime must be provided for LocalProcessBackend")
                if isinstance(runtime, str):
                    runtime = self.get_runtime(runtime)

                span.set_attribute("runtime.name", runtime.name)

                name = None
                if options:
                    job_spec = {}
                    for option in options:
                        option(job_spec, trainer, self)
                    name = job_spec.get("metadata", {}).get("name")

                trainjob_name = name or (
                    random.choice(string.ascii_lowercase)
                    + uuid.uuid4().hex[: constants.JOB_NAME_UUID_LENGTH]
                )
                span.set_attribute("trainjob.name", trainjob_name)

                if not isinstance(trainer, types.CustomTrainer):
                    raise ValueError("CustomTrainer must be set with LocalProcessBackend")

                span.set_attribute("trainer.type", "CustomTrainer")
                if trainer.packages_to_install:
                    span.set_attribute(
                        "trainer.packages_to_install", str(trainer.packages_to_install)
                    )

                venv_dir = tempfile.mkdtemp(prefix=trainjob_name)
                span.set_attribute("trainjob.venv_dir", venv_dir)

                runtime.trainer = local_utils.get_local_runtime_trainer(
                    runtime_name=runtime.name,
                    venv_dir=venv_dir,
                    framework=runtime.trainer.framework,
                )

                # Capture SpanContext before the thread boundary.
                # The thread cannot see the current span because OTel context
                # is thread-local. We pass the SpanContext (just IDs) to
                # LocalJob so it can re-attach the parent on its own thread.
                parent_span_context = trace.get_current_span().get_span_context()

                training_command = local_utils.get_local_train_job_script(
                    trainer=trainer,
                    runtime=runtime,
                    train_job_name=trainjob_name,
                    venv_dir=venv_dir,
                    cleanup_venv=self.cfg.cleanup_venv,
                    tracer_provider=self._tracer_provider,
                )

                runtime.trainer.set_command(training_command)

                train_job = LocalJob(
                    name=f"{trainjob_name}-train",
                    command=training_command,
                    execution_dir=venv_dir,
                    env=trainer.env,
                    dependencies=[],
                    parent_span_context=parent_span_context,
                    tracer_provider=self._tracer_provider,
                )

                self.__register_job(
                    train_job_name=trainjob_name,
                    step_name="train",
                    job=train_job,
                    runtime=runtime,
                )
                train_job.start()
                return trainjob_name

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        with self._tracer.start_as_current_span("localprocess.list_jobs") as span:
            if runtime:
                span.set_attribute("runtime.name", runtime.name)
            result = []
            for _job in self.__local_jobs:
                if runtime and _job.runtime.name != runtime.name:
                    continue
                result.append(
                    types.TrainJob(
                        name=_job.name,
                        creation_timestamp=_job.created,
                        runtime=runtime,
                        num_nodes=1,
                        steps=[
                            types.Step(name=s.step_name, pod_name=s.step_name, status=s.job.status)
                            for s in _job.steps
                        ],
                    )
                )
            span.set_attribute("jobs.count", len(result))
            return result

    def get_job(self, name: str) -> types.TrainJob:
        with self._tracer.start_as_current_span("localprocess.get_job") as span:
            span.set_attribute("trainjob.name", name)
            _job = next((j for j in self.__local_jobs if j.name == name), None)
            if _job is None:
                span.set_status(StatusCode.ERROR, f"No TrainJob with name {name}")
                raise ValueError(f"No TrainJob with name {name}")
            status = self.__get_job_status(_job)
            span.set_attribute("trainjob.status", status)
            return types.TrainJob(
                name=_job.name,
                creation_timestamp=_job.created,
                steps=[
                    types.Step(name=s.step_name, pod_name=s.step_name, status=s.job.status)
                    for s in _job.steps
                ],
                runtime=_job.runtime,
                num_nodes=1,
                status=status,
            )

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        with self._tracer.start_as_current_span("localprocess.get_job_logs") as span:
            span.set_attribute("trainjob.name", name)
            span.set_attribute("logs.follow", follow)
            span.set_attribute("logs.step", step)
            _job = [j for j in self.__local_jobs if j.name == name]
            if not _job:
                span.set_status(StatusCode.ERROR, f"No TrainJob with name {name}")
                raise ValueError(f"No TrainJob with name {name}")
            want_all_steps = step == constants.NODE + "-0"
            for _step in _job[0].steps:
                if not want_all_steps and _step.step_name != step:
                    continue
                yield from _step.job.logs(follow=follow)

    def get_job_events(self, name: str) -> list[types.Event]:
        raise NotImplementedError()

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: list[Callable[[types.TrainJob], None]] | None = None,
    ) -> types.TrainJob:
        with self._tracer.start_as_current_span("localprocess.wait_for_job_status") as span:
            span.set_attribute("trainjob.name", name)
            span.set_attribute("wait.timeout_seconds", timeout)
            span.set_attribute("wait.polling_interval_seconds", polling_interval)
            span.set_attribute("wait.target_statuses", str(status))

            if polling_interval <= 0:
                raise ValueError(
                    f"Polling interval must be a positive number, got polling_interval={polling_interval}"
                )
            if polling_interval >= timeout:
                raise ValueError(
                    f"Polling interval must be strictly less than timeout. "
                    f"Received polling_interval={polling_interval}, timeout={timeout}"
                )

            # find first match or fallback
            _job = next((_job for _job in self.__local_jobs if _job.name == name), None)

            _job = next((_j for _j in self.__local_jobs if _j.name == name), None)
            if _job is None:
                span.set_status(StatusCode.ERROR, f"No TrainJob with name {name}")
                raise ValueError(f"No TrainJob with name {name}")

            for _ in range(round(timeout / polling_interval)):
                # Get current job status
                trainjob = self.get_job(name)

            span.set_status(StatusCode.ERROR, f"Timeout waiting for {name}")
            raise TimeoutError(f"Timeout waiting for TrainJob {name} to reach status: {status}")

    def delete_job(self, name: str):
        with self._tracer.start_as_current_span("localprocess.delete_job") as span:
            span.set_attribute("trainjob.name", name)
            _job = next((j for j in self.__local_jobs if j.name == name), None)
            if _job is None:
                span.set_status(StatusCode.ERROR, f"No TrainJob with name {name}")
                raise ValueError(f"No TrainJob with name {name}")
            _ = [step.job.cancel() for step in _job.steps]
            self.__local_jobs.remove(_job)

    def __get_job_status(self, job: LocalBackendJobs) -> str:
        statuses = [s.job.status for s in job.steps]
        if constants.TRAINJOB_FAILED in statuses:
            return constants.TRAINJOB_FAILED
        if constants.TRAINJOB_RUNNING in statuses:
            return constants.TRAINJOB_RUNNING
        return constants.TRAINJOB_CREATED

    def __register_job(self, train_job_name, step_name, job, runtime):
        existing = [j for j in self.__local_jobs if j.name == train_job_name]
        if not existing:
            _job = LocalBackendJobs(name=train_job_name, runtime=runtime, created=datetime.now())
            self.__local_jobs.append(_job)
        else:
            _job = existing[0]
        if not any(s.step_name == step_name for s in _job.steps):
            _job.steps.append(LocalBackendStep(step_name=step_name, job=job))
        else:
            logger.warning(f"Step '{step_name}' already registered.")

    def __convert_local_runtime_to_runtime(self, local_runtime) -> types.Runtime:
        return types.Runtime(
            name=local_runtime.name,
            trainer=types.RuntimeTrainer(
                trainer_type=local_runtime.trainer.trainer_type,
                framework=local_runtime.trainer.framework,
                num_nodes=local_runtime.trainer.num_nodes,
                device_count=local_runtime.trainer.device_count,
                device=local_runtime.trainer.device,
                image=local_runtime.trainer.image,
            ),
        )
