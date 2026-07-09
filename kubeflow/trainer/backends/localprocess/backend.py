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

"""Backend that runs Kubeflow TrainJobs as local subprocesses."""

from collections.abc import Callable, Iterator
from datetime import datetime
import logging
import random
import string
import tempfile
import time
import uuid

from kubeflow.trainer.backends.base import RuntimeBackend
from kubeflow.trainer.backends.localprocess import utils as local_utils
from kubeflow.trainer.backends.localprocess.constants import local_runtimes
from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.backends.localprocess.types import (
    LocalBackendJobs,
    LocalBackendStep,
    LocalProcessBackendConfig,
    LocalRuntimeTrainer,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class LocalProcessBackend(RuntimeBackend):
    """Execute TrainJobs as local subprocesses inside per-job virtual environments."""

    def __init__(
        self,
        cfg: LocalProcessBackendConfig,
    ) -> None:
        """Initialize the backend with the given configuration.

        Args:
            cfg: Configuration controlling the local execution behavior.
        """
        # list of running subprocesses
        self.__local_jobs: list[LocalBackendJobs] = []
        self.cfg = cfg

    def list_runtimes(self) -> list[types.Runtime]:
        """Return all runtimes supported by the local backend."""
        return [self.__convert_local_runtime_to_runtime(local_runtime=rt) for rt in local_runtimes]

    def get_runtime(self, name: str) -> types.Runtime:
        """Return the runtime with the given name.

        Args:
            name: Name of the runtime to look up.

        Returns:
            The matching runtime.

        Raises:
            ValueError: If no runtime with the given name exists.
        """
        runtime = next(
            (
                self.__convert_local_runtime_to_runtime(rt)
                for rt in local_runtimes
                if rt.name == name
            ),
            None,
        )
        if not runtime:
            raise ValueError(f"Runtime '{name}' not found.")

        return runtime

    def get_runtime_packages(self, runtime: types.Runtime) -> list[str]:
        """Return the packages installed by the given runtime.

        Args:
            runtime: Runtime whose packages should be returned.

        Returns:
            The list of package requirement strings.

        Raises:
            ValueError: If no runtime with the given name exists.
        """
        local_runtime = next((rt for rt in local_runtimes if rt.name == runtime.name), None)
        if not local_runtime:
            raise ValueError(f"Runtime '{runtime.name}' not found.")

        if not isinstance(local_runtime.trainer, LocalRuntimeTrainer):
            raise ValueError(f"Runtime '{runtime.name}' does not expose local packages.")

        return local_runtime.trainer.packages

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
        """Start a training job as a local subprocess.

        Args:
            runtime: Runtime (or its name) to train with.
            initializer: Unused by the local backend; accepted for interface parity.
            trainer: The trainer describing the workload. Only CustomTrainer is supported.
            options: Optional list of job options. Kubernetes-only options are rejected.

        Returns:
            The generated (or user-provided) TrainJob name.

        Raises:
            ValueError: If the runtime is missing or the trainer is not a CustomTrainer.
        """
        if runtime is None:
            raise ValueError("Runtime must be provided for LocalProcessBackend")
        if isinstance(runtime, str):
            runtime = self.get_runtime(runtime)

        # Process options to extract configuration
        name = None
        if options:
            job_spec = {}
            for option in options:
                option(job_spec, trainer, self)

            metadata_section = job_spec.get("metadata", {})
            name = metadata_section.get("name")

        # Generate train job name if not provided via options
        trainjob_name = name or (
            random.choice(string.ascii_lowercase)
            + uuid.uuid4().hex[: constants.JOB_NAME_UUID_LENGTH]
        )

        # localprocess backend only supports CustomTrainer
        if not isinstance(trainer, types.CustomTrainer):
            raise ValueError("CustomTrainer must be set with LocalProcessBackend")

        # create temp dir
        venv_dir = tempfile.mkdtemp(prefix=trainjob_name)
        logger.debug(f"operating in {venv_dir}")

        # get local runtime trainer
        runtime.trainer = local_utils.get_local_runtime_trainer(
            runtime_name=runtime.name,
            venv_dir=venv_dir,
            framework=runtime.trainer.framework,
        )

        # build training job command
        training_command = local_utils.get_local_train_job_script(
            trainer=trainer,
            runtime=runtime,
            train_job_name=trainjob_name,
            venv_dir=venv_dir,
            cleanup_venv=self.cfg.cleanup_venv,
        )

        # set the command in the runtime trainer
        runtime.trainer.set_command(training_command)

        # create subprocess object
        train_job = LocalJob(
            name=f"{trainjob_name}-train",
            command=training_command,
            execution_dir=venv_dir,
            env=trainer.env,
            dependencies=[],
        )

        self.__register_job(
            train_job_name=trainjob_name,
            step_name="train",
            job=train_job,
            runtime=runtime,
        )
        # start the job.
        train_job.start()

        return trainjob_name

    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        """Return all tracked TrainJobs, optionally filtered by runtime.

        Args:
            runtime: If provided, only jobs for this runtime are returned.

        Returns:
            The list of matching TrainJobs.
        """
        result = []

        for _job in self.__local_jobs:
            job_runtime = _job.runtime
            created = _job.created
            if job_runtime is None or created is None:
                continue
            if runtime and job_runtime.name != runtime.name:
                continue
            result.append(
                types.TrainJob(
                    name=_job.name,
                    creation_timestamp=created,
                    runtime=job_runtime,
                    num_nodes=1,
                    steps=[
                        types.Step(name=s.step_name, pod_name=s.step_name, status=s.job.status)
                        for s in _job.steps
                    ],
                )
            )
        return result

    def get_job(self, name: str) -> types.TrainJob:
        """Return the TrainJob with the given name.

        Args:
            name: Name of the TrainJob to fetch.

        Returns:
            The matching TrainJob with its aggregated status.

        Raises:
            ValueError: If no TrainJob with the given name exists.
        """
        _job = next((j for j in self.__local_jobs if j.name == name), None)
        if _job is None:
            raise ValueError(f"No TrainJob with name {name}")

        runtime = _job.runtime
        created = _job.created
        if runtime is None or created is None:
            raise ValueError(f"TrainJob {name} is missing runtime or creation metadata")

        # check and set the correct job status to match `TrainerClient` supported statuses
        status = self.__get_job_status(_job)

        return types.TrainJob(
            name=_job.name,
            creation_timestamp=created,
            steps=[
                types.Step(
                    name=_step.step_name,
                    pod_name=_step.step_name,
                    status=_step.job.status,
                )
                for _step in _job.steps
            ],
            runtime=runtime,
            num_nodes=1,
            status=status,
        )

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        """Yield logs for a TrainJob, for a single step or for all steps.

        Args:
            name: Name of the TrainJob.
            follow: If True, stream logs live as they are produced.
            step: Step to read logs from. The default reads all steps.

        Yields:
            Chunks of log output.

        Raises:
            ValueError: If no TrainJob with the given name exists.
        """
        _job = [j for j in self.__local_jobs if j.name == name]
        if not _job:
            raise ValueError(f"No TrainJob with name {name}")

        want_all_steps = step == constants.NODE + "-0"

        for _step in _job[0].steps:
            if not want_all_steps and _step.step_name != step:
                continue
            # Flatten the generator and pass through flags so it behaves as expected
            # (adjust args if stream_logs has different signature)
            yield from _step.job.logs(follow=follow)

    def get_job_events(self, name: str) -> list[types.Event]:
        """Return events for a TrainJob (not supported by the local backend).

        Args:
            name: Name of the TrainJob.

        Raises:
            NotImplementedError: Always, since local jobs do not emit events.
        """
        raise NotImplementedError()

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: list[Callable[[types.TrainJob], None]] | None = None,
    ) -> types.TrainJob:
        """Poll a TrainJob until it reaches one of the desired statuses.

        Args:
            name: Name of the TrainJob to wait for.
            status: Set of statuses that are considered terminal for this wait.
            timeout: Maximum number of seconds to wait.
            polling_interval: Seconds between status checks; must be < timeout.
            callbacks: Callables invoked with the TrainJob on every poll.

        Returns:
            The TrainJob once it reaches one of the desired statuses.

        Raises:
            ValueError: If polling_interval is not positive or is >= timeout, or the job
                does not exist.
            TimeoutError: If the timeout elapses before a desired status is reached.
        """
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

        if _job is None:
            raise ValueError(f"No TrainJob with name {name}")

        for _ in range(round(timeout / polling_interval)):
            # Get current job status
            trainjob = self.get_job(name)

            # Invoke callbacks if provided
            if callbacks:
                for callback in callbacks:
                    callback(trainjob)

            # Return if job has reached desired status
            if trainjob.status in status:
                return trainjob

            time.sleep(polling_interval)

        # Timeout reached
        raise TimeoutError(f"Timeout waiting for TrainJob {name} to reach status: {status}")

    def delete_job(self, name: str) -> None:
        """Cancel and remove the TrainJob with the given name.

        Args:
            name: Name of the TrainJob to delete.

        Raises:
            ValueError: If no TrainJob with the given name exists.
        """
        # find job first.
        _job = next((j for j in self.__local_jobs if j.name == name), None)
        if _job is None:
            raise ValueError(f"No TrainJob with name {name}")

        # cancel all nested step jobs in target job
        _ = [step.job.cancel() for step in _job.steps]
        # remove the job from the list of jobs
        self.__local_jobs.remove(_job)

    def __get_job_status(self, job: LocalBackendJobs) -> str:
        if not job.steps:
            return constants.TRAINJOB_CREATED
        statuses = [_step.job.status for _step in job.steps]
        # if status is running or failed will take precedence over completed
        if constants.TRAINJOB_FAILED in statuses:
            status = constants.TRAINJOB_FAILED
        elif constants.TRAINJOB_RUNNING in statuses:
            status = constants.TRAINJOB_RUNNING
        elif constants.TRAINJOB_CREATED in statuses:
            status = constants.TRAINJOB_CREATED
        else:
            status = constants.TRAINJOB_COMPLETE

        return status

    def __register_job(
        self,
        train_job_name: str,
        step_name: str,
        job: LocalJob,
        runtime: types.Runtime,
    ) -> None:
        existing_jobs = [j for j in self.__local_jobs if j.name == train_job_name]
        if not existing_jobs:
            _job = LocalBackendJobs(name=train_job_name, runtime=runtime, created=datetime.now())
            self.__local_jobs.append(_job)
        else:
            _job = existing_jobs[0]

        existing_steps = [s for s in _job.steps if s.step_name == step_name]
        if not existing_steps:
            _step = LocalBackendStep(step_name=step_name, job=job)
            _job.steps.append(_step)
        else:
            logger.warning(f"Step '{step_name}' already registered.")

    def __convert_local_runtime_to_runtime(self, local_runtime: types.Runtime) -> types.Runtime:
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
