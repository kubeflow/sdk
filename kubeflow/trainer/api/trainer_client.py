# Copyright 2024 The Kubeflow Authors.
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

from collections.abc import Callable, Iterator
import logging

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, TracerProvider

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.backends.container.backend import ContainerBackend
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.backends.localprocess.backend import (
    LocalProcessBackend,
    LocalProcessBackendConfig,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)
TRACER_NAME = "kf.trainer"


class TrainerClient:
    def __init__(
        self,
        backend_config: KubernetesBackendConfig
        | LocalProcessBackendConfig
        | ContainerBackendConfig
        | None = None,
        tracer_provider: TracerProvider | None = None,
    ):
        """Initialize a Kubeflow Trainer client.

        Args:
            backend_config: Backend configuration. Defaults to KubernetesBackendConfig.
            tracer_provider: Optional OpenTelemetry TracerProvider. Controls where telemetry
                data is sent (Jaeger, Tempo, Honeycomb, etc.).
                - Pass a configured provider to route spans to a specific backend.
                - Pass None to use the globally installed provider (if any).
                - If no provider is installed at all, all tracing is a silent no-op.
                The SDK never installs or modifies a global provider.
        """
        # Resolve the provider:
        # 1. Use the explicitly passed provider, or
        # 2. Fall back to the global provider (which may be the no-op provider).
        # get_tracer_provider() always returns something — either what the user
        # installed globally, or OTel's built-in no-op. Never None.
        _provider = tracer_provider or trace.get_tracer_provider()

        # TrainerClient is the root of all kubeflow tracing. Every span created
        # anywhere in the SDK is a descendant of a span started here.
        self._tracer = _provider.get_tracer(TRACER_NAME)

        if not backend_config:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config, tracer_provider=_provider)
        elif isinstance(backend_config, LocalProcessBackendConfig):
            self.backend = LocalProcessBackend(backend_config, tracer_provider=_provider)
        elif isinstance(backend_config, ContainerBackendConfig):
            self.backend = ContainerBackend(backend_config, tracer_provider=_provider)
        else:
            raise ValueError(f"Invalid backend config '{backend_config}'")

    def list_runtimes(self) -> list[types.Runtime]:
        with self._tracer.start_as_current_span("list runtimes", kind=SpanKind.INTERNAL) as span:
            result = self.backend.list_runtimes()
            span.set_status(Status(StatusCode.OK))
            return result

    def get_runtime(self, name: str) -> types.Runtime:
        with self._tracer.start_as_current_span("get runtime", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("runtime.name", name)
            result = self.backend.get_runtime(name=name)
            span.set_status(Status(StatusCode.OK))
            return result

    def get_runtime_packages(self, runtime: types.Runtime):
        with self._tracer.start_as_current_span(
            "get runtime packages", kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute("runtime.name", runtime.name)
            result = self.backend.get_runtime_packages(runtime=runtime)
            span.set_status(Status(StatusCode.OK))
            return result

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
        """Create a TrainJob. You can configure the TrainJob using one of these trainers:

        - CustomTrainer: Runs training with a user-defined function that fully encapsulates the
            training process.
        - CustomTrainerContainer: Runs training with a user-defined image that fully encapsulates
            the training process.
        - BuiltinTrainer: Uses a predefined trainer with built-in post-training logic, requiring
            only parameter configuration.

        Args:
            runtime: Optional reference to one of the existing runtimes. It can accept the runtime
                name or Runtime object from the `get_runtime()` API.
                Defaults to the torch-distributed runtime if not provided.
            initializer: Optional configuration for the dataset and model initializers.
            trainer: Optional configuration for a CustomTrainer, CustomTrainerContainer, or
                BuiltinTrainer. If not specified, the TrainJob will use the
                runtime's default values.
            options: Optional list of configuration options to apply to the TrainJob.
                Options can be imported from kubeflow.trainer.options.

        Returns:
            The unique name of the TrainJob that has been generated.

        Raises:
            ValueError: Input arguments are invalid.
            TimeoutError: Timeout to create TrainJobs.
            RuntimeError: Failed to create TrainJobs.
        """
        with self._tracer.start_as_current_span("train job", kind=SpanKind.INTERNAL) as span:
            try:
                runtime_name = (
                    runtime if isinstance(runtime, str) else getattr(runtime, "name", "unknown")
                )
                span.set_attribute("runtime.name", runtime_name)
                span.set_attribute("trainer.type", type(trainer).__name__ if trainer else "none")

                result = self.backend.train(
                    runtime=runtime,
                    initializer=initializer,
                    trainer=trainer,
                    options=options,
                )
                span.set_attribute("trainjob.name", result)
                span.set_status(Status(StatusCode.OK))
                return result
            except TimeoutError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            except RuntimeError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            except ValueError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        """List of the created TrainJobs. If a runtime is specified, only TrainJobs associated with
        that runtime are returned.

        Args:
            runtime: Reference to one of the existing runtimes.

        Returns:
            List of created TrainJobs. If no TrainJobs exist, an empty list is returned.

        Raises:
            TimeoutError: Timeout to list TrainJobs.
            RuntimeError: Failed to list TrainJobs.
        """
        with self._tracer.start_as_current_span("list jobs", kind=SpanKind.INTERNAL) as span:
            try:
                if runtime:
                    span.set_attribute("runtime.name", runtime.name)
                result = self.backend.list_jobs(runtime=runtime)
                span.set_attribute("jobs.count", len(result))
                span.set_status(Status(StatusCode.OK))
            except TimeoutError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            except RuntimeError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            return result

    def get_job(self, name: str) -> types.TrainJob:
        """Get the TrainJob object.

        Args:
            name: Name of the TrainJob.

        Returns:
            A TrainJob object.

        Raises:
            TimeoutError: Timeout to get a TrainJob.
            RuntimeError: Failed to get a TrainJob.
        """
        with self._tracer.start_as_current_span("get job", kind=SpanKind.INTERNAL) as span:
            try:
                span.set_attribute("trainjob.name", name)
                result = self.backend.get_job(name=name)
                span.set_status(Status(StatusCode.OK))
            except TimeoutError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            except RuntimeError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            return result

    def get_job_logs(
        self,
        name: str,
        step: str = constants.NODE + "-0",
        follow: bool | None = False,
    ) -> Iterator[str]:
        """Get logs from a specific step of a TrainJob.

        You can watch for the logs in realtime as follows:
        ```python
        from kubeflow.trainer import TrainerClient

        for logline in TrainerClient().get_job_logs(name="s8d44aa4fb6d", follow=True):
            print(logline)
        ```

        Args:
            name: Name of the TrainJob.
            step: Step of the TrainJob to collect logs from, like dataset-initializer or node-0.
            follow: Whether to stream logs in realtime as they are produced.

        Returns:
            Iterator of log lines.


        Raises:
            TimeoutError: Timeout to get a TrainJob.
            RuntimeError: Failed to get a TrainJob.
        """
        with self._tracer.start_as_current_span("get job logs", kind=SpanKind.INTERNAL) as span:
            try:
                span.set_attribute("trainjob.name", name)
                span.set_attribute("logs.step", step)
                span.set_attribute("logs.follow", follow)
                result = self.backend.get_job_logs(name=name, follow=follow, step=step)
                span.set_status(Status(StatusCode.OK))
            except TimeoutError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            except RuntimeError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            return result

    def get_job_events(self, name: str) -> list[types.Event]:
        """Get events for a TrainJob.

        This provides additional clarity about the state of the TrainJob
        when logs alone are not sufficient. Events include information about
        pod state changes, errors, and other significant occurrences.

        Args:
            name: Name of the TrainJob.

        Returns:
            A list of Event objects associated with the TrainJob.

        Raises:
            TimeoutError: Timeout to get a TrainJob events.
            RuntimeError: Failed to get a TrainJob events.
        """
        with self._tracer.start_as_current_span("get job events", kind=SpanKind.INTERNAL) as span:
            try:
                span.set_attribute("trainjob.name", name)
                result = self.backend.get_job_events(name=name)
                span.set_status(Status(StatusCode.OK))
            except TimeoutError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            except RuntimeError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            return result

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: list[Callable[[types.TrainJob], None]] | None = None,
    ) -> types.TrainJob:
        """Wait for a TrainJob to reach a desired status.

        Args:
            name: Name of the TrainJob.
            status: Expected statuses. Must be a subset of Created, Running, Complete, and
                Failed statuses.
            timeout: Maximum number of seconds to wait for the TrainJob to reach one of the
                expected statuses.
            polling_interval: The polling interval in seconds to check TrainJob status.
            callbacks: Optional list of callback functions to be invoked after each polling
                interval. Each callback should accept a single argument: the TrainJob object.

        Returns:
            A TrainJob object that reaches the desired status.

        Raises:
            ValueError: The input values are incorrect.
            RuntimeError: Failed to get TrainJob or TrainJob reaches unexpected Failed status.
            TimeoutError: Timeout to wait for TrainJob status.
        """
        with self._tracer.start_as_current_span(
            "wait for job status", kind=SpanKind.INTERNAL
        ) as span:
            try:
                span.set_attribute("trainjob.name", name)
                span.set_attribute("wait.timeout_seconds", timeout)
                span.set_attribute("wait.target_statuses", str(status))
                result = self.backend.wait_for_job_status(
                    name=name,
                    status=status,
                    timeout=timeout,
                    polling_interval=polling_interval,
                    callbacks=callbacks,
                )
                span.set_status(Status(StatusCode.OK))
                return result

            except ValueError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            except RuntimeError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            except TimeoutError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def delete_job(self, name: str):
        """Delete the TrainJob.

        Args:
            name: Name of the TrainJob.

        Raises:
            TimeoutError: Timeout to delete TrainJob.
            RuntimeError: Failed to delete TrainJob.
        """
        with self._tracer.start_as_current_span("delete job", kind=SpanKind.INTERNAL) as span:
            try:
                span.set_attribute("trainjob.name", name)
                result = self.backend.delete_job(name=name)
                span.set_status(Status(StatusCode.OK))
            except TimeoutError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            except RuntimeError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            return result
