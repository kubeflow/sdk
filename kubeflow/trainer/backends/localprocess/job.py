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

from datetime import datetime
import logging
import os
import subprocess
import threading
from typing import TYPE_CHECKING

from opentelemetry import context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode

if TYPE_CHECKING:
    # Only imported when a type checker runs — never at runtime.
    # This keeps the runtime import clean (API only) while giving
    # editors and mypy the correct type information.
    from opentelemetry.sdk.trace import TracerProvider
from kubeflow.trainer.constants import constants

logger = logging.getLogger(__name__)


class LocalJob(threading.Thread):
    def __init__(
        self,
        name,
        command: list | tuple[str] | str,
        execution_dir: str = None,
        env: dict[str, str] = None,
        dependencies: list = None,
        parent_span_context: SpanContext | None = None,
        tracer_provider: TracerProvider | None = None,
    ):
        super().__init__()
        self.name = name
        self.command = command
        self._stdout = ""
        self._returncode = None
        self._success = False
        self._status = constants.TRAINJOB_CREATED
        self._lock = threading.Lock()
        self._process = None
        self._output_updated = threading.Event()
        self._cancel_requested = threading.Event()
        self._start_time = None
        self._end_time = None
        self.env = env or {}
        self.dependencies = dependencies or []
        self.execution_dir = execution_dir or os.getcwd()

        # Get a tracer from the same provider the rest of the SDK is using.
        _provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = _provider.get_tracer(__name__)

        # Reconstruct an OTel Context from the parent SpanContext so the span
        # opened in run() is linked as a child.
        #
        # Why not just store the SpanContext and pass it to start_as_current_span?
        # start_as_current_span takes a Context object, not a SpanContext.
        # SpanContext is just raw IDs (trace_id, span_id, flags).
        # Context is the thread-local bag that holds the "current span" slot.
        #
        # NonRecordingSpan wraps the IDs into something that looks like a span
        # to OTel — it can act as a parent (exposes get_span_context()) but
        # never records attributes or exports anything, so there's no risk of
        # double-exporting the parent which is already being recorded on the
        # calling thread.
        if parent_span_context and parent_span_context.is_valid:
            self._parent_otel_ctx = trace.set_span_in_context(NonRecordingSpan(parent_span_context))
        else:
            self._parent_otel_ctx = None

    def run(self):
        for dep in self.dependencies:
            dep.join()
            if not dep.success:
                with self._lock:
                    self._stdout = f"Dependency {dep.name} failed. Skipping"
                return

        current_dir = os.getcwd()

        # Attach the parent context onto this thread's local context stack.
        # After attach(), any span opened with start_as_current_span will
        # automatically find the parent and nest under it.
        # The token is a receipt — detach(token) rolls back to the previous
        # context state. Always call detach in finally — if you skip it the
        # context slot on this thread is permanently corrupted for any future
        # work the thread pool might run on this OS thread.
        token = context.attach(self._parent_otel_ctx) if self._parent_otel_ctx else None

        with self._tracer.start_as_current_span("localprocess.job.run") as span:
            span.set_attribute("job.name", self.name)
            span.set_attribute("job.execution_dir", self.execution_dir)
            cmd_str = (
                " ".join(self.command) if isinstance(self.command, (list, tuple)) else self.command
            )
            span.set_attribute("job.command", cmd_str)

            try:
                self._start_time = datetime.now()
                logger.debug(
                    f"[{self.name}] Started at {self._start_time} with command:\n{cmd_str}"
                )

                os.chdir(self.execution_dir)

                self._process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    env=self.env,
                )
                self._status = constants.TRAINJOB_RUNNING
                span.set_attribute("job.pid", self._process.pid)

                while True:
                    if self._cancel_requested.is_set():
                        self._process.terminate()
                        self._stdout += "[JobCancelled]\n"
                        self._status = constants.TRAINJOB_FAILED
                        self._success = False
                        span.set_status(StatusCode.ERROR, "Job cancelled")
                        span.set_attribute("job.cancelled", True)
                        return

                    output_line = self._process.stdout.readline()
                    with self._lock:
                        if output_line:
                            self._stdout += output_line
                            self._output_updated.set()

                    if not output_line and self._process.poll() is not None:
                        break

                self._process.stdout.close()
                self._returncode = self._process.wait()
                self._end_time = datetime.now()
                self._success = self._returncode == 0

                duration = (self._end_time - self._start_time).total_seconds()
                span.set_attribute("job.exit_code", self._returncode)
                span.set_attribute("job.duration_seconds", duration)

                self._status = (
                    constants.TRAINJOB_COMPLETE if self._success else constants.TRAINJOB_FAILED
                )
                self._stdout += (
                    f"[{self.name}] Completed with code {self._returncode} in {duration:.2f}s."
                )

                if self._success:
                    span.set_status(StatusCode.OK)
                else:
                    span.set_status(StatusCode.ERROR, f"Job exited with code {self._returncode}")

            except Exception as e:
                with self._lock:
                    self._stdout += f"Exception: {e}\n"
                    self._success = False
                    self._status = constants.TRAINJOB_FAILED
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, str(e))

            finally:
                os.chdir(current_dir)
                if token is not None:
                    context.detach(token)

    @property
    def stdout(self):
        with self._lock:
            return self._stdout

    @property
    def success(self):
        return self._success

    @property
    def status(self):
        return self._status

    def cancel(self):
        self._cancel_requested.set()

    @property
    def returncode(self):
        return self._returncode

    def logs(self, follow=False) -> list[str]:
        if not follow:
            return self._stdout.splitlines()
        try:
            for chunk in self.stream_logs():
                print(chunk, end="", flush=True)
        except StopIteration:
            pass
        return self._stdout.splitlines()

    def stream_logs(self):
        last_index = 0
        while self.is_alive() or last_index < len(self._stdout):
            self._output_updated.wait(timeout=1)
            with self._lock:
                data = self._stdout
                new_data = data[last_index:]
                last_index = len(data)
                self._output_updated.clear()
            if new_data:
                yield new_data

    @property
    def creation_time(self):
        return self._start_time

    @property
    def completion_time(self):
        return self._end_time
