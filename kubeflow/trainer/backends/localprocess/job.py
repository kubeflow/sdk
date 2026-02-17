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
from datetime import datetime
import logging
import os
import subprocess
import threading

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
    ):
        """Creates a LocalJob.

        Creates a local subprocess with threading to allow users to create background jobs.

        Args:
            name (str): The name of the job.
            command (str): The command to run.
            execution_dir (str): The execution directory.
            env (Dict[str, str], optional): Environment variables. Defaults to None.
            dependencies (List[str], optional): List of dependencies. Defaults to None.
        """
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

    def run(self):
        for dep in self.dependencies:
            dep.join()
            if not dep.success:
                with self._lock:
                    self._stdout = f"Dependency {dep.name} failed. Skipping"
                return

        current_dir = os.getcwd()
        try:
            self._start_time = datetime.now()
            _c = " ".join(self.command)
            logger.debug(f"[{self.name}] Started at {self._start_time} with command: \n {_c}")

            # change working directory to venv before executing script
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
            # set job status
            self._status = constants.TRAINJOB_RUNNING

            while True:
                if self._cancel_requested.is_set():
                    self._process.terminate()
                    self._stdout += "[JobCancelled]\n"
                    self._status = constants.TRAINJOB_FAILED
                    self._success = False
                    return

                # Read output line by line (for streaming)
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
            self._success = self._process.returncode == 0
            msg = (
                f"[{self.name}] Completed with code {self._returncode}"
                f" in {self._end_time - self._start_time} seconds."
            )
            # set status based on success or failure
            self._status = (
                constants.TRAINJOB_COMPLETE if self._success else (constants.TRAINJOB_FAILED)
            )
            self._stdout += msg
            logger.debug("Job output: ", self._stdout)

        except Exception as e:
            with self._lock:
                self._stdout += f"Exception: {e}\n"
                self._success = False
                self._status = constants.TRAINJOB_FAILED
        finally:
            os.chdir(current_dir)

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
                print(chunk, end="", flush=True)  # stream to console live
        except StopIteration:
            pass

        return self._stdout.splitlines()

    def stream_logs(self):
        """Generator that yields new output lines as they come in."""
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
