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

import sys
import time
from unittest.mock import MagicMock

from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.constants import constants


def test_local_job_success():
    command = [sys.executable, "-c", "print('hello world')"]
    job = LocalJob(name="test-success", command=command)

    job.start()
    job.join(timeout=5)

    assert job.success is True
    assert job.status == constants.TRAINJOB_COMPLETE
    assert "hello world" in job.stdout
    assert job.returncode == 0
    assert job.creation_time is not None
    assert job.completion_time is not None


def test_local_job_failure():
    command = [sys.executable, "-c", "import sys; sys.exit(1)"]
    job = LocalJob(name="test-failure", command=command)

    job.start()
    job.join(timeout=5)

    assert job.success is False
    assert job.status == constants.TRAINJOB_FAILED
    assert job.returncode == 1


def test_local_job_dependency_failure():
    mock_dep = MagicMock()
    mock_dep.join = MagicMock()
    mock_dep.success = False
    mock_dep.name = "mock-dep"

    command = [sys.executable, "-c", "print('should not run')"]
    job = LocalJob(name="test-dep-fail", command=command, dependencies=[mock_dep])

    job.start()
    job.join(timeout=5)

    assert job.success is False
    assert job.status == constants.TRAINJOB_CREATED
    assert "Dependency mock-dep failed. Skipping" in job.stdout


def test_local_job_cancel():
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    job = LocalJob(name="test-cancel", command=command)

    job.start()
    # Wait briefly to ensure the subprocess spawns before we cancel
    time.sleep(0.5)
    job.cancel()
    job.join(timeout=5)

    assert job.success is False
    assert job.status == constants.TRAINJOB_FAILED
    assert "[JobCancelled]" in job.stdout


def test_local_job_stream_logs():
    script = (
        "import sys\nimport time\n"
        "print('line1')\nsys.stdout.flush()\n"
        "time.sleep(0.5)\n"
        "print('line2')\nsys.stdout.flush()\n"
    )
    command = [sys.executable, "-c", script]
    job = LocalJob(name="test-stream", command=command)

    job.start()

    streamed_output = ""
    for chunk in job.stream_logs():
        streamed_output += chunk

    job.join(timeout=5)

    assert "line1" in streamed_output
    assert "line2" in streamed_output
    assert job.success is True
