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

"""
Unit tests for the LocalJob class in the Kubeflow Trainer SDK.
"""

import logging
import sys

import pytest

from kubeflow.trainer.backends.localprocess import job as job_module
from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.constants import constants


def test_run_logs_job_output_at_debug(caplog, monkeypatch):
    """LocalJob.run() must log the completed job output without a logging error.

    Regression test for the debug log using printf-style arguments without a
    format placeholder, which raised TypeError at record-format time (swallowed
    by logging.Handler.handleError) instead of logging the job output.
    """
    marker = "kubeflow-local-job-output-marker"

    # Any logging-format failure is normally swallowed by Handler.handleError; make
    # it fail the test loudly so a broken log call cannot pass silently.
    def fail_on_handle_error(self, record):
        raise AssertionError("logging raised while formatting the job output record")

    monkeypatch.setattr(logging.Handler, "handleError", fail_on_handle_error)

    job = LocalJob(name="test-job", command=[sys.executable, "-c", f"print('{marker}')"])

    with caplog.at_level(logging.DEBUG, logger=job_module.logger.name):
        job.run()

    assert job.status == constants.TRAINJOB_COMPLETE
    assert any(
        record.getMessage() == f"Job output: {job.stdout}"
        for record in caplog.records
        if record.levelno == logging.DEBUG
    )


@pytest.mark.parametrize(
    "stdout_fragment",
    [
        "plain output",
        "100% done",
        "value {with braces}",
    ],
)
def test_run_debug_log_formats_with_percent_in_output(caplog, monkeypatch, stdout_fragment):
    """The debug log must format even when the job output contains a percent sign."""

    def fail_on_handle_error(self, record):
        raise AssertionError("logging raised while formatting the job output record")

    monkeypatch.setattr(logging.Handler, "handleError", fail_on_handle_error)

    job = LocalJob(
        name="test-job",
        command=[sys.executable, "-c", f"print({stdout_fragment!r})"],
    )

    with caplog.at_level(logging.DEBUG, logger=job_module.logger.name):
        job.run()

    assert job.status == constants.TRAINJOB_COMPLETE
    assert stdout_fragment in job.stdout
