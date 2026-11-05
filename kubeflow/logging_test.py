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
import logging
import subprocess
import sys

LIBRARY_LOGGING_SCRIPT = """
import logging
import kubeflow
logging.getLogger("kubeflow.trainer").warning("library warning")
"""


def test_kubeflow_logger_has_null_handler():
    handlers = logging.getLogger("kubeflow").handlers

    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


def test_import_is_silent_without_logging_configuration():
    # Without a handler on the "kubeflow" logger, Python's last-resort handler would print
    # warnings to stderr in an application that never configured logging.
    result = subprocess.run(
        [sys.executable, "-c", LIBRARY_LOGGING_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stderr == ""


def test_application_can_opt_in_to_sdk_logs(caplog):
    with caplog.at_level(logging.DEBUG, logger="kubeflow"):
        logging.getLogger("kubeflow.trainer").debug("opted in")

    assert "opted in" in caplog.text
