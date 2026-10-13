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

import sys

from kubeflow.trainer.backends.localprocess.job import LocalJob

# A child script that reports whether it received an inherited PATH and the
# value of a custom variable. Uses sys.executable so the interpreter itself is
# resolvable via an absolute path regardless of the child's PATH, isolating the
# test to the env-inheritance behaviour under test.
_PROBE = (
    "import os;"
    "print('HAS_PATH', bool(os.environ.get('PATH')));"
    "print('CUSTOM', os.environ.get('KF_TEST_CUSTOM'))"
)


def _run(env):
    job = LocalJob(name="env-probe", command=[sys.executable, "-c", _PROBE], env=env)
    job.start()
    job.join(timeout=30)
    return job


def test_local_job_inherits_parent_env_when_env_not_set():
    """With no custom env, the child must still inherit the parent environment
    (e.g. PATH). Passing env=None previously wiped it via subprocess.Popen."""
    job = _run(None)

    assert job.success, job.stdout
    assert "HAS_PATH True" in job.stdout


def test_local_job_merges_custom_env_with_parent_env():
    """Custom env vars must be merged on top of the inherited environment, not
    replace it, so both the custom var and PATH are present in the child."""
    job = _run({"KF_TEST_CUSTOM": "hello"})

    assert job.success, job.stdout
    assert "HAS_PATH True" in job.stdout
    assert "CUSTOM hello" in job.stdout
