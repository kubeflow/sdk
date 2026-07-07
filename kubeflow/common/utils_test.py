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
import pytest

from kubeflow.common import utils


def test_validate_wait_for_job_status():
    # Valid case should not raise.
    utils.validate_wait_for_job_status(polling_interval=2, timeout=600)

    with pytest.raises(ValueError, match="Timeout must be a positive number"):
        utils.validate_wait_for_job_status(polling_interval=2, timeout=0)

    with pytest.raises(ValueError, match="Polling interval must be a positive number"):
        utils.validate_wait_for_job_status(polling_interval=0, timeout=600)

    with pytest.raises(ValueError, match="Polling interval must be a positive number"):
        utils.validate_wait_for_job_status(polling_interval=-5, timeout=600)

    with pytest.raises(ValueError, match="Polling interval must be strictly less than timeout"):
        utils.validate_wait_for_job_status(polling_interval=10, timeout=10)
