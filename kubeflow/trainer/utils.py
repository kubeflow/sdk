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

"""Utilities for Kubeflow Trainer.

This module provides utility functions for training scripts running inside
Kubeflow TrainJobs, including `update_runtime_status()` for reporting progress.

Environment variables (injected by controller when TrainJobProgress feature gate is enabled):
    - KUBEFLOW_TRAINER_STATUS_URL: HTTPS endpoint URL for status updates
    - KUBEFLOW_TRAINER_STATUS_CA_CERT: Path to CA certificate for TLS verification
    - KUBEFLOW_TRAINER_STATUS_TOKEN: Path to projected service account token

Example:
    ```python
    from kubeflow.trainer.utils import update_runtime_status

    update_runtime_status(progress_percent=0, force=True)  # Start

    for step in range(total_steps):
        loss = train_step(batch)
        update_runtime_status(
            progress_percent=int((step / total_steps) * 100),
            metrics={"loss": loss, "step": step}
        )  # SDK handles throttling

    update_runtime_status(progress_percent=100, force=True)  # End
    ```

Note:
    For HuggingFace Transformers, use KubeflowCallback which calls this automatically.
    See: https://github.com/huggingface/transformers/pull/44487
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from requests import Session

__all__ = ["update_runtime_status"]

logger = logging.getLogger(__name__)

# Environment variable names injected by the controller
_ENV_STATUS_URL = "KUBEFLOW_TRAINER_STATUS_URL"
_ENV_CA_CERT = "KUBEFLOW_TRAINER_STATUS_CA_CERT"
_ENV_TOKEN_PATH = "KUBEFLOW_TRAINER_STATUS_TOKEN"

# Module-level state for throttling and caching
_lock = threading.Lock()
_last_update_time: float = 0.0
_cached_token: str | None = None
_token_read_time: float = 0.0
_session: Session | None = None

# Configuration
_MIN_UPDATE_INTERVAL_SECONDS = 5.0
_TOKEN_CACHE_TTL_SECONDS = 300.0


def _get_session() -> Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _get_cached_token(token_path: str) -> str | None:
    global _cached_token, _token_read_time

    now = time.monotonic()
    if _cached_token and (now - _token_read_time) < _TOKEN_CACHE_TTL_SECONDS:
        return _cached_token

    if not token_path or not os.path.exists(token_path):
        return None

    try:
        with open(token_path) as f:
            _cached_token = f.read().strip()
            _token_read_time = now
            return _cached_token
    except OSError:
        return None


def _should_throttle() -> bool:
    now = time.monotonic()
    return (now - _last_update_time) < _MIN_UPDATE_INTERVAL_SECONDS


def _update_last_time() -> None:
    global _last_update_time
    _last_update_time = time.monotonic()


def update_runtime_status(
    progress_percent: int | None = None,
    estimated_time_remaining: int | timedelta | None = None,
    metrics: dict[str, float | int | str] | None = None,
    force: bool = False,
) -> bool:
    """Report training progress to Kubeflow Trainer controller.

    Safe to call in any environment. Returns False silently if not running
    inside a Kubeflow TrainJob. Never raises exceptions.

    Includes automatic throttling (max 1 update per 5 seconds) to avoid
    overwhelming the controller. Use force=True to bypass throttling for
    critical updates like training start (0%) and end (100%).

    Args:
        progress_percent: Training completion percentage (0-100).
        estimated_time_remaining: ETA in seconds or as a timedelta.
        metrics: Dict of metric name -> value. Values are converted to strings.
        force: If True, bypass throttling. Use for start/end events.

    Returns:
        True if update was sent successfully, False otherwise.
    """
    try:
        url = os.environ.get(_ENV_STATUS_URL)
        if not url:
            return False

        with _lock:
            if not force and _should_throttle():
                return False
            _update_last_time()

        ca_file = os.environ.get(_ENV_CA_CERT)
        token_path = os.environ.get(_ENV_TOKEN_PATH)

        token = _get_cached_token(token_path)
        if not token:
            logger.debug("No authentication token available")
            return False

        trainer_status: dict = {
            "lastUpdatedTime": datetime.now(timezone.utc).isoformat(),
        }

        if progress_percent is not None:
            trainer_status["progressPercentage"] = max(0, min(100, progress_percent))

        if estimated_time_remaining is not None:
            if isinstance(estimated_time_remaining, timedelta):
                estimated_time_remaining = int(estimated_time_remaining.total_seconds())
            trainer_status["estimatedRemainingSeconds"] = max(0, estimated_time_remaining)

        if metrics:
            trainer_status["metrics"] = [
                {"name": str(k), "value": str(v)} for k, v in metrics.items()
            ]

        session = _get_session()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        verify: bool | str = True
        if ca_file and os.path.exists(ca_file):
            verify = ca_file

        response = session.post(
            url,
            json={"trainerStatus": trainer_status},
            headers=headers,
            verify=verify,
            timeout=5,
        )

        if response.status_code == 200:
            logger.debug(f"Status update sent: {progress_percent}%")
            return True
        else:
            logger.warning(f"Status update failed: {response.status_code} {response.text}")
            return False

    except Exception as e:
        logger.warning(f"Failed to send status update: {e}")
        return False
