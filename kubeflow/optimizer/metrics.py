# Copyright 2026 The Kubeflow Authors.
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

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from typing import Any


def report_metrics(
    metrics: Mapping[str, Any],
    *,
    step: int | None = None,
    epoch: int | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Report metrics for hyperparameter optimization.

    This helper prints a single JSON object to stdout. JSON output is commonly used by
    metrics collectors (e.g., Katib) to parse metrics emitted by training code.

    Args:
        metrics: Mapping of metric names to values.
        step: Optional step number.
        epoch: Optional epoch number.
        timestamp: Optional timestamp. If not provided, current UTC time is used.

    Raises:
        ValueError: If metrics is empty.
        TypeError: If metrics is not a mapping.
    """
    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")
    if not metrics:
        raise ValueError("metrics must be non-empty")

    ts = timestamp or datetime.now(timezone.utc)
    payload: dict[str, Any] = {"timestamp": ts.isoformat()}
    if step is not None:
        payload["step"] = step
    if epoch is not None:
        payload["epoch"] = epoch

    # Emit metric values as strings for collector compatibility.
    for name, value in metrics.items():
        payload[name] = str(value)

    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)

