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

"""Unit tests for the telemetry module.

Tests verify two scenarios:
1. OTel is installed: spans are created with correct names and attributes.
2. OTel is absent: all operations no-op without errors.

Since ``opentelemetry-api`` is an optional dependency and may not be installed
in the dev environment, the "OTel installed" tests inject fake modules into
``sys.modules`` rather than using ``unittest.mock.patch`` on the real package.
"""

import sys
from unittest.mock import MagicMock

import pytest

import kubeflow.common.telemetry as telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    """Reset module-level state before each test.

    The telemetry module uses a module-level ``_initialized`` flag to ensure
    ``_setup()`` runs only once. Tests need a clean slate each time, so we
    reset the flag and singletons before every test.
    """
    telemetry._initialized = False
    telemetry._tracer = None
    telemetry._meter = None
    yield
    telemetry._initialized = False
    telemetry._tracer = None
    telemetry._meter = None


@pytest.fixture()
def mock_otel():
    """Inject fake opentelemetry modules into sys.modules.

    This fixture creates mock ``opentelemetry``, ``opentelemetry.trace``, and
    ``opentelemetry.metrics`` modules with controllable ``get_tracer`` and
    ``get_meter`` functions. It cleans up sys.modules after the test.

    Yields a dict with ``tracer``, ``meter``, ``get_tracer``, and ``get_meter``
    mocks for assertions.
    """
    mock_tracer = MagicMock(name="mock_tracer")
    mock_meter = MagicMock(name="mock_meter")

    mock_trace_mod = MagicMock()
    mock_trace_mod.get_tracer = MagicMock(return_value=mock_tracer)

    mock_metrics_mod = MagicMock()
    mock_metrics_mod.get_meter = MagicMock(return_value=mock_meter)

    mock_otel_mod = MagicMock()
    mock_otel_mod.trace = mock_trace_mod
    mock_otel_mod.metrics = mock_metrics_mod

    # Save any existing entries so we can restore them.
    saved = {}
    otel_keys = ["opentelemetry", "opentelemetry.trace", "opentelemetry.metrics"]
    for key in otel_keys:
        if key in sys.modules:
            saved[key] = sys.modules[key]

    sys.modules["opentelemetry"] = mock_otel_mod
    sys.modules["opentelemetry.trace"] = mock_trace_mod
    sys.modules["opentelemetry.metrics"] = mock_metrics_mod

    yield {
        "tracer": mock_tracer,
        "meter": mock_meter,
        "get_tracer": mock_trace_mod.get_tracer,
        "get_meter": mock_metrics_mod.get_meter,
    }

    # Restore original state.
    for key in otel_keys:
        if key in saved:
            sys.modules[key] = saved[key]
        else:
            sys.modules.pop(key, None)


@pytest.fixture()
def mock_no_otel():
    """Ensure opentelemetry is NOT importable.

    Temporarily removes any opentelemetry entries from sys.modules and installs
    an import hook that blocks the import.
    """
    saved = {}
    otel_keys = [k for k in sys.modules if k.startswith("opentelemetry")]
    for key in otel_keys:
        saved[key] = sys.modules.pop(key)

    original_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def blocking_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = blocking_import
    else:
        __builtins__.__import__ = blocking_import

    yield

    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = original_import
    else:
        __builtins__.__import__ = original_import

    for key, mod in saved.items():
        sys.modules[key] = mod


class TestSetupWithOTel:
    """Tests for when opentelemetry-api is installed."""

    def test_setup_creates_tracer_and_meter(self, mock_otel):
        """_setup() should create a tracer and meter from the OTel API."""
        telemetry._setup()

        mock_otel["get_tracer"].assert_called_once_with("kubeflow.sdk", "0.5.0")
        mock_otel["get_meter"].assert_called_once_with("kubeflow.sdk", "0.5.0")
        assert telemetry._tracer is mock_otel["tracer"]
        assert telemetry._meter is mock_otel["meter"]

    def test_setup_runs_only_once(self, mock_otel):
        """_setup() should only attempt the import once, even if called multiple times."""
        telemetry._setup()
        telemetry._setup()
        telemetry._setup()

        mock_otel["get_tracer"].assert_called_once()

    def test_get_tracer_returns_tracer(self, mock_otel):
        """get_tracer() should return the OTel tracer."""
        result = telemetry.get_tracer()
        assert result is mock_otel["tracer"]

    def test_get_meter_returns_meter(self, mock_otel):
        """get_meter() should return the OTel meter."""
        result = telemetry.get_meter()
        assert result is mock_otel["meter"]


class TestSetupWithoutOTel:
    """Tests for when opentelemetry-api is NOT installed."""

    def test_setup_without_otel_sets_none(self, mock_no_otel):
        """When OTel is missing, _setup() should set tracer and meter to None."""
        telemetry._setup()

        assert telemetry._tracer is None
        assert telemetry._meter is None
        assert telemetry._initialized is True

    def test_get_tracer_returns_none_without_otel(self, mock_no_otel):
        """get_tracer() should return None when OTel is not installed."""
        assert telemetry.get_tracer() is None

    def test_get_meter_returns_none_without_otel(self, mock_no_otel):
        """get_meter() should return None when OTel is not installed."""
        assert telemetry.get_meter() is None


class TestSdkSpan:
    """Tests for the sdk_span context manager."""

    def test_sdk_span_creates_span_with_otel(self, mock_otel):
        """sdk_span() should create a real span when OTel is available."""
        mock_span = MagicMock()
        mock_otel["tracer"].start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_otel["tracer"].start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with telemetry.sdk_span(
            "kubeflow.trainer.train",
            attributes={"kubeflow.namespace": "default"},
        ) as span:
            assert span is mock_span

        mock_otel["tracer"].start_as_current_span.assert_called_once_with(
            "kubeflow.trainer.train",
            attributes={"kubeflow.namespace": "default"},
        )

    def test_sdk_span_yields_none_without_otel(self):
        """sdk_span() should yield None and not error when OTel is absent."""
        telemetry._initialized = True
        telemetry._tracer = None

        with telemetry.sdk_span("kubeflow.trainer.train") as span:
            assert span is None

    def test_sdk_span_allows_attribute_setting_on_span(self, mock_otel):
        """Callers should be able to set attributes on the yielded span."""
        mock_span = MagicMock()
        mock_otel["tracer"].start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_otel["tracer"].start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with telemetry.sdk_span("kubeflow.trainer.train") as span:
            if span:
                span.set_attribute("kubeflow.trainjob.name", "my-job")

        mock_span.set_attribute.assert_called_once_with("kubeflow.trainjob.name", "my-job")

    def test_sdk_span_no_error_when_setting_attributes_without_otel(self):
        """The ``if span:`` guard pattern should safely skip when OTel is absent."""
        telemetry._initialized = True
        telemetry._tracer = None

        # This must not raise.
        with telemetry.sdk_span("kubeflow.trainer.train") as span:
            if span:
                span.set_attribute("kubeflow.trainjob.name", "my-job")

    def test_sdk_span_with_no_attributes(self, mock_otel):
        """sdk_span() should work when called without attributes."""
        mock_span = MagicMock()
        mock_otel["tracer"].start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_otel["tracer"].start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with telemetry.sdk_span("kubeflow.trainer.list_jobs") as span:
            assert span is mock_span

        mock_otel["tracer"].start_as_current_span.assert_called_once_with(
            "kubeflow.trainer.list_jobs",
            attributes=None,
        )
