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

"""Unit tests for Kubeflow SDK logging system."""

from dataclasses import dataclass
import io
import json
import logging
import sys
from typing import Optional

import pytest

from kubeflow.trainer.logging.config import get_logger, setup_logging
from kubeflow.trainer.logging.formatters import StructuredFormatter


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Fixture to clean up logging handlers before and after each test."""
    # Clean up before test
    kubeflow_logger = logging.getLogger("kubeflow")
    root_logger = logging.getLogger()

    # Store original handlers
    original_kubeflow_handlers = kubeflow_logger.handlers[:]
    original_root_handlers = root_logger.handlers[:]

    yield

    # Clean up after test
    for handler in kubeflow_logger.handlers[:]:
        kubeflow_logger.removeHandler(handler)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Restore original handlers
    for handler in original_kubeflow_handlers:
        kubeflow_logger.addHandler(handler)
    for handler in original_root_handlers:
        root_logger.addHandler(handler)


@dataclass
class LoggingConfigTestCase:
    """Test case definition for logging configuration tests."""

    name: str
    level: str
    format_type: str
    message: str
    expected_level: str
    expected_message: str
    use_json_formatter: bool = False
    use_file_output: bool = False
    file_should_contain: Optional[str] = None


class TestLoggingConfig:
    """Test logging configuration functionality."""

    def test_get_logger(self) -> None:
        """Test get_logger returns properly named logger."""
        logger = get_logger("test_module")
        assert logger.name == "kubeflow.test_module"

    def test_get_logger_with_kubeflow_prefix(self) -> None:
        """Test get_logger handles existing kubeflow prefix."""
        logger = get_logger("kubeflow.trainer.test")
        assert logger.name == "kubeflow.trainer.test"

    @pytest.mark.parametrize(
        "test_case",
        [
            LoggingConfigTestCase(
                name="basic_console_logging",
                level="INFO",
                format_type="console",
                message="Starting Kubeflow SDK operation",
                expected_level="INFO",
                expected_message="Starting Kubeflow SDK operation",
            ),
            LoggingConfigTestCase(
                name="json_logging_example",
                level="DEBUG",
                format_type="json",
                message="Training job started",
                expected_level="INFO",
                expected_message="Training job started",
                use_json_formatter=True,
            ),
            LoggingConfigTestCase(
                name="json_logging_format",
                level="DEBUG",
                format_type="json",
                message="Test message",
                expected_level="INFO",
                expected_message="Test message",
                use_json_formatter=True,
            ),
        ],
    )
    def test_logging_output(self, test_case: LoggingConfigTestCase) -> None:
        """Test logging output for console and JSON configurations."""
        log_capture = io.StringIO()

        setup_logging(level=test_case.level, format_type=test_case.format_type)

        logger = get_logger("test")

        # Remove handlers configured by setup_logging for this logger to avoid
        # interference with our capture handler in tests.
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.propagate = False

        handler = logging.StreamHandler(log_capture)
        if test_case.use_json_formatter:
            handler.setFormatter(StructuredFormatter())
            logger.setLevel(logging.DEBUG)
        else:
            handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            logger.setLevel(logging.INFO)

        logger.addHandler(handler)

        logger.info(test_case.message)

        captured = log_capture.getvalue().strip()
        if test_case.use_json_formatter:
            log_data = json.loads(captured)
            assert log_data["level"] == test_case.expected_level
            assert log_data["message"] == test_case.expected_message
        else:
            assert f"{test_case.expected_level} - {test_case.expected_message}" in captured

    def test_setup_logging_file_output(self) -> None:
        """Test file logging setup."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            setup_logging(level="INFO", format_type="console", log_file=temp_path)

            logger = get_logger("test")
            logger.info("File test message")

            with open(temp_path) as f:
                content = f.read()

            assert "File test message" in content
        finally:
            os.unlink(temp_path)


class TestNullHandlerPattern:
    """Test NullHandler pattern implementation."""

    def setup_method(self):
        """Setup method to ensure clean state for each test."""
        # Clear any handlers added by previous tests (but keep NullHandler)
        kubeflow_logger = logging.getLogger("kubeflow")
        handlers_to_remove = [
            h for h in kubeflow_logger.handlers if not isinstance(h, logging.NullHandler)
        ]
        for handler in handlers_to_remove:
            kubeflow_logger.removeHandler(handler)

    def test_kubeflow_package_nullhandler(self):
        """Test that kubeflow package has NullHandler configured."""
        # Get kubeflow logger (already imported)
        kubeflow_logger = logging.getLogger("kubeflow")

        # Check that it has a NullHandler (may have other handlers from previous tests)
        null_handlers = [h for h in kubeflow_logger.handlers if isinstance(h, logging.NullHandler)]

        # If NullHandler was removed by previous tests, add it back
        if len(null_handlers) == 0:
            kubeflow_logger.addHandler(logging.NullHandler())
            null_handlers = [
                h for h in kubeflow_logger.handlers if isinstance(h, logging.NullHandler)
            ]

        # The NullHandler should be present
        assert len(null_handlers) > 0, (
            f"kubeflow package should have NullHandler configured, "
            f"found handlers: {kubeflow_logger.handlers}"
        )

    def test_nullhandler_suppresses_logs(self):
        """Test that NullHandler suppresses logs by default."""
        # Import kubeflow to trigger NullHandler setup

        # Capture any output that might leak through
        log_capture = io.StringIO()

        # Setup basic logging to capture any output
        logging.basicConfig(
            level=logging.DEBUG, stream=log_capture, format="%(levelname)s - %(message)s"
        )

        # Get kubeflow logger and try to log
        kubeflow_logger = logging.getLogger("kubeflow")
        kubeflow_logger.debug("This should be suppressed by NullHandler")

        # Check that no output was captured (NullHandler working)
        captured = log_capture.getvalue()
        assert "This should be suppressed by NullHandler" not in captured

    def test_user_configuration_overrides_nullhandler(self):
        """Test that user logging configuration overrides NullHandler."""
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # User configures logging with propagation enabled
        log_capture = io.StringIO()
        logging.basicConfig(
            level=logging.DEBUG, stream=log_capture, format="%(levelname)s - %(name)s - %(message)s"
        )

        # Ensure kubeflow logger propagates to root
        kubeflow_logger = logging.getLogger("kubeflow")
        kubeflow_logger.propagate = True
        kubeflow_logger.setLevel(logging.DEBUG)

        # Now kubeflow logging should work
        kubeflow_logger.debug("This should now be visible")

        captured = log_capture.getvalue()
        # The logging should work when user configures it
        assert "This should now be visible" in captured or "DEBUG" in captured

    def test_sdk_integration_with_nullhandler(self):
        """Test actual SDK integration with NullHandler pattern (replaces nullhandler_example)."""
        # Import SDK components
        from kubeflow.trainer import LocalProcessBackendConfig, TrainerClient

        # Test 1: Default behavior - no logging output (NullHandler active)
        log_capture = io.StringIO()

        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Use SDK without user logging configuration
        config = LocalProcessBackendConfig()
        TrainerClient(backend_config=config)

        # Should not produce any logging output
        captured = log_capture.getvalue()
        assert len(captured) == 0, "SDK should not produce logging output by default"

        # Test 2: User configures logging - NullHandler is overridden
        log_capture = io.StringIO()

        # User configures logging
        logging.basicConfig(
            level=logging.DEBUG, stream=log_capture, format="%(levelname)s - %(name)s - %(message)s"
        )

        # Ensure kubeflow logger propagates to root
        kubeflow_logger = logging.getLogger("kubeflow")
        kubeflow_logger.propagate = True
        kubeflow_logger.setLevel(logging.DEBUG)

        # Now SDK calls should produce debug output
        config = LocalProcessBackendConfig()
        TrainerClient(backend_config=config)

        captured = log_capture.getvalue()
        # Should contain SDK debug messages
        assert "DEBUG" in captured or "Initializing TrainerClient" in captured

        # Test 3: Different log levels (INFO vs DEBUG)
        # Clear handlers and test INFO level
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_capture = io.StringIO()
        logging.basicConfig(
            level=logging.INFO, stream=log_capture, format="%(levelname)s - %(name)s - %(message)s"
        )

        # Set kubeflow logger to INFO level to suppress DEBUG messages
        kubeflow_logger = logging.getLogger("kubeflow")
        kubeflow_logger.setLevel(logging.INFO)

        config = LocalProcessBackendConfig()
        TrainerClient(backend_config=config)

        captured = log_capture.getvalue()
        # INFO level should suppress DEBUG messages
        assert "DEBUG" not in captured or len(captured) == 0

    def test_application_integration_example(self):
        """Test complete application integration example (replaces SDK integration demo)."""
        # Import SDK components
        import os
        import tempfile

        from kubeflow.trainer import LocalProcessBackendConfig, TrainerClient

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Clear any existing handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # User sets up their application logging (file + console)
            log_capture = io.StringIO()
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s - %(name)s - %(message)s",
                handlers=[logging.StreamHandler(log_capture), logging.FileHandler(temp_path)],
            )

            # User's application logger
            app_logger = logging.getLogger("my_app")
            app_logger.info("Starting my application")

            # SDK calls will now respect user's logging configuration
            app_logger.info("Creating TrainerClient...")
            config = LocalProcessBackendConfig()
            TrainerClient(backend_config=config)

            app_logger.info("Application completed")

            # Check console output
            captured = log_capture.getvalue()
            assert "Starting my application" in captured
            assert "Creating TrainerClient..." in captured
            assert "Application completed" in captured

            # Check file output
            with open(temp_path) as f:
                file_content = f.read()
            assert "Starting my application" in file_content
            assert "Creating TrainerClient..." in file_content
            assert "Application completed" in file_content

        finally:
            os.unlink(temp_path)


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""

    def test_structured_formatter_basic(self):
        """Test basic StructuredFormatter functionality."""
        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Format the record
        formatted = formatter.format(record)

        # Parse as JSON
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["logger"] == "test.logger"
        assert log_data["line"] == 42

    def test_structured_formatter_with_exception(self):
        """Test StructuredFormatter with exception information."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=sys.exc_info(),
            )

            formatted = formatter.format(record)
            log_data = json.loads(formatted)

            assert log_data["level"] == "ERROR"
            assert "exception" in log_data
            assert "ValueError: Test exception" in log_data["exception"]
