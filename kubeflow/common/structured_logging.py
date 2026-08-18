"""Structured logging utilities for the Kubeflow SDK."""

from __future__ import annotations

import logging
import os
import threading
import warnings

import structlog

_CONFIGURED = False
_CONFIGURE_LOCK = threading.Lock()


def configure_logging(level: str = "INFO", *, json_output: bool | None = None) -> None:
    """Configure structlog processors and stdlib logging for the kubeflow namespace.

    This is opt-in. Libraries should not call this at import time. Application
    entrypoints (scripts, CLI tools) should call it once at startup.

    Safe to call multiple times; only the first call takes effect. Subsequent
    calls emit a warning and are ignored.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit JSON lines; if False, human-readable console
            output. When None, JSON is used when the ``CI`` env var is set
            to a non-empty value other than "0" or "false".

    Raises:
        ValueError: If level is not a valid Python logging level name.
    """
    global _CONFIGURED
    with _CONFIGURE_LOCK:
        if _CONFIGURED:
            warnings.warn(
                "configure_logging() has already been called; ignoring subsequent call.",
                stacklevel=2,
            )
            return
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level!r}")

        if json_output is None:
            ci_val = os.environ.get("CI", "").lower()
            json_output = ci_val not in ("", "0", "false")

        shared_processors: list[structlog.types.Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        renderer: structlog.types.Processor = (
            structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
        )

        # Configure structlog globally. This is intentional here because
        # configure_logging() is an opt-in call by the application, not
        # triggered at library import time.
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        ns_logger = logging.getLogger("kubeflow")
        if ns_logger.handlers:
            warnings.warn(
                "The 'kubeflow' logger already has handlers; "
                "configure_logging() will append a structured handler.",
                stacklevel=2,
            )
        ns_logger.addHandler(handler)
        ns_logger.setLevel(numeric_level)
        ns_logger.propagate = False

        _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger wrapping a stdlib logger.

    Without a prior :func:`configure_logging` call, the returned logger routes
    through stdlib's logging infrastructure with no additional processors. This
    means it inherits whatever handler configuration the application has set up
    (safe library default).

    After :func:`configure_logging` is called, loggers use the configured
    structlog processors and renderer.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A structlog bound logger compatible with stdlib logging levels.
    """
    if _CONFIGURED:
        # After configure_logging(), structlog.get_logger() returns loggers
        # that use the configured processors and renderer.
        return structlog.get_logger(name)
    # Before configure_logging(), wrap a stdlib logger directly. We cannot use
    # structlog.get_logger() here because cache_logger_on_first_use=True (set
    # by configure_logging) would freeze these early loggers with the
    # pre-configuration factory, making them ignore a later configure_logging()
    # call. wrap_logger() bypasses that cache entirely.
    return structlog.wrap_logger(
        logging.getLogger(name),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
