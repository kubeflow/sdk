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

"""Logging configuration for Kubeflow SDK."""

import logging
import logging.config
import os
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = "INFO",
    format_type: str = "console",
    log_file: Optional[str] = None,
) -> None:
    """Setup logging configuration for Kubeflow SDK.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format type ('console', 'json', 'detailed')
        log_file: Optional log file path for file output
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "json": {
                "()": "kubeflow.trainer.logging.formatters.StructuredFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": format_type,
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "kubeflow": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    # Add file handler if log_file is specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": format_type,
            "filename": log_file,
            "mode": "a",
        }
        config["loggers"]["kubeflow"]["handlers"].append("file")
        config["root"]["handlers"].append("file")

    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Logger instance configured for Kubeflow SDK
    """
    # Ensure the logger name starts with 'kubeflow'
    if not name.startswith("kubeflow"):
        name = f"kubeflow.{name}"

    return logging.getLogger(name)


def configure_from_env() -> None:
    """Configure logging from environment variables.

    Environment variables:
        KUBEFLOW_LOG_LEVEL: Logging level (default: INFO)
        KUBEFLOW_LOG_FORMAT: Output format (default: console)
        KUBEFLOW_LOG_FILE: Log file path (optional)
    """
    level = os.getenv("KUBEFLOW_LOG_LEVEL", "INFO")
    format_type = os.getenv("KUBEFLOW_LOG_FORMAT", "console")
    log_file = os.getenv("KUBEFLOW_LOG_FILE")

    setup_logging(level=level, format_type=format_type, log_file=log_file)
