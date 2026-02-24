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
import os

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name.

        The log level can be configured via the KUBEFLOW_LOG_LEVEL environment
        variable. Defaults to INFO if not set.

        Example:
    ```python
            from kubeflow.common.logging import get_logger
            logger = get_logger(__name__)
            logger.info("Training started")
    ```

        Args:
            name: The logger name, typically __name__ of the calling module.

        Returns:
            A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    log_level = os.environ.get("KUBEFLOW_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    return logger
