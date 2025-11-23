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

"""Base client class for Spark SDK.

This module provides the abstract base class for all Spark clients,
implementing shared functionality like resource management, context
manager protocol, and logging.
"""

import abc
import logging
from typing import Any

from kubeflow.spark.backends.base import SparkBackend


class BaseSparkClient(abc.ABC):
    """Abstract base class for Spark clients.

    This class implements common functionality shared by all Spark client types:
    - Resource management (close() method)
    - Context manager protocol (__enter__/__exit__)
    - Logging infrastructure

    Subclasses (BatchSparkClient, SparkSessionClient) implement specific
    functionality for their use cases.

    This design follows the Template Method Pattern, where the base class
    defines the skeleton of operations and subclasses fill in specific steps.
    """

    def __init__(self, backend: SparkBackend):
        """Initialize the base client.

        Args:
            backend: Spark backend instance (BatchSparkBackend or SessionSparkBackend)
        """
        self._backend = backend
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info(f"Initialized {self.__class__.__name__} with {backend.__class__.__name__}")

    def close(self):
        """Close the client and release all resources.

        This method delegates to the backend's close() method to clean up:
        - Kubernetes API clients
        - HTTP connections
        - gRPC channels
        - Active sessions

        It's safe to call this multiple times.
        """
        try:
            self._backend.close()
            self._logger.info(f"{self.__class__.__name__} closed successfully")
        except Exception as e:
            self._logger.error(f"Error closing {self.__class__.__name__}: {e}")
            raise

    def __enter__(self):
        """Context manager entry.

        Returns:
            Self for use in with statements
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Context manager exit - ensures cleanup.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()

    def __repr__(self) -> str:
        """String representation.

        Returns:
            String describing the client and backend
        """
        return f"{self.__class__.__name__}(backend={self._backend.__class__.__name__})"
