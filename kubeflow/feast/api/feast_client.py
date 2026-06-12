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

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from feast.repo_config import RepoConfig

    from feast import FeatureStore


class FeastClient:
    """Client for Feast feature store operations.

    Feast is a feature store that enables offline retrieval of historical datasets
    and online serving of features/data for ML applications.

    This is a minimal wrapper that provides simplified initialization. For full Feast
    functionality, use the `feature_store` property to access the underlying FeatureStore.

    Requires the feast package to be installed. Install it with:

        pip install 'kubeflow[feast]'

    Example:
        ```python
        from kubeflow.feast import FeastClient

        # Initialize client
        client = FeastClient(repo_path="/path/to/feast/repo")

        # Access full Feast functionality
        client.feature_store.get_online_features(...)
        client.feature_store.materialize(...)
        ```
    """

    def __init__(
        self,
        repo_path: str | None = None,
        config: RepoConfig | Mapping[str, Any] | None = None,
    ):
        """Initialize the FeastClient.

        Args:
            repo_path: Path to a Feast feature repository containing `feature_store.yaml`.
                If not provided, Feast uses the current directory.
            config: Optional Feast configuration. Accepts either a `RepoConfig`
                instance or a dictionary of arguments used to construct one. If
                provided, takes precedence over `repo_path`.

        Raises:
            ImportError: If feast is not installed.
        """
        try:
            from feast import FeatureStore, RepoConfig
        except ImportError as e:
            raise ImportError(
                "feast is not installed. Install it with:\n\n"  # fmt: skip
                "  pip install 'kubeflow[feast]'\n"
            ) from e

        self._feature_store: FeatureStore
        if config is not None:
            repo_config = RepoConfig(**config) if isinstance(config, Mapping) else config
            self._feature_store = FeatureStore(config=repo_config)
        else:
            self._feature_store = FeatureStore(repo_path=repo_path)

    @property
    def feature_store(self) -> FeatureStore:
        """Access the underlying Feast FeatureStore instance.

        Use this property to access the full Feast API for operations like:
        - get_online_features() / get_historical_features()
        - materialize() / materialize_incremental()
        - apply() - Deploy feature definitions
        - list_feature_views() / list_entities() / list_data_sources()

        Returns:
            The Feast FeatureStore instance.

        Example:
            ```python
            client = FeastClient(repo_path="/path/to/feast/repo")

            # Get online features
            features = client.feature_store.get_online_features(
                features=["feature_view:feature1"],
                entity_rows=[{"entity_id": 1}],
            )

            # List feature views
            for fv in client.feature_store.list_feature_views():
                print(fv.name)
            ```
        """
        return self._feature_store
