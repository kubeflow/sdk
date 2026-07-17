# Copyright The Kubeflow Authors.
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

"""Tests for kubeflow.pipelines re-exports and ImportError handling."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

_real_import = builtins.__import__


def _block_kfp_import(name, *args, **kwargs):
    """Mock import that blocks any kfp imports."""
    if name == "kfp" or name.startswith("kfp."):
        raise ImportError(f"No module named '{name}'")
    return _real_import(name, *args, **kwargs)


def _reload_pipelines_modules():
    """Remove cached kubeflow.pipelines modules so lazy __getattr__ re-fires."""
    for mod_name in list(sys.modules):
        if mod_name.startswith("kubeflow.pipelines"):
            del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def skip_if_no_kfp():
    """Skip tests if kfp is not installed."""
    pytest.importorskip("kfp")


class TestLazyImport:
    """Test that importing kubeflow.pipelines succeeds without kfp."""

    def test_module_imports_without_kfp(self, monkeypatch):
        _reload_pipelines_modules()
        for mod_name in list(sys.modules):
            if mod_name == "kfp" or mod_name.startswith("kfp."):
                del sys.modules[mod_name]
        monkeypatch.setattr("builtins.__import__", _block_kfp_import)

        mod = importlib.import_module("kubeflow.pipelines")
        assert mod is not None


class TestPipelinesClientReExport:
    """Test that PipelinesClient is properly re-exported."""

    def test_import_from_package(self):
        from kubeflow.pipelines import PipelinesClient

        assert PipelinesClient is not None

    def test_import_from_api_module(self):
        from kubeflow.pipelines.api.pipelines_client import PipelinesClient

        assert PipelinesClient is not None


class TestDslReExports:
    """Test that KFP DSL modules are re-exported at kubeflow.pipelines level."""

    @pytest.mark.parametrize("module_name", ["dsl", "compiler", "components", "kubernetes"])
    def test_dsl_reexport(self, module_name):
        import kfp

        import kubeflow.pipelines as kp

        reexported = getattr(kp, module_name)
        original = getattr(kfp, module_name)
        assert reexported is original


class TestTypeReExports:
    """Test that KFP types are re-exported at kubeflow.pipelines level."""

    @pytest.mark.parametrize(
        "type_name",
        [
            "Pipeline",
            "PipelineVersion",
            "Run",
            "Experiment",
            "KubernetesBackendConfig",
            "constants",
            "ListPipelinesResponse",
            "ListPipelineVersionsResponse",
            "ListRunsResponse",
            "ListExperimentsResponse",
        ],
    )
    def test_type_reexport(self, type_name):
        from kfp import kubeflow_client as kfp_kc

        import kubeflow.pipelines as kp

        reexported = getattr(kp, type_name)
        original = getattr(kfp_kc, type_name)
        assert reexported is original


class TestImportErrorHandling:
    """Test that helpful ImportError is raised when kfp is not installed."""

    @pytest.mark.parametrize("attr", ["PipelinesClient", "dsl", "Pipeline"])
    def test_import_error_without_kfp(self, attr, monkeypatch):
        _reload_pipelines_modules()
        for mod_name in list(sys.modules):
            if mod_name == "kfp" or mod_name.startswith("kfp."):
                del sys.modules[mod_name]
        monkeypatch.setattr("builtins.__import__", _block_kfp_import)

        mod = importlib.import_module("kubeflow.pipelines")

        with pytest.raises(ImportError, match=r"pip install 'kubeflow\[pipelines\]'"):
            getattr(mod, attr)


class TestAllExports:
    """Test that __all__ is properly defined."""

    _EXPECTED_EXPORTS = [
        "PipelinesClient",
        "compiler",
        "components",
        "dsl",
        "kubernetes",
        "Experiment",
        "Pipeline",
        "PipelineVersion",
        "Run",
        "ListExperimentsResponse",
        "ListPipelinesResponse",
        "ListPipelineVersionsResponse",
        "ListRunsResponse",
        "KubernetesBackendConfig",
        "constants",
    ]

    def test_all_exports(self):
        import kubeflow.pipelines as kp

        for name in self._EXPECTED_EXPORTS:
            assert name in kp.__all__, f"{name} missing from __all__"
            assert hasattr(kp, name), f"{name} in __all__ but not importable"


class TestUnknownAttribute:
    """Test that accessing unknown attributes raises AttributeError."""

    def test_unknown_attr(self):
        import kubeflow.pipelines as kp

        with pytest.raises(AttributeError, match="has no attribute"):
            kp.nonexistent_thing  # noqa: B018
