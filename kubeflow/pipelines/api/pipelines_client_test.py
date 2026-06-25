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

"""Tests for kubeflow.pipelines re-exports and ImportError handling."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase

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

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="import kubeflow.pipelines succeeds without triggering kfp",
                expected_status=SUCCESS,
            ),
        ],
    )
    def test_module_imports_without_kfp(self, test_case, monkeypatch):
        _reload_pipelines_modules()
        for mod_name in list(sys.modules):
            if mod_name == "kfp" or mod_name.startswith("kfp."):
                del sys.modules[mod_name]
        monkeypatch.setattr("builtins.__import__", _block_kfp_import)

        # Module-level import must NOT fail — lazy pattern
        mod = importlib.import_module("kubeflow.pipelines")
        assert mod is not None
        assert test_case.expected_status == SUCCESS


class TestPipelinesClientReExport:
    """Test that PipelinesClient is properly re-exported."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="PipelinesClient importable from kubeflow.pipelines",
                expected_status=SUCCESS,
            ),
        ],
    )
    def test_import_pipelines_client(self, test_case):
        from kubeflow.pipelines import PipelinesClient

        assert PipelinesClient is not None
        assert test_case.expected_status == SUCCESS

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="PipelinesClient importable from api module",
                expected_status=SUCCESS,
            ),
        ],
    )
    def test_import_from_api_module(self, test_case):
        from kubeflow.pipelines.api.pipelines_client import PipelinesClient

        assert PipelinesClient is not None
        assert test_case.expected_status == SUCCESS


class TestDslReExports:
    """Test that KFP DSL modules are re-exported at kubeflow.pipelines level."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="dsl re-exported from kubeflow.pipelines",
                config={"module": "dsl"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="compiler re-exported from kubeflow.pipelines",
                config={"module": "compiler"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="components re-exported from kubeflow.pipelines",
                config={"module": "components"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="kubernetes re-exported from kubeflow.pipelines",
                config={"module": "kubernetes"},
                expected_status=SUCCESS,
            ),
        ],
    )
    def test_dsl_reexport(self, test_case):
        import kfp

        import kubeflow.pipelines as kp

        module_name = test_case.config["module"]
        reexported = getattr(kp, module_name)
        original = getattr(kfp, module_name)
        assert reexported is original
        assert test_case.expected_status == SUCCESS


class TestTypeReExports:
    """Test that KFP types are re-exported at kubeflow.pipelines level."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="Pipeline type re-exported",
                config={"type_name": "Pipeline"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="PipelineVersion type re-exported",
                config={"type_name": "PipelineVersion"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="Run type re-exported",
                config={"type_name": "Run"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="Experiment type re-exported",
                config={"type_name": "Experiment"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="KubernetesBackendConfig re-exported",
                config={"type_name": "KubernetesBackendConfig"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="constants module re-exported",
                config={"type_name": "constants"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="ListPipelinesResponse re-exported",
                config={"type_name": "ListPipelinesResponse"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="ListPipelineVersionsResponse re-exported",
                config={"type_name": "ListPipelineVersionsResponse"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="ListRunsResponse re-exported",
                config={"type_name": "ListRunsResponse"},
                expected_status=SUCCESS,
            ),
            TestCase(
                name="ListExperimentsResponse re-exported",
                config={"type_name": "ListExperimentsResponse"},
                expected_status=SUCCESS,
            ),
        ],
    )
    def test_type_reexport(self, test_case):
        from kfp import kubeflow_client as kfp_kc

        import kubeflow.pipelines as kp

        type_name = test_case.config["type_name"]
        reexported = getattr(kp, type_name)
        original = getattr(kfp_kc, type_name)
        assert reexported is original
        assert test_case.expected_status == SUCCESS


class TestImportErrorHandling:
    """Test that helpful ImportError is raised when kfp is not installed."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="PipelinesClient access raises ImportError without kfp",
                config={"attr": "PipelinesClient"},
                expected_status=FAILED,
                expected_error=ImportError,
            ),
            TestCase(
                name="dsl access raises ImportError without kfp",
                config={"attr": "dsl"},
                expected_status=FAILED,
                expected_error=ImportError,
            ),
            TestCase(
                name="Pipeline type access raises ImportError without kfp",
                config={"attr": "Pipeline"},
                expected_status=FAILED,
                expected_error=ImportError,
            ),
        ],
    )
    def test_import_error_without_kfp(self, test_case, monkeypatch):
        _reload_pipelines_modules()
        # Also remove cached kfp modules so importlib.import_module can't find them
        for mod_name in list(sys.modules):
            if mod_name == "kfp" or mod_name.startswith("kfp."):
                del sys.modules[mod_name]
        monkeypatch.setattr("builtins.__import__", _block_kfp_import)

        mod = importlib.import_module("kubeflow.pipelines")

        try:
            getattr(mod, test_case.config["attr"])
            assert test_case.expected_status == SUCCESS
        except ImportError as e:
            assert test_case.expected_status == FAILED
            assert "pip install 'kubeflow[pipelines]'" in str(e)


class TestAllExports:
    """Test that __all__ is properly defined."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="__all__ contains all expected exports",
                expected_status=SUCCESS,
                expected_output=[
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
                ],
            ),
        ],
    )
    def test_all_exports(self, test_case):
        import kubeflow.pipelines as kp

        for name in test_case.expected_output:
            assert name in kp.__all__, f"{name} missing from __all__"
            assert hasattr(kp, name), f"{name} in __all__ but not importable"
        assert test_case.expected_status == SUCCESS


class TestUnknownAttribute:
    """Test that accessing unknown attributes raises AttributeError."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="unknown attribute raises AttributeError",
                config={"attr": "nonexistent_thing"},
                expected_status=FAILED,
                expected_error=AttributeError,
            ),
        ],
    )
    def test_unknown_attr(self, test_case):
        import kubeflow.pipelines as kp

        try:
            getattr(kp, test_case.config["attr"])
            assert test_case.expected_status == SUCCESS
        except AttributeError:
            assert test_case.expected_status == FAILED
