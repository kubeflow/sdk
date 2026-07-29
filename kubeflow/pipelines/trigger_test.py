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

"""Tests for the trigger_pipeline component."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def skip_if_no_kfp():
    pytest.importorskip("kfp")


class TestTriggerPipelineComponent:
    """Tests for the trigger_pipeline component definition and spec."""

    def test_component_input_spec(self):
        from kubeflow.pipelines.trigger import trigger_pipeline

        inputs = trigger_pipeline.component_spec.inputs
        assert "pipeline_name" in inputs
        assert inputs["pipeline_name"].type == "String"
        assert not inputs["pipeline_name"].optional

        assert "parameters" in inputs
        assert inputs["parameters"].optional

        assert "wait_for_completion" in inputs
        assert inputs["wait_for_completion"].type == "Boolean"
        assert inputs["wait_for_completion"].default is False

        assert "poke_interval" in inputs
        assert inputs["poke_interval"].type == "Integer"
        assert inputs["poke_interval"].default == 30

        assert "experiment_name" in inputs
        assert inputs["experiment_name"].type == "String"
        assert inputs["experiment_name"].optional

        assert "run_name" in inputs
        assert inputs["run_name"].type == "String"
        assert inputs["run_name"].optional

    def test_component_output_spec(self):
        from kubeflow.pipelines.trigger import trigger_pipeline

        outputs = trigger_pipeline.component_spec.outputs
        assert "Output" in outputs
        assert outputs["Output"].type == "String"

    def test_component_name(self):
        from kubeflow.pipelines.trigger import trigger_pipeline

        assert trigger_pipeline.name == "trigger-pipeline"

    def test_component_packages_to_install(self):
        from kubeflow.pipelines.trigger import trigger_pipeline

        impl = trigger_pipeline.component_spec.implementation
        container = impl.container
        assert container is not None
        # packages_to_install becomes pip install commands in the container
        assert any("kfp" in cmd for cmd in container.command)

    def test_compiles_in_pipeline(self):
        from kfp import compiler, dsl

        from kubeflow.pipelines.trigger import trigger_pipeline

        @dsl.pipeline(name="test-parent")
        def parent_pipeline():
            trigger_pipeline(
                pipeline_name="child-pipeline",
                parameters={"key": "value"},
                wait_for_completion=False,
            )

        import os
        import tempfile

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "pipeline.yaml")
        try:
            compiler.Compiler().compile(pipeline_func=parent_pipeline, package_path=path)
            import yaml

            with open(path) as f:
                spec = yaml.safe_load(f)
            ps = spec.get("pipelineSpec", spec)
            components = ps.get("components", {})
            trigger_components = [name for name in components if "trigger" in name.lower()]
            assert len(trigger_components) > 0
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


class TestTriggerPipelineExports:
    """Tests for trigger_pipeline exports."""

    def test_import_from_package(self):
        from kubeflow.pipelines import trigger_pipeline
        from kubeflow.pipelines.trigger import trigger_pipeline as tp

        assert trigger_pipeline is tp

    def test_import_from_dsl(self):
        from kubeflow.pipelines import dsl
        from kubeflow.pipelines.trigger import trigger_pipeline

        assert dsl.trigger_pipeline is trigger_pipeline

    def test_dsl_module_path(self):
        from kubeflow.pipelines import dsl

        assert dsl.__name__ == "kubeflow.pipelines.dsl"
        assert dsl.__file__.endswith("kubeflow/pipelines/dsl.py")
