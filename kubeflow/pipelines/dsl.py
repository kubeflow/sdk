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

"""Kubeflow Pipelines DSL — re-exports from ``kfp.dsl`` with extras.

Usage::

    from kubeflow.pipelines import dsl


    @dsl.pipeline(name="my-pipeline")
    def my_pipeline():
        ...
        trigger = dsl.trigger_pipeline(pipeline_name="child")
"""

# Re-export everything from kfp.dsl
# Runtime dependencies (available at runtime)
# Compile-time dependencies
import os as _os

from kfp.dsl import (
    HTML,
    PIPELINE_JOB_CREATE_TIME_UTC_PLACEHOLDER,
    PIPELINE_JOB_ID_PLACEHOLDER,
    PIPELINE_JOB_NAME_PLACEHOLDER,
    PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
    PIPELINE_JOB_SCHEDULE_TIME_UTC_PLACEHOLDER,
    PIPELINE_ROOT_PLACEHOLDER,
    PIPELINE_TASK_EXECUTOR_INPUT_PLACEHOLDER,
    PIPELINE_TASK_EXECUTOR_OUTPUT_PATH_PLACEHOLDER,
    PIPELINE_TASK_ID_PLACEHOLDER,
    PIPELINE_TASK_NAME_PLACEHOLDER,
    WORKSPACE_PATH_PLACEHOLDER,
    Artifact,
    ClassificationMetrics,
    Dataset,
    Input,
    InputPath,
    Markdown,
    Metrics,
    Model,
    Output,
    OutputPath,
    PipelineTaskFinalStatus,
    SlicedClassificationMetrics,
    TaskConfig,
    get_uri,
    run_notebook,
)

if _os.environ.get("_KFP_RUNTIME", "false") != "true":
    from kfp.dsl import (
        Collected,
        ConcatPlaceholder,
        Condition,
        ContainerSpec,
        Elif,
        Else,
        ExitHandler,
        If,
        IfPresentPlaceholder,
        KubernetesWorkspaceConfig,
        OneOf,
        ParallelFor,
        PipelineConfig,
        PipelineTask,
        TaskConfigField,
        TaskConfigPassthrough,
        WorkspaceConfig,
        component,
        container_component,
        importer,
        notebook_component,
        pipeline,
    )

# Custom additions
from kubeflow.pipelines.trigger import trigger_pipeline

__all__ = [
    # Runtime
    "Artifact",
    "ClassificationMetrics",
    "Dataset",
    "get_uri",
    "HTML",
    "Input",
    "InputPath",
    "Markdown",
    "Metrics",
    "Model",
    "Output",
    "OutputPath",
    "PIPELINE_JOB_CREATE_TIME_UTC_PLACEHOLDER",
    "PIPELINE_JOB_ID_PLACEHOLDER",
    "PIPELINE_JOB_NAME_PLACEHOLDER",
    "PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER",
    "PIPELINE_JOB_SCHEDULE_TIME_UTC_PLACEHOLDER",
    "PIPELINE_ROOT_PLACEHOLDER",
    "PIPELINE_TASK_EXECUTOR_INPUT_PLACEHOLDER",
    "PIPELINE_TASK_EXECUTOR_OUTPUT_PATH_PLACEHOLDER",
    "PIPELINE_TASK_ID_PLACEHOLDER",
    "PIPELINE_TASK_NAME_PLACEHOLDER",
    "PipelineTaskFinalStatus",
    "run_notebook",
    "SlicedClassificationMetrics",
    "TaskConfig",
    "WORKSPACE_PATH_PLACEHOLDER",
    # Compile-time
    "Collected",
    "component",
    "ConcatPlaceholder",
    "Condition",
    "container_component",
    "ContainerSpec",
    "Elif",
    "Else",
    "ExitHandler",
    "If",
    "IfPresentPlaceholder",
    "importer",
    "KubernetesWorkspaceConfig",
    "notebook_component",
    "OneOf",
    "ParallelFor",
    "pipeline",
    "PipelineConfig",
    "PipelineTask",
    "TaskConfigField",
    "TaskConfigPassthrough",
    "WorkspaceConfig",
    # Custom additions
    "trigger_pipeline",
]
