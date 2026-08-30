# Copyright 2026 The Kubeflow Authors.
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

"""Unit tests for jar `main_class` wiring in the SparkApplication CR."""

from kubeflow.spark.backends.kubernetes.utils import get_spark_application_cr_from_file_job


def test_jar_job_sets_java_type_and_main_class() -> None:
    app = get_spark_application_cr_from_file_job(
        name="jar-job",
        namespace="default",
        main_file="s3a://bucket/app.jar",
        arguments=["--input", "s3a://bucket/data"],
        main_class="org.apache.spark.examples.SparkPi",
        num_executors=2,
    )

    assert app.spec.type == "Java"
    assert app.spec.main_application_file == "s3a://bucket/app.jar"
    assert app.spec.main_class == "org.apache.spark.examples.SparkPi"
    assert app.spec.arguments == ["--input", "s3a://bucket/data"]


def test_python_job_omits_main_class() -> None:
    app = get_spark_application_cr_from_file_job(
        name="python-job",
        namespace="default",
        main_file="s3a://bucket/job.py",
    )

    assert app.spec.type == "Python"
    assert app.spec.main_class is None
