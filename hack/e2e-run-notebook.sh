#!/usr/bin/env bash

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

# This shell is used to run Jupyter Notebook with Papermill.

set -o errexit
set -o nounset
set -o pipefail
set -x

if [ -z "${NOTEBOOK_INPUT}" ]; then
    echo "NOTEBOOK_INPUT env variable must be set to run this script."
    exit 1
fi

if [ -z "${NOTEBOOK_OUTPUT}" ]; then
    echo "NOTEBOOK_OUTPUT env variable must be set to run this script."
    exit 1
fi

if [ -z "${PAPERMILL_TIMEOUT}" ]; then
    echo "PAPERMILL_TIMEOUT env variable must be set to run this script."
    exit 1
fi

# PAPERMILL_PARAMS should contain full papermill parameter flags.
# Example: "-p num_cpu 3 -p gpu 1"
PAPERMILL_PARAMS="${PAPERMILL_PARAMS:-}"

print_results() {
    # Only run kubectl commands if we're testing Kubernetes notebooks
    if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
        local namespace="${SPARK_TEST_NAMESPACE:-spark-test}"
        echo "=== Spark Pods in ${namespace} ==="
        kubectl get pods -n "${namespace}" || true
        echo "=== SparkConnect Resources in ${namespace} ==="
        kubectl get sparkconnect -n "${namespace}" 2>/dev/null || true
        echo "=== SparkApplication Resources in ${namespace} ==="
        kubectl get sparkapplications -n "${namespace}" 2>/dev/null || true
        echo "=== Spark Connect Server Logs in ${namespace} ==="
        kubectl logs -n "${namespace}" -l app.kubernetes.io/component=server --tail=100 || true
        echo "=== Spark Driver Logs in ${namespace} ==="
        kubectl logs -n "${namespace}" -l spark-role=driver --tail=100 || true
    else
        echo "Skipping kubectl commands (not a Kubernetes test)"
    fi
}

(papermill "${NOTEBOOK_INPUT}" "${NOTEBOOK_OUTPUT}" ${PAPERMILL_PARAMS} --execution-timeout "${PAPERMILL_TIMEOUT}" && print_results) ||
    (print_results && exit 1)
