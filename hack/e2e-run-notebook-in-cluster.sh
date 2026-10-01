#!/usr/bin/env bash

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

set -euo pipefail

NOTEBOOK_INPUT=${NOTEBOOK_INPUT:-}
NOTEBOOK_OUTPUT=${NOTEBOOK_OUTPUT:-}
PAPERMILL_TIMEOUT=${PAPERMILL_TIMEOUT:-900}
SPARK_TEST_NAMESPACE=${SPARK_TEST_NAMESPACE:-spark-test}
RUNNER_IMAGE=${RUNNER_IMAGE:-spark-e2e-runner:local}

if [ -z "$NOTEBOOK_INPUT" ]; then
  echo "Error: NOTEBOOK_INPUT is required."
  exit 1
fi

if [ -z "$NOTEBOOK_OUTPUT" ]; then
  echo "Error: NOTEBOOK_OUTPUT is required."
  exit 1
fi

mkdir -p "$(dirname "$NOTEBOOK_OUTPUT")"

NOTEBOOK_NAME=$(basename "$NOTEBOOK_INPUT" .ipynb)
CLEAN_NAME=$(echo "$NOTEBOOK_NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
JOB_NAME="spark-e2e-${CLEAN_NAME}"
# K8s job names must be <= 63 characters
JOB_NAME="${JOB_NAME:0:63}"
JOB_NAME="${JOB_NAME%-}"

echo "================================================================="
echo "Running notebook in-cluster: $NOTEBOOK_INPUT"
echo "Job Name: $JOB_NAME"
echo "Namespace: $SPARK_TEST_NAMESPACE"
echo "Timeout: ${PAPERMILL_TIMEOUT}s"
echo "================================================================="

# Clean up any existing job with the same name
kubectl delete job "$JOB_NAME" -n "$SPARK_TEST_NAMESPACE" --ignore-not-found=true

SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-spark}

# Create the Job manifest and apply it
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${SPARK_TEST_NAMESPACE}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: ${PAPERMILL_TIMEOUT}
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: ${SERVICE_ACCOUNT}
      containers:
        - name: runner
          image: ${RUNNER_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /app
          command:
            - papermill
            - "${NOTEBOOK_INPUT}"
            - "/output/${NOTEBOOK_NAME}.ipynb"
            - "--log-output"
            - "--execution-timeout"
            - "${PAPERMILL_TIMEOUT}"
          env:
            - name: SPARK_TEST_NAMESPACE
              value: "${SPARK_TEST_NAMESPACE}"
            - name: SPARK_E2E_RUN_IN_CLUSTER
              value: "1"
EOF

# Wait for pod creation to begin tracking
echo "Waiting for Job pod to start..."
for i in {1..30}; do
  POD_NAME=$(kubectl get pods -n "$SPARK_TEST_NAMESPACE" -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [ -n "$POD_NAME" ]; then
    break
  fi
  sleep 2
done

if [ -z "$POD_NAME" ]; then
  echo "Error: Timed out waiting for pod for Job $JOB_NAME"
  kubectl describe job "$JOB_NAME" -n "$SPARK_TEST_NAMESPACE"
  exit 1
fi

echo "Streaming logs from pod: $POD_NAME"
kubectl logs -f "$POD_NAME" -n "$SPARK_TEST_NAMESPACE" || true

# Wait for Job completion condition
echo "Waiting for Job to complete..."
JOB_STATUS=0
kubectl wait --for=condition=complete job "$JOB_NAME" -n "$SPARK_TEST_NAMESPACE" --timeout=30s || JOB_STATUS=$?

if [ "${JOB_STATUS:-0}" -ne 0 ]; then
  echo "Error: Job $JOB_NAME failed or timed out."
  echo "==================== POD LOGS (CONTAINER: runner) ===================="
  kubectl logs "$POD_NAME" -n "$SPARK_TEST_NAMESPACE" -c runner || true
  echo "======================================================================"
  kubectl describe job "$JOB_NAME" -n "$SPARK_TEST_NAMESPACE"
  kubectl describe pod "$POD_NAME" -n "$SPARK_TEST_NAMESPACE"
  exit 1
fi

echo "Job completed successfully. Copying output notebook to $NOTEBOOK_OUTPUT..."
kubectl cp "${SPARK_TEST_NAMESPACE}/${POD_NAME}:/output/${NOTEBOOK_NAME}.ipynb" "$NOTEBOOK_OUTPUT"

# Clean up Job
kubectl delete job "$JOB_NAME" -n "$SPARK_TEST_NAMESPACE" --ignore-not-found=true

echo "Notebook execution finished: $NOTEBOOK_OUTPUT"
