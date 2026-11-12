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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_FILE="${SCRIPT_DIR}/seaweedfs.yaml"

NAMESPACE="${E2E_S3_NAMESPACE:-seaweedfs}"
SERVICE_NAME="seaweedfs"
PORT="8333"

AK="${E2E_S3_ACCESS_KEY:-anyadmin}"
SK="${E2E_S3_SECRET_KEY:-anyadmin}"

echo "Ensuring namespace '${NAMESPACE}' exists..." >&2
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >&2

echo "Deploying SeaweedFS in namespace '${NAMESPACE}'..." >&2
kubectl apply -n "${NAMESPACE}" -f "${MANIFEST_FILE}" >&2

echo "Waiting for SeaweedFS deployment to become ready..." >&2
kubectl rollout status deployment/seaweedfs -n "${NAMESPACE}" --timeout=300s >&2 || (kubectl describe pod -l app=seaweedfs -n "${NAMESPACE}" >&2; exit 1)

echo "Configuring SeaweedFS user via admin shell..." >&2
kubectl -n "${NAMESPACE}" exec deploy/seaweedfs -- sh -c "
  for i in \$(seq 1 30); do
    if echo \"s3.configure -user e2e -access_key ${AK} -secret_key ${SK} -actions Read,Write,List,Admin -apply\" | weed shell -master=127.0.0.1:9333; then
      exit 0
    fi
    sleep 2
  done
  exit 1
" >&2

ENDPOINT="http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}"
echo "SeaweedFS is ready and configured at ${ENDPOINT}" >&2
echo "${ENDPOINT}"
