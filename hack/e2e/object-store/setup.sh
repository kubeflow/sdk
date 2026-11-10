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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Input interface
E2E_OBJECT_STORE="${E2E_OBJECT_STORE:-seaweedfs}"
E2E_S3_NAMESPACE="${E2E_S3_NAMESPACE:-seaweedfs}"
E2E_S3_CLIENT_NAMESPACES="${E2E_S3_CLIENT_NAMESPACES:-default}"
E2E_S3_BUCKETS="${E2E_S3_BUCKETS:-test-bucket}"
E2E_S3_SEED="${E2E_S3_SEED:-}"
E2E_S3_REGION="${E2E_S3_REGION:-us-east-1}"
E2E_S3_ENV_FILE="${E2E_S3_ENV_FILE:-.e2e/object-store.env}"

# Generate random credentials per run if not explicitly provided
RAND_ID="$(head -c 16 /dev/urandom | xxd -p | head -c 8 || date +%s | tail -c 8)"
E2E_S3_ACCESS_KEY="${E2E_S3_ACCESS_KEY:-ak-${RAND_ID}}"
E2E_S3_SECRET_KEY="${E2E_S3_SECRET_KEY:-sk-${RAND_ID}}"

mkdir -p "$(dirname "${E2E_S3_ENV_FILE}")"
rm -f "${E2E_S3_ENV_FILE}"

# 1. Deploy the storage backend
BACKEND_SCRIPT="${SCRIPT_DIR}/backends/${E2E_OBJECT_STORE}.sh"
if [[ ! -f "${BACKEND_SCRIPT}" ]]; then
  echo "Error: Backend script ${BACKEND_SCRIPT} not found." >&2
  exit 1
fi

echo "Running backend script for ${E2E_OBJECT_STORE}..."
export E2E_S3_NAMESPACE E2E_S3_ACCESS_KEY E2E_S3_SECRET_KEY
S3_ENDPOINT="$(bash "${BACKEND_SCRIPT}")"

if [[ -z "${S3_ENDPOINT}" ]]; then
  echo "Error: Backend script did not output an S3 endpoint." >&2
  exit 1
fi

# 2. Distribute Secret and ConfigMap to each client namespace
for ns in ${E2E_S3_CLIENT_NAMESPACES}; do
  echo "Setting up neutral Secret and ConfigMap in client namespace '${ns}'..."
  kubectl create namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f -

  # Neutral Secret: e2e-s3-credentials
  kubectl create secret generic e2e-s3-credentials \
    --namespace="${ns}" \
    --from-literal=AWS_ACCESS_KEY_ID="${E2E_S3_ACCESS_KEY}" \
    --from-literal=AWS_SECRET_ACCESS_KEY="${E2E_S3_SECRET_KEY}" \
    --dry-run=client -o yaml | kubectl apply -f -

  # Neutral ConfigMap: e2e-s3-config
  kubectl create configmap e2e-s3-config \
    --namespace="${ns}" \
    --from-literal=AWS_ENDPOINT_URL="${S3_ENDPOINT}" \
    --from-literal=AWS_REGION="${E2E_S3_REGION}" \
    --dry-run=client -o yaml | kubectl apply -f -
done

# 3. Build ConfigMap for seed files and run seed Job
PRIMARY_CLIENT_NS="${E2E_S3_CLIENT_NAMESPACES%% *}"

kubectl delete configmap e2e-s3-seed-files -n "${PRIMARY_CLIENT_NS}" --ignore-not-found
if [ -n "${E2E_S3_SEED}" ]; then
  cm_cmd=(kubectl create configmap e2e-s3-seed-files -n "${PRIMARY_CLIENT_NS}")
  for item in ${E2E_S3_SEED}; do
    local_path="${item#*=}"
    if [[ "${local_path}" != /* ]]; then
      local_path="${REPO_ROOT}/${local_path}"
    fi
    cm_cmd+=(--from-file="$(basename "${local_path}")=${local_path}")
  done
  "${cm_cmd[@]}" --dry-run=client -o yaml | kubectl apply -f -
fi

echo "Running seed Job in namespace '${PRIMARY_CLIENT_NS}'..."
kubectl delete job e2e-s3-seed -n "${PRIMARY_CLIENT_NS}" --ignore-not-found

export E2E_S3_BUCKETS E2E_S3_SEED
envsubst '${E2E_S3_BUCKETS} ${E2E_S3_SEED}' < "${SCRIPT_DIR}/seed-job.yaml" | kubectl apply -n "${PRIMARY_CLIENT_NS}" -f -

echo "Waiting for seed Job to complete..."
kubectl wait --for=condition=complete job/e2e-s3-seed -n "${PRIMARY_CLIENT_NS}" --timeout=120s || (kubectl logs -n "${PRIMARY_CLIENT_NS}" -l job-name=e2e-s3-seed --tail=50 && kubectl describe job/e2e-s3-seed -n "${PRIMARY_CLIENT_NS}" && exit 1)

# 4. Write full outputs to env file and GITHUB_ENV if present
cat << ENV_EOF > "${E2E_S3_ENV_FILE}"
S3_ENDPOINT=${S3_ENDPOINT}
S3_REGION=${E2E_S3_REGION}
S3_SECRET_NAME=e2e-s3-credentials
S3_CONFIGMAP_NAME=e2e-s3-config
S3_BUCKETS=${E2E_S3_BUCKETS}
ENV_EOF

if [[ -n "${GITHUB_ENV:-}" ]]; then
  cat << ENV_EOF >> "${GITHUB_ENV}"
S3_ENDPOINT=${S3_ENDPOINT}
S3_REGION=${E2E_S3_REGION}
S3_SECRET_NAME=e2e-s3-credentials
S3_CONFIGMAP_NAME=e2e-s3-config
S3_BUCKETS=${E2E_S3_BUCKETS}
ENV_EOF
fi

echo "Object store setup complete. Environment written to ${E2E_S3_ENV_FILE}."
