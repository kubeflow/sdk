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

TARGET_NAMESPACE="${1:-default}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_FILE="${SCRIPT_DIR}/seaweedfs.yaml"

echo "[INFO] Deploying SeaweedFS S3 storage and KFP artifact secrets..."
kubectl apply -f "${MANIFEST_FILE}"

# Replicate KFP artifact secret to target namespace if different from default
if [[ "${TARGET_NAMESPACE}" != "default" ]]; then
    kubectl create secret generic mlpipeline-minio-artifact \
      --namespace="${TARGET_NAMESPACE}" \
      --from-literal=accesskey="minio" \
      --from-literal=secretkey="minio123" \
      --dry-run=client -o yaml | kubectl apply -f -
fi

echo "[INFO] Waiting for SeaweedFS deployment to roll out..."
kubectl rollout status deployment/seaweedfs -n seaweedfs --timeout=120s
kubectl wait --for=condition=ready pod -l app=seaweedfs -n seaweedfs --timeout=60s

# Obtain pod name
POD_NAME="$(kubectl get pods -n seaweedfs -l app=seaweedfs -o jsonpath='{.items[0].metadata.name}')"

echo "[INFO] Initializing default bucket 'mlpipeline'..."
kubectl exec -n seaweedfs "${POD_NAME}" -- /bin/sh -c '
printf "PUT /mlpipeline HTTP/1.1\r\nHost: 127.0.0.1:8333\r\nConnection: close\r\n\r\n" | nc 127.0.0.1 8333'

echo "[INFO] Uploading test spark_job.py script into SeaweedFS..."
kubectl exec -i -n seaweedfs "${POD_NAME}" -- /bin/sh -c 'cat << "PYEOF" > /tmp/spark_job.py
import sys
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("SeaweedFSS3ATest").getOrCreate()
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    count = spark.sparkContext.parallelize(range(1, 100), partitions).count()
    print(f"Successfully processed count: {count}")
    spark.stop()
PYEOF
SIZE=$(wc -c < /tmp/spark_job.py | tr -d " ")
{
  printf "PUT /mlpipeline/spark_job.py HTTP/1.1\r\n"
  printf "Host: 127.0.0.1:8333\r\n"
  printf "Content-Length: %s\r\n" "$SIZE"
  printf "Connection: close\r\n\r\n"
  cat /tmp/spark_job.py
} | nc 127.0.0.1 8333
'

# Verification step to ensure bucket exists
echo "[INFO] Verifying bucket exists..."
kubectl exec -n seaweedfs "${POD_NAME}" -- /bin/sh -c '
printf "GET /mlpipeline HTTP/1.1\r\nHost: 127.0.0.1:8333\r\nConnection: close\r\n\r\n" | nc 127.0.0.1 8333 | grep -E "HTTP/1.1 (200|403)"
'

echo "[INFO] SeaweedFS setup completed successfully."
