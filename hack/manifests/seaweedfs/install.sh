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
    kubectl get secret mlpipeline-minio-artifact -n default -o json \
      | jq --arg ns "${TARGET_NAMESPACE}" '.metadata.namespace = $ns | del(.metadata.resourceVersion, .metadata.uid, .metadata.creationTimestamp)' \
      | kubectl apply -f - 2>/dev/null || \
    kubectl get secret mlpipeline-minio-artifact -n default -o yaml \
      | sed "s/namespace: default/namespace: ${TARGET_NAMESPACE}/" \
      | kubectl apply -f -
fi

echo "[INFO] Waiting for SeaweedFS deployment to roll out..."
kubectl rollout status deployment/seaweedfs -n seaweedfs --timeout=90s

# Setup background port-forward to access SeaweedFS S3 from the host runner
echo "[INFO] Forwarding SeaweedFS S3 port 8333 to localhost:8333..."
kubectl port-forward -n seaweedfs svc/seaweedfs-s3 8333:8333 > /dev/null 2>&1 &
PF_PID=$!
trap 'kill ${PF_PID} 2>/dev/null || true' EXIT

# Wait until S3 port 8333 responds on localhost
echo "[INFO] Waiting for SeaweedFS S3 port 8333 to start accepting connections..."
READY=0
for _ in {1..30}; do
    if curl -s -m 2 http://localhost:8333/ > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
    echo "[ERROR] SeaweedFS S3 service failed to become ready on port 8333."
    exit 1
fi

echo "[INFO] Initializing default buckets ('mlpipeline', 'test-bucket')..."
curl -s -X PUT http://localhost:8333/mlpipeline > /dev/null
curl -s -X PUT http://localhost:8333/test-bucket > /dev/null

echo "[INFO] Uploading test spark_job.py script into SeaweedFS buckets..."
cat << 'PYEOF' > /tmp/spark_job.py
import sys
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("SeaweedFSS3ATest").getOrCreate()
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    count = spark.sparkContext.parallelize(range(1, 100), partitions).count()
    print(f"Successfully processed count: {count}")
    spark.stop()
PYEOF

curl -s -T /tmp/spark_job.py http://localhost:8333/mlpipeline/spark_job.py
curl -s -T /tmp/spark_job.py http://localhost:8333/test-bucket/spark_job.py
rm -f /tmp/spark_job.py

echo "[INFO] SeaweedFS setup completed successfully."
