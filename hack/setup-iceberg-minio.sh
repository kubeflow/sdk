#!/bin/bash
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

NAMESPACE="${SPARK_TEST_NAMESPACE:-spark-test}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
MINIO_IMAGE="public.ecr.aws/y9w4h2u9/minio:RELEASE.2025-09-07T16-13-09Z"
ICEBERG_REST_IMAGE="tabulario/iceberg-rest:latest"
AWS_CLI_IMAGE="public.ecr.aws/aws-cli/aws-cli:2.37.4"

show_fixture_diagnostics() {
    kubectl get pods -n "$NAMESPACE" -o wide || true
    kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp || true
    kubectl logs -n "$NAMESPACE" job/minio-init --all-containers=true || true
}
trap show_fixture_diagnostics ERR

if [[ "${SPARK_E2E_LOAD_FIXTURE_IMAGES:-0}" == "1" ]]; then
    for image in "$MINIO_IMAGE" "$ICEBERG_REST_IMAGE" "$AWS_CLI_IMAGE"; do
        docker pull "$image"
        "${KIND:-kind}" load docker-image "$image" --name "${SPARK_TEST_CLUSTER:-spark-test}"
    done
fi

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic minio-credentials -n "$NAMESPACE" \
    --from-literal=AWS_ACCESS_KEY_ID="$MINIO_ROOT_USER" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$MINIO_ROOT_PASSWORD" \
    --from-literal=MINIO_ACCESS_KEY="$MINIO_ROOT_USER" \
    --from-literal=MINIO_SECRET_KEY="$MINIO_ROOT_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl delete job minio-init -n "$NAMESPACE" --ignore-not-found=true
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: $NAMESPACE
spec:
  selector: {app: minio}
  ports:
    - {name: api, port: 9000, targetPort: 9000}
    - {name: console, port: 9001, targetPort: 9001}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector: {matchLabels: {app: minio}}
  template:
    metadata: {labels: {app: minio}}
    spec:
      containers:
        - name: minio
          image: $MINIO_IMAGE
          imagePullPolicy: IfNotPresent
          args: [server, /data, --console-address, ":9001"]
          env:
            - {name: MINIO_ROOT_USER, valueFrom: {secretKeyRef: {name: minio-credentials, key: MINIO_ACCESS_KEY}}}
            - {name: MINIO_ROOT_PASSWORD, valueFrom: {secretKeyRef: {name: minio-credentials, key: MINIO_SECRET_KEY}}}
          ports: [{containerPort: 9000}, {containerPort: 9001}]
---
apiVersion: v1
kind: Service
metadata:
  name: iceberg-rest
  namespace: $NAMESPACE
spec:
  selector: {app: iceberg-rest}
  ports: [{name: http, port: 8181, targetPort: 8181}]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iceberg-rest
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector: {matchLabels: {app: iceberg-rest}}
  template:
    metadata: {labels: {app: iceberg-rest}}
    spec:
      containers:
        - name: iceberg-rest
          image: $ICEBERG_REST_IMAGE
          imagePullPolicy: IfNotPresent
          envFrom: [{secretRef: {name: minio-credentials}}]
          env:
            - {name: AWS_REGION, value: us-east-1}
            - {name: CATALOG_WAREHOUSE, value: "s3://warehouse/"}
            - {name: CATALOG_IO__IMPL, value: org.apache.iceberg.aws.s3.S3FileIO}
            - {name: CATALOG_S3_ENDPOINT, value: "http://minio:9000"}
            - {name: CATALOG_S3_PATH__STYLE__ACCESS, value: "true"}
          ports: [{containerPort: 8181}]
EOF

kubectl rollout status deployment/minio -n "$NAMESPACE" --timeout=180s

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: minio-init
  namespace: $NAMESPACE
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: aws-cli
          image: $AWS_CLI_IMAGE
          imagePullPolicy: IfNotPresent
          command: ["sh", "-c"]
          args:
            - |
              until aws --endpoint-url http://minio:9000 s3api list-buckets >/dev/null 2>&1; do
                sleep 2
              done
              if ! aws --endpoint-url http://minio:9000 s3api head-bucket --bucket warehouse >/dev/null 2>&1; then
                aws --endpoint-url http://minio:9000 s3api create-bucket --bucket warehouse
              fi
              echo 'Bucket warehouse is ready'
          env:
            - {name: AWS_ACCESS_KEY_ID, valueFrom: {secretKeyRef: {name: minio-credentials, key: AWS_ACCESS_KEY_ID}}}
            - {name: AWS_SECRET_ACCESS_KEY, valueFrom: {secretKeyRef: {name: minio-credentials, key: AWS_SECRET_ACCESS_KEY}}}
            - {name: AWS_DEFAULT_REGION, value: us-east-1}
            - {name: AWS_EC2_METADATA_DISABLED, value: "true"}
EOF

kubectl wait --for=condition=complete job/minio-init -n "$NAMESPACE" --timeout=180s
kubectl rollout status deployment/iceberg-rest -n "$NAMESPACE" --timeout=180s
