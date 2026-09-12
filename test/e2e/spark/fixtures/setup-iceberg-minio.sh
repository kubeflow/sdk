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
          image: quay.io/minio/minio:RELEASE.2024-06-13T22-53-53Z
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
          image: tabulario/iceberg-rest
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
        - name: mc
          image: quay.io/minio/mc:RELEASE.2024-06-13T22-53-53Z
          command: [/bin/sh, -c]
          args:
            - mc alias set local http://minio:9000 "\$MINIO_ACCESS_KEY" "\$MINIO_SECRET_KEY" && mc mb local/warehouse --ignore-existing
          envFrom: [{secretRef: {name: minio-credentials}}]
EOF

kubectl wait --for=condition=complete job/minio-init -n "$NAMESPACE" --timeout=180s
kubectl rollout status deployment/iceberg-rest -n "$NAMESPACE" --timeout=180s
