#!/bin/bash
# Copyright 2025 The Kubeflow Authors.
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

CLUSTER_NAME="${SPARK_TEST_CLUSTER:-spark-test}"
NAMESPACE="${SPARK_TEST_NAMESPACE:-spark-test}"
SPARK_OPERATOR_VERSION="${SPARK_OPERATOR_VERSION:-2.1.0}"
SPARK_OPERATOR_IMAGE_TAG="${SPARK_OPERATOR_IMAGE_TAG:-latest}"
K8S_VERSION="${K8S_VERSION:-1.32.0}"
KIND_BIN="${KIND:-kind}"

# Construct Kind node image from K8S version
KIND_NODE_IMAGE="kindest/node:v${K8S_VERSION}"

# Future GPU Support:
# For GPU/nvkind clusters, set CLUSTER_TYPE=nvkind environment variable
# Example: CLUSTER_TYPE=nvkind make test-e2e-setup-cluster
# This will require nvkind binary and appropriate node images
# CLUSTER_TYPE="${CLUSTER_TYPE:-kind}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

_phase_start=0
phase_start() {
    _phase_start=$(date +%s)
    log_info "Phase start: $1"
}
phase_end() {
    local name="$1"
    local end now elapsed
    now=$(date +%s)
    end=$((now - _phase_start))
    if [[ $end -ge 60 ]]; then
        elapsed="${end}s ($((end / 60))m $((end % 60))s)"
    else
        elapsed="${end}s"
    fi
    log_info "Phase done: $name (elapsed: $elapsed)"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    local missing=()

    if ! command -v "$KIND_BIN" &>/dev/null; then
        missing+=("kind")
    fi
    command -v kubectl &>/dev/null || missing+=("kubectl")
    command -v helm &>/dev/null || missing+=("helm")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        echo "Install with:"
        echo "  brew install kind kubectl helm  # macOS"
        echo "  Or see: https://kind.sigs.k8s.io/docs/user/quick-start/"
        exit 1
    fi
    log_info "All prerequisites met (using Kind: $KIND_BIN)"
}

delete_cluster() {
    log_info "Deleting Kind cluster: $CLUSTER_NAME"
    "$KIND_BIN" delete cluster --name "$CLUSTER_NAME" 2>/dev/null || true
    log_info "Cluster deleted"
}

create_cluster() {
    phase_start "create_cluster"
    if "$KIND_BIN" get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        log_warn "Cluster '$CLUSTER_NAME' already exists"
        phase_end "create_cluster (skipped)"
        return 0
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local kind_config="${script_dir}/kind-config.yaml"
    log_info "Creating Kind cluster: $CLUSTER_NAME (Kubernetes $K8S_VERSION, 1 control-plane + 2 workers)"
    "$KIND_BIN" create cluster \
        --name "$CLUSTER_NAME" \
        --image "$KIND_NODE_IMAGE" \
        --config "$kind_config" \
        --wait 60s
    log_info "Cluster created with Kubernetes $K8S_VERSION"
    phase_end "create_cluster"
}

install_spark_operator() {
    local install_start
    install_start=$(date +%s)
    phase_start "install_spark_operator"
    log_info "Installing Spark Operator v${SPARK_OPERATOR_VERSION}..."

    kubectl config use-context "kind-${CLUSTER_NAME}" 2>/dev/null || true

    if [[ "${SPARK_OPERATOR_PRELOAD_IMAGE:-0}" == "1" ]]; then
        local node_arch
        node_arch="$(kubectl get nodes -o jsonpath='{.items[0].status.nodeInfo.architecture}' 2>/dev/null)" || node_arch=""
        if [[ -n "$node_arch" ]]; then
            log_info "Preloading controller image for $node_arch..."
            if docker pull --platform "linux/${node_arch}" ghcr.io/kubeflow/spark-operator/controller:latest 2>/dev/null; then
                "$KIND_BIN" load docker-image ghcr.io/kubeflow/spark-operator/controller:latest --name "$CLUSTER_NAME" 2>/dev/null || log_warn "kind load failed, helm will pull from node"
            fi
        fi
    fi

    phase_start "helm_repo_update"
    helm repo add spark-operator https://kubeflow.github.io/spark-operator 2>/dev/null || true
    helm repo update
    phase_end "helm_repo_update"

    phase_start "helm_install_or_upgrade"
    local helm_extra_args=(
        --set webhook.enable=true
        --set image.registry=ghcr.io
        --set image.repository=kubeflow/spark-operator/controller
        --set "image.tag=$SPARK_OPERATOR_IMAGE_TAG"
        --set "spark.jobNamespaces[0]=$NAMESPACE"
    )
    local helm_timeout="${HELM_TIMEOUT:-15m}"
    log_info "Helm may take up to $helm_timeout. 'context canceled' usually means the process was killed (external timeout or Ctrl+C)."

    local helm_ret=0
    if helm list -n spark-operator 2>/dev/null | grep -q spark-operator; then
        log_info "Spark Operator already installed, upgrading..."
        helm upgrade spark-operator spark-operator/spark-operator \
            --namespace spark-operator \
            --version "$SPARK_OPERATOR_VERSION" \
            "${helm_extra_args[@]}" \
            --timeout "$helm_timeout" \
            --wait || helm_ret=$?
    else
        helm install spark-operator spark-operator/spark-operator \
            --namespace spark-operator \
            --create-namespace \
            --version "$SPARK_OPERATOR_VERSION" \
            "${helm_extra_args[@]}" \
            --timeout "$helm_timeout" \
            --wait || helm_ret=$?
    fi
    phase_end "helm_install_or_upgrade"

    _dump_controller_logs_if_not_ready() {
        local ready
        ready=$(kubectl get deployment -n spark-operator spark-operator-controller -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [[ "${ready:-0}" != "1" ]]; then
            log_error "Controller not ready (readyReplicas=${ready:-0}). Pods and controller logs:"
            kubectl get pods -n spark-operator 2>/dev/null || true
            local controller_pod
            controller_pod=$(kubectl get pods -n spark-operator -o name 2>/dev/null | grep -E 'controller' | head -1 | sed 's|pod/||')
            if [[ -n "$controller_pod" ]]; then
                kubectl logs -n spark-operator "$controller_pod" --tail=80 2>/dev/null || true
            fi
            return 1
        fi
        return 0
    }

    if [[ $helm_ret -ne 0 ]]; then
        log_error "Helm failed (exit $helm_ret). If you saw 'context canceled', the process was likely killed by an external timeout or Ctrl+C."
        _dump_controller_logs_if_not_ready || true
        exit 1
    fi

    log_info "Verifying controller is ready..."
    sleep 3
    if ! _dump_controller_logs_if_not_ready; then
        exit 1
    fi
    log_info "Spark Operator installed (CRDs from Helm chart)"
    local install_end
    install_end=$(($(date +%s) - install_start))
    if [[ $install_end -ge 60 ]]; then
        log_info "Phase done: install_spark_operator (total: ${install_end}s ($((install_end / 60))m $((install_end % 60))s))"
    else
        log_info "Phase done: install_spark_operator (total: ${install_end}s)"
    fi
}

setup_test_namespace() {
    phase_start "setup_test_namespace"
    log_info "Setting up test namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" 2>/dev/null || true
    log_info "Test namespace created"
    phase_end "setup_test_namespace"
}

# 1) Operator: grant Spark Operator controller permission to manage SparkConnect and pods in the test namespace.
# 2) Driver: grant default SA in test namespace permission so the Spark Connect server (driver) can create/watch executor pods.
# 3) ClusterRole for endpointslices: Helm chart may not grant discovery.k8s.io/endpointslices; bind to controller so Service/EndpointSlice updates succeed.
ensure_sparkconnect_rbac() {
    phase_start "ensure_sparkconnect_rbac"
    log_info "Creating Role, RoleBinding, and ClusterRole for SparkConnect (namespace $NAMESPACE)"

    kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: spark-operator-e2e-endpointslices
rules:
  - apiGroups: ["discovery.k8s.io"]
    resources: ["endpointslices"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["endpoints"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: spark-operator-e2e-endpointslices
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: spark-operator-e2e-endpointslices
subjects:
  - kind: ServiceAccount
    name: spark-operator-controller
    namespace: spark-operator
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: spark-operator-sparkconnect
  namespace: $NAMESPACE
rules:
  - apiGroups: ["sparkoperator.k8s.io"]
    resources: ["sparkconnects", "sparkconnects/status"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "endpoints"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["discovery.k8s.io"]
    resources: ["endpointslices"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: spark-operator-sparkconnect
  namespace: $NAMESPACE
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: spark-operator-sparkconnect
subjects:
  - kind: ServiceAccount
    name: spark-operator-controller
    namespace: spark-operator
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: spark-driver
  namespace: $NAMESPACE
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps"]
    verbs: ["get", "list", "watch", "create", "delete", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: spark-driver
  namespace: $NAMESPACE
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: spark-driver
subjects:
  - kind: ServiceAccount
    name: default
    namespace: $NAMESPACE
---
# E2E in-cluster runner: default SA can create/get SparkConnect so Job pods use in-cluster URL (no port-forward).
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: e2e-sparkconnect-client
  namespace: $NAMESPACE
rules:
  - apiGroups: ["sparkoperator.k8s.io"]
    resources: ["sparkconnects", "sparkconnects/status"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: e2e-sparkconnect-client
  namespace: $NAMESPACE
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: e2e-sparkconnect-client
subjects:
  - kind: ServiceAccount
    name: default
    namespace: $NAMESPACE
EOF
    phase_end "ensure_sparkconnect_rbac"
}

# Apply SparkConnect CRD from vendored hack/crds (Helm chart may not include it).
apply_sparkconnect_crd() {
    phase_start "apply_sparkconnect_crd"
    local script_dir repo_root crd_file
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"
    crd_file="$repo_root/hack/crds/sparkoperator.k8s.io_sparkconnects.yaml"
    if [[ -f "$crd_file" ]]; then
        log_info "Applying SparkConnect CRD (controller requires it)"
        kubectl apply -f "$crd_file"
    else
        log_warn "SparkConnect CRD not found at $crd_file; controller may CrashLoopBackOff"
    fi
    phase_end "apply_sparkconnect_crd"
}

apply_crd_only() {
    phase_start "apply_crd_only"
    kubectl config use-context "kind-${CLUSTER_NAME}" 2>/dev/null || true
    local script_dir repo_root crds_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"
    crds_dir="$repo_root/hack/crds"
    if [[ ! -d "$crds_dir" ]]; then
        log_error "CRDs dir not found: $crds_dir"
        exit 1
    fi
    for f in "$crds_dir"/*.yaml; do
        [[ -f "$f" ]] || continue
        log_info "Applying CRD: $(basename "$f")"
        kubectl apply -f "$f"
    done
    phase_end "apply_crd_only"
}

print_status() {
    echo ""
    log_info "=== Cluster Status ==="
    echo "Cluster: $CLUSTER_NAME"
    echo "Kubernetes version: $K8S_VERSION"
    echo "Test namespace: $NAMESPACE"
    if [[ "${E2E_CRD_ONLY:-0}" == "1" ]]; then
        echo "Mode: CRD-only (no Spark Operator controller)"
        echo "CRDs:"
        kubectl get crd | grep sparkoperator || true
    else
        echo "Spark Operator version: $SPARK_OPERATOR_VERSION"
        echo "Image tag: $SPARK_OPERATOR_IMAGE_TAG"
        echo ""
        echo "Spark Operator Deployment:"
        kubectl get deployment -n spark-operator 2>/dev/null || true
    fi
    echo ""
    echo "Test Namespace Pods:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "No pods yet"
    echo ""
    log_info "=== Usage ==="
    if [[ "${E2E_CRD_ONLY:-0}" == "1" ]]; then
        echo "Smoke test: uv run pytest test/e2e/spark/test_spark_examples.py -v -k smoke"
    else
        echo "To run E2E tests:"
        echo "  python -m pytest test/e2e/spark/test_spark_examples.py -v"
    fi
    echo ""
    echo "To delete cluster:"
    echo "  make test-e2e-setup-cluster K8S_VERSION=$K8S_VERSION --delete"
    echo "  Or: ./hack/e2e-setup-cluster.sh --delete"
}

main() {
    if [[ "${1:-}" == "--delete" ]]; then
        delete_cluster
        exit 0
    fi

    local main_start
    main_start=$(date +%s)
    check_prerequisites
    create_cluster
    setup_test_namespace
    if [[ "${E2E_CRD_ONLY:-0}" == "1" ]]; then
        apply_crd_only
    else
        ensure_sparkconnect_rbac
        apply_sparkconnect_crd
        if [[ "$SPARK_OPERATOR_IMAGE_TAG" == "local" ]]; then
            phase_start "kind_load_local_image"
            log_info "Loading locally built controller image into Kind..."
            "$KIND_BIN" load docker-image "ghcr.io/kubeflow/spark-operator/controller:local" --name "$CLUSTER_NAME"
            phase_end "kind_load_local_image"
        fi
        install_spark_operator
    fi
    print_status
    local total_elapsed
    total_elapsed=$(($(date +%s) - main_start))
    if [[ $total_elapsed -ge 60 ]]; then
        log_info "Total setup time: ${total_elapsed}s ($((total_elapsed / 60))m $((total_elapsed % 60))s)"
    else
        log_info "Total setup time: ${total_elapsed}s"
    fi
}

main "$@"
