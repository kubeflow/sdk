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
K8S_VERSION="${K8S_VERSION:-1.32.11}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${REPO_ROOT}/bin"
SPARK_OPERATOR_DIR="${REPO_ROOT}/spark-operator"
SPARK_OPERATOR_REPO="https://github.com/kubeflow/spark-operator.git"
SPARK_OPERATOR_CHART="${SPARK_OPERATOR_DIR}/charts/spark-operator-chart"

# Prefer the tools installed into ./bin by `make test-e2e-setup-cluster`.
export PATH="${BIN_DIR}:${PATH}"

SPARK_OPERATOR_RELEASE="spark-operator"
SPARK_OPERATOR_NAMESPACE="spark-operator"

IMAGE_TAG="local"
CONTROLLER_IMAGE_REPOSITORY="kubeflow/spark-operator/controller"
HOOK_IMAGE_REPOSITORY="kubeflow/spark-operator/kubectl"
CONTROLLER_IMAGE="ghcr.io/${CONTROLLER_IMAGE_REPOSITORY}:${IMAGE_TAG}"
HOOK_IMAGE="ghcr.io/${HOOK_IMAGE_REPOSITORY}:${IMAGE_TAG}"

SPARK_SERVICE_ACCOUNT="${SPARK_OPERATOR_RELEASE}-spark"

KIND_NODE_IMAGE="kindest/node:v${K8S_VERSION}"

KIND_CLUSTER_NAME=""
CLUSTER_REUSED=false

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

delete_cluster() {
    log_info "Deleting Kind cluster: $CLUSTER_NAME"
    kind delete cluster --name "$CLUSTER_NAME" 2>/dev/null || true
    log_info "Cluster deleted"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    local missing=()
    local tool
    for tool in docker git kind kubectl helm; do
        command -v "$tool" &>/dev/null || missing+=("$tool")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        echo "Install kind, kubectl and helm into ./bin with: make kind kubectl helm"
        echo "(make test-e2e-setup-cluster does this automatically)"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not reachable; Kind and the image builds need it."
        exit 1
    fi
    log_info "Using $(kind version), kubectl $(kubectl version --client -o yaml | awk '/gitVersion/ {print $2}'), Helm $(helm version --short)"
}

ensure_spark_operator_checkout() {
    if [[ -d "${SPARK_OPERATOR_DIR}/.git" || -f "${SPARK_OPERATOR_DIR}/go.mod" ]]; then
        log_info "Using Spark Operator checkout: ${SPARK_OPERATOR_DIR} ($(git -C "$SPARK_OPERATOR_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown))"
        return 0
    fi
    phase_start "clone_spark_operator"
    log_info "Cloning ${SPARK_OPERATOR_REPO} into ${SPARK_OPERATOR_DIR}"
    git clone --depth 1 "$SPARK_OPERATOR_REPO" "$SPARK_OPERATOR_DIR"
    phase_end "clone_spark_operator"
}

build_spark_operator_images() {
    phase_start "build_spark_operator_images"
    log_info "Building ${CONTROLLER_IMAGE}"
    docker build -t "$CONTROLLER_IMAGE" "$SPARK_OPERATOR_DIR"
    log_info "Building ${HOOK_IMAGE}"
    docker build -f "${SPARK_OPERATOR_DIR}/docker/Dockerfile.kubectl" -t "$HOOK_IMAGE" "$SPARK_OPERATOR_DIR"
    phase_end "build_spark_operator_images"
}

current_kind_cluster() {
    local context name
    context="$(kubectl config current-context 2>/dev/null)" || return 1
    [[ "$context" == kind-* ]] || return 1
    name="${context#kind-}"
    kind get clusters 2>/dev/null | grep -qx "$name" || return 1
    kubectl cluster-info --request-timeout=10s &>/dev/null || return 1
    echo "$name"
}

ensure_cluster() {
    phase_start "ensure_cluster"
    local existing
    if existing="$(current_kind_cluster)"; then
        KIND_CLUSTER_NAME="$existing"
        CLUSTER_REUSED=true
        log_info "Reusing Kind cluster from current kubectl context: kind-${KIND_CLUSTER_NAME}"
        kubectl get nodes
        phase_end "ensure_cluster (reused)"
        return 0
    fi

    local context
    context="$(kubectl config current-context 2>/dev/null || true)"
    if [[ -n "$context" && "$context" != "kind-${CLUSTER_NAME}" ]]; then
        log_info "Current kubectl context '$context' is not a reachable Kind cluster; using a dedicated one"
    fi

    KIND_CLUSTER_NAME="$CLUSTER_NAME"
    if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
        log_warn "Kind cluster '$CLUSTER_NAME' already exists; reusing it"
    else
        log_info "Creating Kind cluster: $CLUSTER_NAME (Kubernetes $K8S_VERSION, 1 control-plane + 2 workers)"
        kind create cluster \
            --name "$CLUSTER_NAME" \
            --image "$KIND_NODE_IMAGE" \
            --config "${REPO_ROOT}/hack/kind-config.yaml" \
            --wait 60s
        log_info "Cluster created with Kubernetes $K8S_VERSION"
    fi
    kubectl config use-context "kind-${CLUSTER_NAME}"
    phase_end "ensure_cluster"
}

setup_test_namespace() {
    phase_start "setup_test_namespace"
    log_info "Setting up test namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    log_info "Test namespace ready"
    phase_end "setup_test_namespace"
}

load_spark_operator_images() {
    phase_start "kind_load_images"
    local image
    for image in "$CONTROLLER_IMAGE" "$HOOK_IMAGE"; do
        log_info "Loading image into Kind: $image"
        kind load docker-image "$image" --name "$KIND_CLUSTER_NAME"
    done
    phase_end "kind_load_images"
}

verify_chart_resources() {
    phase_start "verify_chart_resources"
    local failed=0
    local crd
    for crd in sparkapplications.sparkoperator.k8s.io sparkconnects.sparkoperator.k8s.io; do
        if kubectl get crd "$crd" &>/dev/null; then
            log_info "CRD present: $crd"
        else
            log_error "CRD missing: $crd (expected from the Helm chart)"
            failed=1
        fi
    done
    if kubectl get serviceaccount "$SPARK_SERVICE_ACCOUNT" -n "$NAMESPACE" &>/dev/null; then
        log_info "Spark ServiceAccount present: $NAMESPACE/$SPARK_SERVICE_ACCOUNT"
    else
        log_error "Spark ServiceAccount missing: $NAMESPACE/$SPARK_SERVICE_ACCOUNT (expected from spark.serviceAccount.create)"
        failed=1
    fi
    phase_end "verify_chart_resources"
    if [[ $failed -ne 0 ]]; then
        exit 1
    fi
}

install_spark_operator() {
    local install_start
    install_start=$(date +%s)
    phase_start "install_spark_operator"

    log_info "Installing Spark Operator from chart: $SPARK_OPERATOR_CHART"

    phase_start "helm_install_or_upgrade"
    local helm_extra_args=(
        --set webhook.enable=true
        --set image.registry=ghcr.io
        --set "image.repository=${CONTROLLER_IMAGE_REPOSITORY}"
        --set "image.tag=$IMAGE_TAG"
        --set "hook.image.tag=$IMAGE_TAG"
        --set "spark.jobNamespaces[0]=$NAMESPACE"
        --set hook.upgradeCrd=true
        --set "controller.defaultServiceAccount=$SPARK_SERVICE_ACCOUNT"
    )
    local helm_timeout="${HELM_TIMEOUT:-15m}"
    log_info "Helm may take up to $helm_timeout. 'context canceled' usually means the process was killed (external timeout or Ctrl+C)."

    local helm_ret=0
    helm upgrade --install "$SPARK_OPERATOR_RELEASE" "$SPARK_OPERATOR_CHART" \
        --namespace "$SPARK_OPERATOR_NAMESPACE" \
        --create-namespace \
        "${helm_extra_args[@]}" \
        --timeout "$helm_timeout" \
        --wait || helm_ret=$?
    phase_end "helm_install_or_upgrade"

    _dump_controller_logs_if_not_ready() {
        local ready
        ready=$(kubectl get deployment -n "$SPARK_OPERATOR_NAMESPACE" "${SPARK_OPERATOR_RELEASE}-controller" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [[ "${ready:-0}" != "1" ]]; then
            log_error "Controller not ready (readyReplicas=${ready:-0}). Pods, hook jobs and controller logs:"
            kubectl get pods,jobs -n "$SPARK_OPERATOR_NAMESPACE" 2>/dev/null || true
            local controller_pod
            controller_pod=$(kubectl get pods -n "$SPARK_OPERATOR_NAMESPACE" -o name 2>/dev/null | grep -E 'controller' | head -1 | sed 's|pod/||')
            if [[ -n "$controller_pod" ]]; then
                kubectl logs -n "$SPARK_OPERATOR_NAMESPACE" "$controller_pod" --tail=80 2>/dev/null || true
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

    verify_chart_resources

    log_info "Spark Operator installed (CRDs and RBAC from Helm chart)"
    local install_end
    install_end=$(($(date +%s) - install_start))
    if [[ $install_end -ge 60 ]]; then
        log_info "Phase done: install_spark_operator (total: ${install_end}s ($((install_end / 60))m $((install_end % 60))s))"
    else
        log_info "Phase done: install_spark_operator (total: ${install_end}s)"
    fi
}

print_status() {
    echo ""
    log_info "=== Cluster Status ==="
    if [[ "$CLUSTER_REUSED" == true ]]; then
        echo "Cluster: kind-$KIND_CLUSTER_NAME (reused from current kubectl context)"
        echo "Kubernetes version: $(kubectl version -o yaml 2>/dev/null | awk '/serverVersion/ {f=1} f && /gitVersion/ {print $2; exit}')"
    else
        echo "Cluster: kind-$KIND_CLUSTER_NAME"
        echo "Kubernetes version: $K8S_VERSION"
    fi
    echo "Test namespace: $NAMESPACE"
    echo "Spark Operator revision: $(git -C "$SPARK_OPERATOR_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Spark Operator chart: $SPARK_OPERATOR_CHART"
    echo "Controller image: $CONTROLLER_IMAGE"
    echo "CRD hook image: $HOOK_IMAGE"
    echo "Spark ServiceAccount: $SPARK_SERVICE_ACCOUNT"
    echo ""
    echo "Spark Operator Deployment:"
    kubectl get deployment -n "$SPARK_OPERATOR_NAMESPACE" 2>/dev/null || true
    echo ""
    echo "Spark Operator CRDs:"
    kubectl get crd -o name 2>/dev/null | grep sparkoperator.k8s.io || true
    echo ""
    echo "Test Namespace ServiceAccounts and Roles:"
    kubectl get serviceaccounts,roles,rolebindings -n "$NAMESPACE" 2>/dev/null || true
    echo ""
    echo "Test Namespace Pods:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "No pods yet"
    echo ""
    log_info "=== Usage ==="
    echo "To run E2E notebook tests:"
    echo "  make test-e2e-notebook"
    echo ""
    echo "To delete cluster:"
    echo "  make test-e2e-delete-cluster"
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
    ensure_spark_operator_checkout
    build_spark_operator_images
    ensure_cluster
    setup_test_namespace
    load_spark_operator_images
    install_spark_operator
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
