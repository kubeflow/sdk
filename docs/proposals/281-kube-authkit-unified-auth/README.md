# KEP-281: Adopt kube-authkit as Unified Authentication for Kubeflow SDK

## Authors

- Saad Zaher - [@szaher](https://github.com/szaher)

Ref: https://github.com/kubeflow/sdk/issues/281

## Summary

This KEP proposes donating the [kube-authkit](https://github.com/szaher/kube-authkit)
library to the Kubeflow project, embedding it in the SDK repository under
`kubeflow/common/auth/`, and publishing it as a standalone PyPI package
(`kubeflow-authkit`). kube-authkit replaces fragmented authentication across
SDK backends: duplicated Kubernetes auth logic in Trainer, Optimizer, and Spark,
plus Hub's separate `user_token`-based path, with a single, strategy-based
authentication layer supporting KubeConfig, in-cluster service account, OIDC,
and OpenShift OAuth.

## Motivation

Authentication in the Kubeflow SDK is fragmented across multiple components, each
implementing its own client initialization logic with subtle variations:

- **Trainer, Optimizer, and Spark backends** each independently call
  `config.load_kube_config()` / `config.load_incluster_config()` with their own
  error handling and fallback logic.
- **Hub (`ModelRegistryClient`)** uses an entirely different auth model based on
  a `user_token` string parameter with no integration to cluster authentication.
- **No OIDC or OpenShift OAuth support** exists anywhere in the SDK. Users in
  enterprise and OpenShift environments must manually obtain and manage tokens.
- **Auth-adjacent helpers** like `is_running_in_k8s()` and
  `get_default_target_namespace()` live in `kubeflow.common.utils` without a
  coherent authentication abstraction around them.

This duplication creates inconsistent behavior, increases maintenance burden, and
limits the SDK to only two authentication methods (KubeConfig and in-cluster)
while the broader Kubernetes ecosystem has moved toward OIDC-based identity.

kube-authkit already solves these problems with a production-tested, strategy-based
authentication library that supports automatic environment detection across four
authentication methods. Donating it to Kubeflow and embedding it in the SDK
provides a unified auth layer for all SDK components while publishing it as an
independent package enables adoption across the wider Kubeflow ecosystem.

### Goals

1. Provide a single, well-tested authentication entry point for all SDK
   components (Trainer, Optimizer, Spark, Hub).
2. Support KubeConfig, in-cluster service account, OIDC (Authorization Code +
   PKCE, Device Code, Client Credentials), and OpenShift OAuth authentication
   methods.
3. Auto-detect the appropriate authentication method based on the runtime
   environment with zero configuration required for common cases.
4. Publish `kubeflow-authkit` as an independent PyPI package consumable by other
   Kubeflow components (e.g., Pipelines SDK) and external tools without requiring
   the full `kubeflow` SDK.
5. Unify Hub's REST-based authentication with the Kubernetes-based authentication
   used by other components through a shared `AuthResult` that exposes both a
   configured `ApiClient` and a resolved bearer token.
6. Maintain full backward compatibility with existing `KubernetesBackendConfig`
   fields (`config_file`, `context`, `client_configuration`) and
   `ModelRegistryClient` parameters (`user_token`, `custom_ca`) throughout the
   migration.

### Non-Goals

- Provider-specific OIDC configuration guides (Dex, Keycloak, cloud-native OIDC).
  These belong in follow-up documentation, not in this KEP.
- Bumping the SDK's minimum `kubernetes` Python client version. kube-authkit does
  not hard-depend on a specific version and will align with whatever floor the SDK
  requires.
- Replacing the Kubernetes Python client itself. kube-authkit wraps it, it does
  not replace it.
- Authentication for the Kubeflow Pipelines SDK. Pipelines lives in a separate
  repository but can consume `kubeflow-authkit` as an independent package.

---

## Proposal

Donate kube-authkit to the Kubeflow project by embedding it in the SDK repository
under `kubeflow/common/auth/`. The module serves dual purpose:

1. **First-party import** — SDK backends import `kubeflow.common.auth` directly,
   replacing their individual `load_kube_config` / `load_incluster_config` blocks.
2. **Standalone package** — Published to PyPI as `kubeflow-authkit` with its own
   version, enabling other Kubeflow components and external tools to use it
   independently.

The public API centers on two constructs:

- `AuthConfig` — A Pydantic model for authentication configuration (method,
  OIDC parameters, TLS settings). Defaults to auto-detection when no method is
  specified.
- `get_k8s_client()` — Accepts an optional `AuthConfig` and returns an
  `AuthResult` containing a configured `kubernetes.client.ApiClient` and an
  optional resolved bearer token.

For Hub, the resolved token from `AuthResult` replaces the manual `user_token`
parameter, unifying the authentication path across all SDK components.

### User Stories

#### Story 1: Data scientist in a Jupyter notebook on a Kubeflow cluster

```python
from kubeflow.trainer import TrainerClient

# Auth is automatic - in-cluster service account detected, zero config
client = TrainerClient()
client.train(...)
```

No change from today's experience. kube-authkit detects the in-cluster
environment and authenticates transparently.

#### Story 2: ML engineer on a laptop connecting to an OpenShift cluster

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.auth import AuthConfig

# Auto-detects OpenShift OAuth from kubeconfig context
client = TrainerClient()

# Or explicit OIDC with device flow for headless environments
client = TrainerClient(
    auth=AuthConfig(
        method="oidc",
        oidc_issuer="https://keycloak.corp.com/realms/ml",
        client_id="kubeflow-cli",
        use_device_flow=True,
    )
)
```

#### Story 3: CI/CD pipeline registering a model

```python
from kubeflow.hub import ModelRegistryClient
from kubeflow.common.auth import AuthConfig

# Same AuthConfig works for Hub - token extracted automatically
client = ModelRegistryClient(
    base_url="https://registry.internal",
    auth=AuthConfig(),
)
client.register_model(...)
```

#### Story 4: External tool consuming auth independently

```python
# pip install kubeflow-authkit
from kubeflow_authkit import AuthConfig, AuthResult, get_k8s_client

api_client = get_k8s_client(AuthConfig(method="oidc", ...))
```

### Notes/Constraints/Caveats

- **Keyring dependency is optional.** Token persistence via system keyring is
  available through `pip install kubeflow-authkit[keyring]`. The default install
  has no keyring dependency.
- **`kubernetes` version floor.** kube-authkit does not hard-pin a `kubernetes`
  version. It will align with the SDK's existing `kubernetes>=27.2.0` requirement.
  Users who independently upgrade to newer `kubernetes` versions benefit from
  security fixes (e.g., CVE patches in `kubernetes>=35.0.0`) without the SDK
  forcing the upgrade.
- **Dual-package build.** The `kubeflow-authkit` package and the `kubeflow`
  package are built from the same source tree. The SDK's `pyproject.toml` lists
  `kubeflow-authkit` as a dependency, but during development the code is available
  as a first-party import.

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing `KubernetesBackendConfig` users who pass `config_file`, `context`, or `client_configuration` | These fields are retained throughout the migration. `AuthConfig` is additive — it takes precedence only when explicitly set. Legacy fields continue to work until formally deprecated in a future major version. |
| OIDC flows requiring browser interaction in headless environments | Device Code flow provides a headless alternative. Client Credentials flow supports fully automated service-to-service authentication. |
| Hub's REST auth model diverging from Kubernetes auth | `AuthResult` exposes both a configured `ApiClient` and a resolved bearer `token`, allowing Hub to extract the token for REST API calls. |
| Dual-package build complexity | Hatchling supports multi-package builds from a single repo. CI validates both packages independently. |
| Auth scope creep — community requesting features beyond SDK needs | Clear non-goals and API boundaries defined in this KEP. Feature requests evaluated against the stated goals. |

---

## Design Details

### Module Layout

```
kubeflow/
├── common/
│   ├── auth/
│   │   ├── __init__.py          # Public API: AuthConfig, get_k8s_client, AuthResult
│   │   ├── config.py            # AuthConfig Pydantic model
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # AuthStrategy protocol
│   │   │   ├── kubeconfig.py    # KubeConfig strategy
│   │   │   ├── incluster.py     # In-cluster service account strategy
│   │   │   ├── oidc.py          # OIDC strategy (all flows)
│   │   │   └── openshift.py     # OpenShift OAuth strategy
│   │   ├── resolver.py          # Auto-detection logic (strategy selection)
│   │   ├── tokens.py            # Token refresh and keyring storage
│   │   └── auth_test.py         # Unit tests
│   ├── types.py                 # Existing KubernetesBackendConfig (updated)
│   └── utils.py                 # Existing utils (auth helpers moved to auth/)
```

### Public API

```python
from pydantic import BaseModel
from kubernetes import client


class AuthConfig(BaseModel):
    """Authentication configuration. Zero-config by default.

    Args:
        method: Authentication method to use. When None, the environment is
            auto-detected in order: in-cluster, kubeconfig, OIDC, OpenShift.
        config_file: Path to kubeconfig file. Defaults to ~/.kube/config.
        context: Kubeconfig context to use.
        k8s_api_host: Kubernetes API server URL.
        oidc_issuer: OIDC issuer URL.
        client_id: OIDC client ID.
        client_secret: OIDC client secret.
        scopes: OIDC scopes to request.
        use_device_flow: Use OIDC Device Code flow instead of Authorization
            Code flow.
        use_keyring: Persist tokens in system keyring for reuse across sessions.
        ca_cert: Path to CA certificate for TLS verification.
        verify_ssl: Enable TLS verification. Defaults to True.
    """

    method: str | None = None
    config_file: str | None = None
    context: str | None = None
    k8s_api_host: str | None = None
    oidc_issuer: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    scopes: list[str] | None = None
    use_device_flow: bool = False
    use_keyring: bool = False
    ca_cert: str | None = None
    verify_ssl: bool = True


class AuthResult(BaseModel):
    """Result of authentication.

    Provides both a Kubernetes API client and a resolved bearer token,
    enabling use with both Kubernetes API and REST-based services (e.g., Hub).

    Args:
        api_client: Configured Kubernetes API client.
        token: Resolved bearer token, if available.
        ca_cert: CA certificate path used for TLS, if applicable.
    """

    api_client: client.ApiClient
    token: str | None = None
    ca_cert: str | None = None

    class Config:
        arbitrary_types_allowed = True


def get_k8s_client(config: AuthConfig | None = None) -> AuthResult:
    """Authenticate and return a configured Kubernetes API client.

    Detects the runtime environment automatically when no method is specified
    in the config. Detection order: in-cluster service account, kubeconfig,
    OIDC, OpenShift OAuth.

    Args:
        config: Authentication configuration. Defaults to auto-detection.

    Returns:
        AuthResult containing the configured API client and optional token.

    Raises:
        AuthenticationError: If no viable authentication method is found.
    """
    ...
```

### Integration into Existing Backends

The `KubernetesBackendConfig` gains an optional `auth` field:

```python
class KubernetesBackendConfig(BaseModel):
    namespace: str | None = None
    config_file: str | None = None              # Retained for backward compat
    context: str | None = None                  # Retained for backward compat
    client_configuration: client.Configuration | None = None  # Retained
    auth: AuthConfig | None = None              # New - takes precedence when set

    class Config:
        arbitrary_types_allowed = True
```

Backend initialization (currently duplicated in Trainer, Optimizer, and Spark)
changes from:

```python
# Before (duplicated in each backend)
if cfg.config_file or not common_utils.is_running_in_k8s():
    config.load_kube_config(config_file=cfg.config_file, context=cfg.context)
else:
    config.load_incluster_config()
k8s_client = client.ApiClient(cfg.client_configuration)
```

To:

```python
# After (shared across all backends)
from kubeflow.common.auth import AuthConfig, get_k8s_client

if cfg.client_configuration is not None:
    # Explicit client configuration takes highest precedence
    k8s_client = client.ApiClient(cfg.client_configuration)
elif cfg.auth is not None:
    auth_result = get_k8s_client(cfg.auth)
    k8s_client = auth_result.api_client
else:
    # Backward compat: build AuthConfig from legacy fields
    auth_config = AuthConfig(config_file=cfg.config_file, context=cfg.context)
    auth_result = get_k8s_client(auth_config)
    k8s_client = auth_result.api_client
```

### Hub Integration

`ModelRegistryClient` gains an optional `auth` parameter. When provided, the
resolved token from `AuthResult` replaces the manual `user_token`:

```python
class ModelRegistryClient:
    def __init__(
        self,
        base_url: str,
        port: int | None = None,
        *,
        author: str | None = None,
        is_secure: bool | None = None,
        user_token: str | None = None,      # Retained for backward compat
        custom_ca: str | None = None,       # Retained for backward compat
        auth: AuthConfig | None = None,     # New - takes precedence
    ):
        if auth is not None:
            auth_result = get_k8s_client(auth)
            user_token = auth_result.token
            if auth_result.ca_cert and custom_ca is None:
                custom_ca = auth_result.ca_cert
        ...
```

### Standalone Package

The `kubeflow-authkit` package is built from the same source tree under
`kubeflow/common/auth/`. It uses a separate build configuration to produce an
independent PyPI package that exports:

```
kubeflow_authkit.AuthConfig
kubeflow_authkit.AuthResult
kubeflow_authkit.get_k8s_client
```

The SDK's `pyproject.toml` adds `kubeflow-authkit` as a core dependency. During
development, the code is available as a first-party import via
`kubeflow.common.auth`.

### Test Plan

[x] I/we understand the owners of the involved components may require updates to
existing tests to make this code solid enough prior to committing the changes
necessary to implement this enhancement.

#### Prerequisite Testing Updates

Existing backend tests that mock `kubernetes.config.load_kube_config` and
`kubernetes.config.load_incluster_config` will need to be updated to mock the
new `get_k8s_client` entry point instead.

#### Unit Tests

- `kubeflow/common/auth/` — Strategy selection logic, `AuthConfig` validation,
  token refresh, fallback behavior, error handling for each strategy
- `kubeflow/trainer/backends/kubernetes/` — Verify `AuthConfig` integration,
  backward compatibility with legacy `config_file`/`context` fields
- `kubeflow/optimizer/backends/kubernetes/` — Same as Trainer
- `kubeflow/spark/backends/kubernetes/` — Same as Trainer
- `kubeflow/hub/api/` — Verify token extraction from `AuthResult`, backward
  compatibility with `user_token`
- Coverage target: >=90% for `kubeflow/common/auth/`

Packages and current coverage:

- `kubeflow.common.auth`: N/A (new package)
- `kubeflow.trainer.backends.kubernetes`: 2026-04-23 — existing coverage
- `kubeflow.optimizer.backends.kubernetes`: 2026-04-23 — existing coverage
- `kubeflow.spark.backends.kubernetes`: 2026-04-23 — existing coverage
- `kubeflow.hub.api`: 2026-04-23 — existing coverage

#### E2E Tests

- Full `TrainerClient` workflow with `AuthConfig` — submit and monitor a training
  job using OIDC authentication
- `ModelRegistryClient` workflow — register a model using auth-derived token
- Multi-backend test — verify consistent auth behavior across Trainer, Optimizer,
  and Spark using the same `AuthConfig`

#### Integration Tests

- In-cluster authentication in a Kind cluster
- KubeConfig authentication with multiple contexts
- OIDC flow with a test identity provider (mock OIDC server)
- OpenShift OAuth with a mock OAuth server
- Token refresh and expiry handling

### Graduation Criteria

| Phase | SDK Version | Behavior | Duration |
|-------|------------|----------|----------|
| **Alpha** | Next minor | `auth=AuthConfig(...)` available on all backends and Hub as opt-in. Legacy fields continue to work unchanged. No deprecation warnings. | 1 release cycle |
| **Beta** | Following minor | `get_k8s_client()` used by default when no `client_configuration` is set. Legacy fields still work but emit deprecation warnings. Documentation updated to recommend `AuthConfig`. | 1-2 release cycles |
| **Stable** | Following minor + 1 | `AuthConfig` is the primary auth interface. Legacy fields (`config_file`, `context`, `client_configuration` on `KubernetesBackendConfig`; `user_token`, `custom_ca` on `ModelRegistryClient`) are deprecated. | — |
| **Cleanup** | Following minor + 2 | Deprecated fields removed. | — |

---

## Implementation History

- 2026-04-23: KEP created
- TBD: KEP reviewed and approved by Kubeflow community
- TBD: Alpha — kube-authkit code donated and integrated into SDK
- TBD: Beta — default auth path switched to kube-authkit
- TBD: Stable — legacy auth fields deprecated
- TBD: Cleanup — deprecated fields removed

## Drawbacks

- **Build and release complexity.** Publishing two packages (`kubeflow` and
  `kubeflow-authkit`) from one repository requires additional CI configuration
  and coordinated versioning. This is manageable with Hatchling workspaces but
  adds maintenance surface.
- **Auth scope creep.** A unified auth module may attract requests for features
  beyond the SDK's needs (e.g., multi-tenant token management, auth proxy
  support). Clear API boundaries and non-goals defined in this KEP help contain
  scope.
- **Migration burden.** While fully backward compatible, users reading examples or
  tutorials will encounter two ways to configure auth until the legacy fields are
  removed. Clear deprecation warnings and updated documentation ease this
  transition.
- **Keyring dependency.** The optional keyring support introduces a
  platform-specific dependency. Keeping it behind
  `pip install kubeflow-authkit[keyring]` ensures it does not affect the default
  install path.

## Alternatives

### Alternative A: Dedicated repository (`kubeflow/kube-authkit`)

Maintain kube-authkit as a standalone repository under the Kubeflow organization
with its own OWNERS file, CI pipelines, and release cycle.

**Pros:**

- Clean separation of concerns with independent versioning.
- The existing `szaher/kube-authkit` repository can be transferred as-is.
- Clear ownership boundary.

**Cons:**

- Fragments ownership across repositories. The community has expressed concern
  about maintaining another standalone repo.
- Cross-repo pull requests required for breaking changes.
- Separate CI/CD pipelines to maintain.
- Harder to enforce consistency with SDK coding standards and conventions.

**Why not chosen:** The community prefers centralized maintenance under the SDK
repository. The proposed hybrid approach achieves independent packaging without
repository fragmentation.

### Alternative B: Embed in SDK with no separate package

Integrate kube-authkit directly into `kubeflow/common/auth/` as an internal
module with no standalone PyPI package.

**Pros:**

- Simplest build setup — one repository, one package, one release.
- No version coordination between packages.

**Cons:**

- Other Kubeflow components (Pipelines SDK, standalone CLI tools) that need auth
  would have to depend on the full `kubeflow` package, pulling in Trainer,
  Optimizer, Spark, and their transitive dependencies.
- Impractical for lightweight consumers that only need authentication.

**Why not chosen:** Forcing a heavyweight dependency for auth-only use violates
the principle of minimal dependencies. A standalone package enables adoption
across the Kubeflow ecosystem without unnecessary coupling.

### Alternative C: Thin wrapper around upstream kube-authkit

Keep kube-authkit as an external package (`pip install kube-authkit`) and add a
thin wrapper in the SDK that delegates to it.

**Pros:**

- No code duplication. Upstream improvements flow in automatically.
- Minimal changes to the SDK repository.

**Cons:**

- External dependency outside Kubeflow governance and release processes.
- Version pinning and compatibility become the SDK's problem.
- No ability to enforce Kubeflow coding standards on the authentication code.
- Community cannot directly maintain or review the auth implementation.

**Why not chosen:** Donating the code to Kubeflow gives the community full
ownership and control over the authentication layer. A thin wrapper would add
indirection without the governance benefits.
