# Machine-readable API surfaces

Index of schemas agents can diff against requirements. Start at **[surfaces.yaml](./surfaces.yaml)**.

## Files

| File | Purpose |
|------|---------|
| [surfaces.yaml](./surfaces.yaml) | Index — SDK OpenAPI, CRD paths, upstream pins, external REST |
| [../../openapi.yaml](../../openapi.yaml) | SDK logical operation catalog (all public client methods) |
| `hack/crds/` | Vendored Kubernetes CRD YAML (SparkConnect only) |

## Commands

```bash
make verify-openapi    # Validate openapi.yaml (openapi-spec-validator via uv)
```

Trainer and Katib CRD schemas are upstream only — see `upstream_url` entries in `surfaces.yaml`.

## Notes

This repository is a **Python SDK**, not a REST server. Paths in `openapi.yaml` name client
operations (`Class.method`), not HTTP URLs.

Model Registry REST and Spark Connect gRPC are external — see `external_rest` and `grpc` in
`surfaces.yaml`.
