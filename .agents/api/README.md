# Machine-readable API surfaces

Start at **[surfaces.yaml](./surfaces.yaml)** — SDK OpenAPI path, CRD inventory, upstream pins, external APIs.

## Commands

```bash
make verify-openapi    # Validate openapi.yaml (openapi-spec-validator via uv)
make lint-imports      # SDK component import boundaries (import-linter)
```
