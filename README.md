# onyx-api

FastAPI server for Onyx inference.

## Registry + Artifact Cache (recommended)

Use `model_id` requests backed by `model-registry` + `artifact-cache-service`:

- `ONYX_REGISTRY_URL=http://127.0.0.1:8001`
- `ONYX_REGISTRY_API_KEY=...`
- `ONYX_ARTIFACT_CACHE_URL=http://127.0.0.1:8002`
- Optional safety gate: `ONYX_MIN_REGISTRY_STATUS=staging` (rejects `experimental`)

Local path overrides are dev-only:
- `ONYX_ALLOW_LOCAL_PATHS=1`
- `ONYX_ALLOWED_PATH_PREFIXES=/some/allowlisted/dir`
