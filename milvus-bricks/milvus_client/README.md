# MilvusClient Test Bricks

This directory contains Milvus test bricks based on `pymilvus.MilvusClient`
and `AsyncMilvusClient`. It keeps the migrated 2.6 scripts and adds a new
request-brick runtime for Milvus 2.6/3.0 feature coverage.

## Layout

```text
milvus_client/
  common/        Shared runtime: args, results, schema, data, validation
  requests/      Standalone request bricks
  scenarios/     Multi-step scenario plans
  manifests/     Feature, capability, schema, and brick catalogs
  tests/         Unit tests for the runtime and manifests
  horizonPoc/    Existing Horizon POC scripts
```

The old helper module was moved from `common.py` to `common_legacy.py` and is
re-exported through the `common` package for compatibility with existing
scripts that use `from common import ...`.

## Request Protocol

New request bricks use a shared CLI shape:

```bash
PYTHONPATH=. python -m milvus_client.requests.precheck \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_brick \
  --feature-set compat_2_6 \
  --compat-mode rollback_safe \
  --capability-probe true \
  --skip-unsupported true \
  --lifecycle-phase steady_state \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/precheck.json
```

Every request writes a structured JSON result containing:

- `status`: `passed`, `failed`, `warning`, or `skipped`
- `feature_set`, `compat_mode`, `lifecycle_phase`
- `capabilities` and `skip_reason`
- `metrics`, `failures`, `artifacts`, and `checkpoint`

## P0 Bricks

```bash
# Validate manifests without connecting to Milvus
PYTHONPATH=. python -m milvus_client.requests.create_schema_matrix \
  --uri http://localhost:19530 \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --dry-run \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/create_schema_matrix.json

# Create collections from a schema matrix
PYTHONPATH=. python -m milvus_client.requests.create_schema_matrix \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/create_schema_matrix.json

# Seed deterministic data and checkpoint expected counts
PYTHONPATH=. python -m milvus_client.requests.seed_data \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --rows-per-collection 1000 \
  --batch-size 100 \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/seed_data.json

# Validate checkpointed data
PYTHONPATH=. python -m milvus_client.requests.validate_data_integrity \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/validate_data_integrity.json
```

`validate_data_integrity` validates the seed checkpoint by PK range, so extra
rows written by `mixed_rw_pressure` outside that range do not cause false count
drift. The checkpoint checksum covers deterministic non-vector fields; vector
compatibility is covered by search/query workload bricks.

## Upgrade/Rollback Scenario

The scenario runner currently supports dry-run planning. Upgrade and rollback
actions are intentionally external so Argo, Helm, or another controller can own
cluster lifecycle operations.

```bash
PYTHONPATH=. python -m milvus_client.scenarios.upgrade_rollback_compatibility \
  --dry-run \
  --scenario-manifest milvus_client/manifests/scenario_upgrade_rollback.yaml \
  --uri http://localhost:19530 \
  --collection-prefix qa_upgrade \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/upgrade_rollback_dry_run.json
```

## Tests

Run from `milvus-bricks/`:

```bash
PYTHONPATH=. pytest milvus_client/tests -v
```
