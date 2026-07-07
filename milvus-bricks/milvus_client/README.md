# MilvusClient Test Bricks

This directory contains Milvus test bricks based on `pymilvus.MilvusClient`
and `AsyncMilvusClient`. It adds a new request-brick runtime for Milvus 2.6/3.0
feature coverage while the existing `milvus-bricks/2.6/` directory remains
available for legacy script users.

## Layout

```text
milvus_client/
  common/        Shared runtime: args, results, schema, data, validation
  requests/      Standalone request bricks
  scenarios/     Multi-step scenario plans
  manifests/     Feature, capability, schema, and brick catalogs
  tests/         Unit tests for the runtime and manifests
  horizonPoc/    Copied Horizon POC scripts for future request-brick migration
```

The original `milvus-bricks/2.6/` files are intentionally kept in place. Copied
scripts in this directory are compatibility material for future wrapper/request
migration; new automation should use modules under `milvus_client.requests`.

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
  --load-after-create true \
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

For `auto_id` schemas, `seed_data` captures the server-returned primary keys and
stores them in the checkpoint. `validate_data_integrity` uses those captured
keys for PK samples and checksum queries.

## Independent Workload Bricks

The mixed workload is also available as independently schedulable request
bricks:

- `milvus_client.requests.search_pressure`
- `milvus_client.requests.query_pressure`
- `milvus_client.requests.query_iterator_scan`
- `milvus_client.requests.count_pressure`
- `milvus_client.requests.upsert_pressure`
- `milvus_client.requests.delete_pressure`

Example:

```bash
PYTHONPATH=. python -m milvus_client.requests.search_pressure \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --duration-sec 60 \
  --max-workers 4 \
  --batch-size 10 \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/search_pressure.json
```

`delete_pressure` only targets the reserved high-PK pressure range and does not
delete seed baseline rows tracked by `seed_data`.
For `auto_id` schemas, destructive pressure operations (`upsert` and `delete`)
are skipped; insert/query/search pressure still runs.
`count_pressure` and the count operation inside `mixed_rw_pressure` check row
counts while traffic is running. For explicit PK schemas they verify the seed PK
range still has exactly `--baseline-rows-per-collection` rows; for `auto_id`
schemas they verify the total collection count is at least that baseline.

## Upgrade/Rollback Scenario

The scenario runner supports dry-run planning and non-dry-run closed-loop
execution. Upgrade and rollback actions are intentionally external so Argo,
Helm, or another controller can own cluster lifecycle operations.

```bash
PYTHONPATH=. python -m milvus_client.scenarios.upgrade_rollback_compatibility \
  --dry-run \
  --scenario-manifest milvus_client/manifests/scenario_upgrade_rollback.yaml \
  --uri http://localhost:19530 \
  --collection-prefix qa_upgrade \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/upgrade_rollback_dry_run.json
```

Non-dry-run execution creates and seeds compat schema, starts continuous mixed
RW pressure and validation loops, waits for external upgrade/rollback signals,
creates and validates forward-only schema after upgrade, validates compat schema
after rollback, and performs a final compat validation.

See `docs/upgrade-rollback.md` for details.

## 4am 2.6 Standalone Upgrade/Rollback

`../argo/standalone-2-6-upgrade-rollback.yaml` is a concrete 4am Argo
WorkflowTemplate for the rollback-safe 2.6 path:

- run client/workflow pods in `qa` with scoped RBAC from
  `../argo/standalone-2-6-upgrade-rollback-rbac.yaml`;
- deploy the Milvus standalone CR in `qa-milvus`;
- create a 4c16g Milvus standalone from `v2.6.18`;
- create and seed `schema_matrix_2_6.yaml` data only;
- run baseline integrity validation;
- keep a pressure daemon loop alive during upgrade and rollback;
- upgrade to a configured 2.6 target image;
- validate existing data after upgrade;
- roll back to `v2.6.18`;
- validate existing data after rollback.

It intentionally skips 3.0 schema and workload creation because new 3.0 data is
not rollback compatible with 2.6. The default cleanup policy is
`keep-milvus=false`; use `keep-milvus=true` only for debug retention.

## Tests

Run from `milvus-bricks/`:

```bash
PYTHONPATH=. pytest milvus_client/tests -v
```
