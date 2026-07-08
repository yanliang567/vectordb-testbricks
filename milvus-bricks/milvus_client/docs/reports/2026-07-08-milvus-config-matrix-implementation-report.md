# Milvus Config Matrix Upgrade/Rollback Implementation Report

## Scope

This report tracks the implementation and validation work for jsonShredding,
LoonFFI, 2.6->2.6, 2.6->master, and 3.0->master standalone upgrade/rollback
workflows in `milvus-bricks/milvus_client`.

## Implemented

- Added config matrix parameters to the 2.6 standalone upgrade/rollback
  workflow.
- Added Milvus CR config patching for `jsonShreddingEnabled`, `useLoonFFI`,
  and `dataNode.storage.format=vortex` whenever `useLoonFFI=true`.
- Added config snapshots after base deploy, upgrade, optional post-upgrade
  config toggle, and rollback.
- Added independent `rollback-milvus-image` so base image and rollback image
  can differ. This is required for validating a rollback from master to the
  latest 2.6 branch image while still starting from v2.6.18.
- Added optional forward-only bricks for 3.0 schema creation, seed, and
  validation.
- Added `standalone-3-0-upgrade-rollback.yaml` for 3.0 baseline upgrade and
  rollback testing.
- Extended final workflow reports with config matrix parameters.
- Kept `milvus-bricks/2.6/` unchanged.

## Scenario Support Status

| Scenario | Current status | Notes |
| --- | --- | --- |
| `v2.6.18 + jsonShredding -> 2.6-latest + jsonShredding -> rollback v2.6.18` | Implemented and tested | Round 1c passed data validation before upgrade, after upgrade, and after rollback. |
| `v2.6.18 + jsonShredding -> master + jsonShredding -> rollback latest 2.6` | Implemented and tested | Rerun with rollback image `2.6-20260707-e9ee9a47` passed readiness and data validation. |
| `v2.6.18 jsonShredding disabled -> master LoonFFI enabled -> post-upgrade jsonShredding enabled` | Implemented as unsupported negative rollback scenario | Upgrade and forward workload passed; rollback to latest 2.6 became ready, but 2.6 baseline validation failed after rollback. This path is not a supported 2.6 rollback gate because 2.6 does not support LoonFFI. |
| `3.0 baseline -> master -> rollback 3.0 baseline` | Implemented and tested | Round 3 passed data validation before upgrade, after upgrade, and after rollback. |
| `3.0 baseline + jsonShredding -> master + jsonShredding + LoonFFI -> rollback 3.0 baseline` | Implemented and tested | Data validation passed before upgrade, after upgrade, after rollback, and rollback-forward validation. |

## Coverage

### 2.6 rollback-safe schema matrix

The 2.6 matrix uses three collections, each with 5,000 rows in the 4am runs:

- `scalar_dynamic_partition_key`: INT8, INT16, INT32, INT64, FLOAT, DOUBLE,
  BOOL, VARCHAR, JSON, ARRAY<INT64>, ARRAY<FLOAT>, ARRAY<BOOL>,
  ARRAY<VARCHAR>, FLOAT_VECTOR, nullable scalar fields, dynamic fields,
  partition key, and 16 logical partitions.
- `vector_autoid_bm25`: auto ID, BM25 function, analyzer-enabled VARCHAR,
  FLOAT_VECTOR, FLOAT16_VECTOR, BFLOAT16_VECTOR, INT8_VECTOR, BINARY_VECTOR,
  SPARSE_FLOAT_VECTOR, and multiple vector fields in one collection.
- `explicit_partitions_nullable`: VARCHAR primary key, explicit partitions
  `p0` to `p3`, nullable DOUBLE/BOOL/JSON, ARRAY<VARCHAR>, and FLOAT_VECTOR.

2.6 index coverage includes BITMAP, INVERTED, STL_SORT, TRIE, NGRAM,
JSON-path INVERTED, HNSW, IVF_RABITQ, DISKANN, AUTOINDEX, BIN_IVF_FLAT,
SPARSE_INVERTED_INDEX, COSINE, L2, HAMMING, IP, and BM25 metrics.

### 3.0 forward schema matrix

The 3.0 matrix uses three forward-compatible collections:

- `nullable_vector`: nullable FLOAT_VECTOR with null-vector semantics
  validation.
- `geometry_rtree`: GEOMETRY scalar field with RTREE index.
- `timestamptz_ttl`: TIMESTAMPTZ plus TTL-oriented timestamp fields.

### Request coverage

The workflow runs these independent request bricks:

- Setup and validation: `precheck`, `create_schema_matrix`, `seed_data`,
  `validate_data_integrity`.
- Continuous pressure: `search_pressure`, `query_pressure`,
  `query_iterator_scan`, `count_pressure`, `upsert_pressure`,
  `delete_pressure`, `mixed_rw_pressure`.
- Workflow reporting: `generate_workflow_report`.

Pressure configuration used in the 5k 4am runs:

- `pressure-slice-duration-sec=10`
- `pressure-max-workers=4`
- `pressure-batch-size=10`
- `pressure-interval-sec=1`
- `observe-after-upgrade-sec=60`
- `observe-after-rollback-sec=60`

## Local Validation

- Phase 1 targeted pytest:
  `PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py -v`
  -> 11 passed.
- Phase 1 Argo lint:
  `argo lint argo/standalone-2-6-upgrade-rollback.yaml` -> passed.
- Phase 2/3 targeted pytest:
  `PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py -v`
  -> 11 passed.
- Phase 2/3 Argo lint:
  `argo lint argo/standalone-2-6-upgrade-rollback.yaml` -> passed.
- Phase 4 targeted pytest:
  `PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py -v`
  -> 12 passed.
- Phase 4 Argo lint:
  `argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml`
  -> passed.
- Full `milvus_client/tests`:
  `PYTHONPATH=. pytest milvus_client/tests -v` -> 70 passed.
- Full Argo lint:
  `argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/upgrade-rollback-compatibility.yaml`
  -> passed.
- `git diff --check origin/main...HEAD`: passed with no output.
- LoonFFI/vortex update verification:
  `PYTHONPATH=. pytest milvus_client/tests -v` -> 70 passed.
- LoonFFI/vortex Argo lint:
  `argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/upgrade-rollback-compatibility.yaml`
  -> passed.
- LoonFFI/vortex `git diff --check origin/main...HEAD`: passed with no output.

## 4am Image Tags

- 2.6 latest: `harbor.milvus.io/milvusdb/milvus:2.6-20260707-e9ee9a47`
- 3.0 baseline: `harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14`
- master latest: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`

## Planned 4am Validation Rounds

1. 2.6.18 with jsonShredding -> latest 2.6 with jsonShredding -> rollback
   2.6.18, using 2.6 rollback-safe workload.
2. 2.6.18 with jsonShredding -> latest master with jsonShredding -> rollback
   2.6.18, using 2.6 rollback-safe workload.
3. 3.0 baseline -> latest master -> rollback 3.0 baseline, using 3.0
   compatible workload.

## 4am Validation Results

### Round 1c: 2.6.18 jsonShredding -> 2.6 latest -> rollback 2.6.18

- Workflow: `milvus-standalone-2-6-upgrade-rollback-b27gm`
- Status: Argo `Succeeded`, final report `warning`
- Duration: 13m48s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:v2.6.18`
  - target: `harbor.milvus.io/milvusdb/milvus:2.6-20260707-e9ee9a47`
- Config:
  - base/target/rollback `jsonShreddingEnabled=true`
  - `useLoonFFI=false`
  - forward workload disabled
- Data:
  - schema matrix: `milvus_client/manifests/schema_matrix_2_6.yaml`
  - 3 collections x 5,000 rows
  - validations passed before upgrade, after upgrade, and after rollback
  - checksum rows stayed at 5,000 for each collection
- Pressure:
  - attempts: 34
  - passed: 32
  - failed: 2
  - failed slices: `count_pressure_4.json`, `mixed_rw_pressure_21.json`
  - run used `pressure-fail-on-error=false` and `gate-allow-warning=true`
    explicitly to tolerate standalone restart-window unavailability while
    preserving the warning in the report.

Review after Round 1c:

- The workflow closed the full 2.6->2.6 upgrade/rollback loop and preserved
  all baseline data checksums.
- Pressure failures were concentrated around restart windows. Strict regression
  gating should keep `pressure-fail-on-error=true` and
  `gate-allow-warning=false`; exploratory standalone restart runs can opt into
  warning gate explicitly.

### Round 2: 2.6.18 jsonShredding -> master -> rollback 2.6.18

- Workflow: `milvus-standalone-2-6-upgrade-rollback-m6grv`
- Status: terminated after blocker confirmation
- Duration before termination: 14m36s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:v2.6.18`
  - target: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`
- Config:
  - base/target/rollback `jsonShreddingEnabled=true`
  - `useLoonFFI=false`
  - forward workload disabled
- Data:
  - schema matrix: `milvus_client/manifests/schema_matrix_2_6.yaml`
  - 3 collections x 5,000 rows
  - validations passed before upgrade and after upgrade on master
- Failure point:
  - `patch-rollback` completed.
  - `wait-rollback-ready` did not recover.
  - v2.6.18 pod entered CrashLoopBackOff with exit code 134.
  - Key Milvus panic:
    `current version(2.6.18), session version(3.0.0-beta): session version check failure`.

Review after Round 2:

- This is a Milvus product compatibility blocker rather than a workflow
  orchestration bug. Master writes session metadata with version `3.0.0-beta`;
  v2.6.18 refuses to take over that session on rollback.
- Until Milvus defines/supports master->2.6 rollback semantics, this scenario
  should remain a negative compatibility test or blocked release gate, not a
  passing upgrade/rollback workflow.

### Round 3: 3.0 baseline -> master -> rollback 3.0 baseline

- Workflow: `milvus-standalone-3-0-upgrade-rollback-bfl78`
- Status: Argo `Succeeded`, final report `warning`
- Duration: 14m03s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14`
  - target: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`
- Config:
  - base/target/rollback `jsonShreddingEnabled=false`
  - `useLoonFFI=false`
  - forward workload disabled
- Data:
  - schema matrix: `milvus_client/manifests/schema_matrix_3_0.yaml`
  - 3 collections x 5,000 rows
  - validations passed before upgrade, after upgrade, and after rollback
  - rollback-forward validation also passed against the same 3.0 matrix
  - checksum rows stayed at 5,000 for each collection
- Pressure:
  - attempts: 32
  - passed: 28
  - failed: 4
  - failed slices: `count_pressure_4.json`, `delete_pressure_20.json`,
    `mixed_rw_pressure_21.json`, `query_iterator_scan_3.json`
  - two failures were explicit connection-unavailable errors during standalone
    restart windows.

Review after Round 3:

- The 3.0 baseline->master->3.0 baseline path completed the full loop with
  baseline data intact.
- The same pressure behavior appears during restart windows. The current
  explicit warning gate is useful for exploratory standalone validation, but
  strict defaults remain the right CI/release-gate behavior.

### Round 2 fixed rollback image: 2.6.18 jsonShredding -> master -> latest 2.6

- Workflow: `milvus-standalone-2-6-upgrade-rollback-x2bfz`
- Status: Argo `Succeeded`, final report `warning`
- Duration: 15m01s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:v2.6.18`
  - target: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`
  - rollback: `harbor.milvus.io/milvusdb/milvus:2.6-20260707-e9ee9a47`
- Config:
  - base/target/rollback `jsonShreddingEnabled=true`
  - `useLoonFFI=false`
  - forward workload disabled
- Data:
  - schema matrix: `milvus_client/manifests/schema_matrix_2_6.yaml`
  - 3 collections x 5,000 rows
  - validations passed before upgrade, after upgrade, and after rollback
  - checksum rows stayed at 5,000 for each collection
- Pressure:
  - attempts: 31
  - passed: 28
  - failed: 3
  - failed slices: `count_pressure_4.json`, `query_iterator_scan_3.json`,
    `upsert_pressure_19.json`

Review:

- Rolling back to latest 2.6 image `2.6-20260707-e9ee9a47` no longer hits the
  previous `current version(2.6.18), session version(3.0.0-beta)` startup
  panic.
- This validates the upstream fix path for the session-version rollback fence.

### Round 4: 2.6.18 disabled -> master LoonFFI -> post-upgrade jsonShredding -> latest 2.6

- Workflow: `milvus-standalone-2-6-upgrade-rollback-txrzw`
- Status: Argo `Failed`
- Duration: 15m21s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:v2.6.18`
  - target: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`
  - rollback: `harbor.milvus.io/milvusdb/milvus:2.6-20260707-e9ee9a47`
- Config:
  - base `jsonShreddingEnabled=false`
  - target `jsonShreddingEnabled=false`
  - target `useLoonFFI=true`
  - post-upgrade config toggle changed `jsonShreddingEnabled=true`
  - rollback `jsonShreddingEnabled=true`
  - forward workload enabled with `schema_matrix_3_0.yaml`
- Progress:
  - 2.6 baseline create/seed/validate passed.
  - Upgrade to master passed.
  - Post-upgrade 2.6 baseline validation passed.
  - Post-upgrade config toggle passed.
  - 3.0 forward schema create/seed/validate passed.
  - Rollback to latest 2.6 image became ready.
- Failure:
  - `validate-after-rollback` failed.
  - Repeated SDK query errors:
    `channel distribution is not serviceable: channel not available`.
  - Example channel:
    `rootcoord-dml_13_467530583958762666v0`.

Follow-up rerun:

- Workflow: `milvus-standalone-2-6-upgrade-rollback-f8xxc`
- Change: only increased `observe-after-rollback-sec` from 60 to 180.
- Result: Argo `Failed` again at `validate-after-rollback`.
- Same error family:
  `failed to search/query delegator ... channel distribution is not serviceable`.

Review:

- This is no longer the session-version panic; latest 2.6 starts and reaches
  ready.
- Because 2.6 does not support LoonFFI, this is an unsupported negative
  rollback scenario rather than a valid 2.6 release gate.
- The workflow still surfaced the unsupported path as a hard data validation
  failure, which is useful for documenting the compatibility boundary.

### Round 4b: 2.6.18 disabled -> master LoonFFI + vortex -> post-upgrade jsonShredding -> latest 2.6

- Workflow: `milvus-standalone-2-6-upgrade-rollback-x7t79`
- Status: Argo `Failed`
- Duration: 18m03s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:v2.6.18`
  - target: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`
  - rollback: `harbor.milvus.io/milvusdb/milvus:2.6-20260707-e9ee9a47`
- Config:
  - base `jsonShreddingEnabled=false`
  - target upgrade patch set `jsonShreddingEnabled=false`,
    `useLoonFFI=true`, and `dataNode.storage.format=vortex`
  - post-upgrade config toggle set `jsonShreddingEnabled=true`,
    retained `useLoonFFI=true`, and retained `dataNode.storage.format=vortex`
  - rollback `jsonShreddingEnabled=true`
  - forward workload enabled with `schema_matrix_3_0.yaml`
- Progress:
  - 2.6 baseline create/seed/validate passed.
  - Upgrade to master with LoonFFI plus vortex passed.
  - Post-upgrade 2.6 baseline validation passed.
  - Post-upgrade config toggle passed.
  - 3.0 forward schema create/seed/validate passed.
  - Rollback to latest 2.6 image became ready.
- Failure:
  - `validate-after-rollback` failed.
  - Repeated SDK query errors:
    `channel distribution is not serviceable: channel not available`.

Review:

- The requested LoonFFI companion config was applied in both target upgrade
  and post-upgrade config toggle phases.
- The final rollback failure is still expected for this negative scenario:
  data written after enabling LoonFFI is not a supported rollback path to 2.6.

### Round 5: 3.0 jsonShredding -> master jsonShredding + LoonFFI -> rollback 3.0

- Workflow: `milvus-standalone-3-0-upgrade-rollback-7xxmz`
- Status: Argo `Succeeded`, final report `warning`
- Duration: 16m08s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14`
  - target: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`
  - rollback: `harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14`
- Config:
  - base/target/rollback `jsonShreddingEnabled=true`
  - target `useLoonFFI=true`
  - forward workload disabled because the base schema matrix is already 3.0
- Data:
  - schema matrix: `milvus_client/manifests/schema_matrix_3_0.yaml`
  - 3 collections x 5,000 rows
  - validations passed before upgrade, after upgrade, after rollback, and
    rollback-forward validation
  - checksum rows stayed at 5,000 for each collection
- Pressure:
  - attempts: 35
  - passed: 31
  - failed: 4
  - failed slices: `delete_pressure_6.json`, `query_pressure_23.json`,
    `search_pressure_22.json`, `upsert_pressure_5.json`

Review:

- The 3.0 jsonShredding + target LoonFFI rollback path completed with data
  intact.
- Remaining pressure failures are consistent with standalone restart-window
  unavailability and were explicitly warning-gated.

### Round 5b: 3.0 jsonShredding -> master jsonShredding + LoonFFI + vortex -> rollback 3.0

- Workflow: `milvus-standalone-3-0-upgrade-rollback-dzprm`
- Status: Argo `Succeeded`, final report `warning`
- Duration: 14m04s
- Images:
  - base: `harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14`
  - target: `harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a`
  - rollback: `harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14`
- Config:
  - base/target/rollback `jsonShreddingEnabled=true`
  - target upgrade patch set `useLoonFFI=true` and
    `dataNode.storage.format=vortex`
  - forward workload disabled because the base schema matrix is already 3.0
- Data:
  - schema matrix: `milvus_client/manifests/schema_matrix_3_0.yaml`
  - 3 collections x 5,000 rows
  - validations passed before upgrade, after upgrade, after rollback, and
    rollback-forward validation
- Pressure:
  - attempts: 33
  - passed: 30
  - failed: 3

Review:

- The 3.0 LoonFFI path is the supported rollback path for the new vortex
  companion config and completed with data validation intact.
- The final report stayed at `warning` only because pressure failures were
  explicitly warning-gated for standalone restart-window behavior.

## Blocking Issues

- Round 1 initial run
  `milvus-standalone-2-6-upgrade-rollback-jxjmt` failed in
  `generate-final-report` because `resolve-inputs` wrote
  `env_snapshot.json` and `flow_summary.json` to its local filesystem without
  mounting the shared `milvus-test-state` PVC. Foreground data validation,
  upgrade, rollback, optional skipped forward bricks, and pressure result
  collection had completed before the report artifact failure. Fix: mount the
  shared PVC in `resolve-inputs` and add report-stage fallback env/flow snapshot
  generation.
- Round 2
  `milvus-standalone-2-6-upgrade-rollback-m6grv` found a product rollback
  blocker when rolling back from master to v2.6.18. Milvus v2.6.18 panicked
  during registration with
  `current version(2.6.18), session version(3.0.0-beta): session version check failure`.
  The workflow correctly reached rollback and waited for readiness; the target
  version could not become ready.
- Round 4 / Round 4b
  `milvus-standalone-2-6-upgrade-rollback-txrzw` and rerun
  `milvus-standalone-2-6-upgrade-rollback-f8xxc` first documented the
  post-rollback failure for the 2.6 disabled -> master LoonFFI -> post-upgrade
  jsonShredding -> latest 2.6 path. Round 4b
  `milvus-standalone-2-6-upgrade-rollback-x7t79` reproduced the same
  unsupported rollback boundary after adding the required vortex companion
  config. Because 2.6 does not support LoonFFI, this should remain an
  unsupported negative scenario rather than a product rollback gate.

## Optimization Notes

- Optional 3.0 forward workload is implemented as regular DAG tasks that write
  `skipped` results when disabled. This avoids ambiguous skipped dependency
  behavior in Argo and keeps report artifacts explicit.
- Pressure daemon remains PVC-free and communicates through workflow-owned
  ConfigMaps, preserving the previous RWO fix.
- Added explicit `gate-allow-warning=false` workflow parameter. The default
  remains strict. Standalone restart-window exploratory runs can set
  `pressure-fail-on-error=false` and `gate-allow-warning=true` so Argo succeeds
  while the report still records pressure status as `warning`.
- Keep strict defaults for formal regression runs. Use warning gate only when
  the test objective is to validate data compatibility across standalone
  restarts and separately record transient pressure failures.
- Add a product-level decision for master/3.0 metadata rollback. If session
  metadata is intentionally forward-only, the 2.6 rollback scenario should
  become a documented unsupported negative test.
- LoonFFI hard gates should target 3.0 baseline -> master -> 3.0 rollback
  unless product explicitly adds 2.6 rollback compatibility for LoonFFI data.
- For unsupported 2.6 rollback after a master phase with forward-only 3.0
  collections, keep collecting validator logs and config snapshots. Deeper
  QueryNode/channel diagnostics are useful for product analysis, but this path
  should not block supported release validation.
