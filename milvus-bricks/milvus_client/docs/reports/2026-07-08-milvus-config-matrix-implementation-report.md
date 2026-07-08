# Milvus Config Matrix Upgrade/Rollback Implementation Report

## Scope

This report tracks the implementation and validation work for jsonShredding,
LoonFFI, 2.6->2.6, 2.6->master, and 3.0->master standalone upgrade/rollback
workflows in `milvus-bricks/milvus_client`.

## Implemented

- Added config matrix parameters to the 2.6 standalone upgrade/rollback
  workflow.
- Added Milvus CR config patching for `jsonShreddingEnabled` and `useLoonFFI`.
- Added config snapshots after base deploy, upgrade, optional post-upgrade
  config toggle, and rollback.
- Added optional forward-only bricks for 3.0 schema creation, seed, and
  validation.
- Added `standalone-3-0-upgrade-rollback.yaml` for 3.0 baseline upgrade and
  rollback testing.
- Extended final workflow reports with config matrix parameters.
- Kept `milvus-bricks/2.6/` unchanged.

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

## Blocking Issues

- None recorded yet.

## Optimization Notes

- Optional 3.0 forward workload is implemented as regular DAG tasks that write
  `skipped` results when disabled. This avoids ambiguous skipped dependency
  behavior in Argo and keeps report artifacts explicit.
- Pressure daemon remains PVC-free and communicates through workflow-owned
  ConfigMaps, preserving the previous RWO fix.
