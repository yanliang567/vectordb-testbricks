# Milvus 2.6 Standalone Upgrade/Rollback Validation Report

Date: 2026-07-07
PR: https://github.com/yanliang567/vectordb-testbricks/pull/3
Branch: `feat/milvus-client-2-6-schema-coverage`

## Scope

Validated the new `milvus_client` 2.6 upgrade/rollback test bricks on 4am Argo with 3 rollback-safe schema collections, 5000 seed rows per collection, continuous pressure traffic, and integrity checks before upgrade, after upgrade, and after rollback.

Images:
- Base: `harbor.milvus.io/milvusdb/milvus:v2.6.18`
- Target: `harbor.milvus.io/milvusdb/milvus:2.6-20260707-682ac8df`
- Runner: `harbor.milvus.io/qa/fouram:2.1`

## PR Review Rounds

Round 1 findings fixed:
- Replaced hardcoded `partition_key` data generation with `field.is_partition_key`.
- Added schema validation for partition key type (`INT64` or `VARCHAR`) and `num_partitions` usage.
- Switched BM25 `Function` creation to PyMilvus keyword arguments.
- Made explicit partition creation fail loudly if the client lacks `create_partition`.

Round 2 findings fixed:
- Fixed checksum stability for `auto_id` schemas by sorting checksum rows with the original primary field even when the primary field is not part of the digested fields.

Round 3 verification:
- `PYTHONPATH=. pytest milvus_client/tests -v`: 63 passed before Argo runs, 64 passed after pressure retry/check additions.
- `argo lint milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml milvus-bricks/argo/upgrade-rollback-compatibility.yaml`: passed.
- `git diff --check`: passed.
- GitHub review threads from Gemini were resolved after fixes.

## Argo Validation Rounds

| Round | Workflow | Result | Duration | Key finding |
| --- | --- | --- | --- | --- |
| 1 | `milvus-standalone-2-6-upgrade-rollback-q24pc` | Failed | 4m02s | Baseline validation failed on `FLOAT` checksum precision in `scalar_dynamic_partition_key`. |
| 2 | `milvus-standalone-2-6-upgrade-rollback-vqr4p` | Succeeded | 11m36s | Data compatibility passed, but `pressure-daemon` stopped after 26s, before the full upgrade/rollback window. |
| 3 | `milvus-standalone-2-6-upgrade-rollback-wlwlh` | Succeeded | 10m43s | Pressure was kept as dependency, but Argo daemon failure was still not gating workflow status. |
| Supplemental | `milvus-standalone-2-6-upgrade-rollback-scxgf` | Succeeded | 11m54s | New stop/check pressure structure worked; pressure summary was produced and data integrity passed after rollback. |

## Fixes From Test Retrospectives

1. Float checksum normalization
   - Problem: Milvus `FLOAT` round-trip produced float32 values such as `16.2 -> 16.200000762939453`; 6-digit rounding could still drift to `16.200001`.
   - Fix: normalize checksum floats to 5 decimal places.
   - Evidence: rerunning validator against the failed 5000-row debug data passed.

2. Pressure daemon lifetime
   - Problem: a DAG daemon is marked ready/succeeded and can stop before downstream observe/rollback steps unless the workflow keeps it in the active dependency chain.
   - Fix: made upgrade, observe, rollback, validation, and stop-pressure steps directly depend on `pressure-daemon`.

3. Pressure failure visibility
   - Problem: pressure pod could exit with Error while Argo workflow still succeeded because daemon readiness had already unblocked downstream tasks.
   - Fix: pressure results are written to the shared PVC, the daemon keeps running until `pressure-stop`, and `check-pressure-results` summarizes failures.

4. Transient startup resilience
   - Problem: pressure bricks can start exactly while standalone is restarting and fail client creation immediately.
   - Fix: added pressure client startup retry (`--startup-retry-sec`, default 30s).

## Final Supplemental Validation Details

Workflow: `milvus-standalone-2-6-upgrade-rollback-scxgf`

Parameters:
- `rows-per-collection=5000`
- `collection-prefix=qa_upgrade_r4_223440`
- `pressure-max-workers=4`
- `pressure-slice-duration-sec=10`
- `pressure-batch-size=10`
- `pressure-fail-on-error=false` for standalone expected restart-window request failures.

Integrity results:
- Before upgrade: passed.
- After upgrade: passed.
- After rollback: passed.
- Final rollback metrics showed all three checkpoint collections at 5000 rows with matching checksums:
  - `scalar_dynamic_partition_key`
  - `vector_autoid_bm25`
  - `explicit_partitions_nullable`

Pressure summary:
- Total pressure result files: 27
- Passed: 24
- Failed/warning: 3
- Failed/warning causes:
  - `count_pressure_4`: Milvus connection unavailable during upgrade window.
  - `query_pressure_16`: Milvus connection unavailable during rollback/transition window.
  - `query_iterator_scan_17`: 4 failed query iterator operations during transition.

The final pressure slices after rollback recovered to zero request failures. Example final successful slices:
- `mixed_rw_pressure`: insert/query/search/count/upsert all passed with `requests_failed=0`.
- `count_pressure`: 14349 count operations, `requests_failed=0`.
- `query_pressure`: 8912 query operations, `requests_failed=0`.
- `query_iterator_scan`: 2040 scanned rows, `requests_failed=0`.

## Coverage

Schemas:
- `scalar_dynamic_partition_key`: dynamic field, partition key, scalar indexes, JSON, ARRAY, nullable scalar, `FLOAT_VECTOR`.
- `vector_autoid_bm25`: auto_id, BM25 function, multi-vector fields, all 2.6 vector types covered by the matrix.
- `explicit_partitions_nullable`: explicit partitions, VARCHAR primary key, nullable scalar fields.

Data types covered:
- Scalar: `BOOL`, `INT8`, `INT16`, `INT32`, `INT64`, `FLOAT`, `DOUBLE`, `VARCHAR`, `JSON`, `ARRAY`.
- Vector: `FLOAT_VECTOR`, `FLOAT16_VECTOR`, `BFLOAT16_VECTOR`, `INT8_VECTOR`, `BINARY_VECTOR`, `SPARSE_FLOAT_VECTOR`.

Index coverage:
- Vector: `HNSW`, `IVF_RABITQ`, `DISKANN`, `AUTOINDEX`, `BIN_IVF_FLAT`, `SPARSE_INVERTED_INDEX`.
- Scalar: `STL_SORT`, `INVERTED`, `BITMAP`, `TRIE`, `NGRAM`, JSON path inverted index, ARRAY inverted index.

Request bricks covered:
- `precheck`
- `create_schema_matrix`
- `seed_data`
- `validate_data_integrity`
- `search_pressure`
- `query_pressure`
- `query_iterator_scan`
- `count_pressure`
- `upsert_pressure`
- `delete_pressure`
- `mixed_rw_pressure`

## Follow-up Status

Implemented after this validation:
- Standalone workflow default changed to `pressure-fail-on-error=false`; pressure failures are summarized as warnings for restart-window request interruptions.
- The workflow now generates both `orchestrator_report.json` and `final_report.md`, merging validation results, pressure summary, environment metadata, flow summary, and Kubernetes snapshot paths.
- Pressure attempts are logged separately from result JSON files, missing result files are summarized, and strict pressure gating runs only after the final report is generated.
- `mixed_rw_pressure` now uses the same startup retry behavior as the independent pressure bricks.

Remaining future work:
- Add a separate cluster-mode workflow/template if the target is strict zero-request-failure rolling upgrade validation.
