# Milvus Upgrade/Rollback Type and Index Coverage Implementation Report

Date: 2026-08-07

## Scope

This change implements the supplementary coverage plan for Milvus `2.6.18 -> 3.0 -> 2.6` and `3.0.0 -> latest 3.0 -> 3.0.0` upgrade/rollback paths. The implementation adds deterministic data generation, schema/index metadata validation, runtime feature semantics checks, promoted gate scenarios, and WorkflowTemplate integration.

The work was implemented on:

- Branch: `feat/upgrade-rollback-type-index-coverage`
- Base commit after PR #24 merge: `bcafe367b80558fff064313e372b7bee8c9e0188`
- Working tree state: uncommitted; no commit or push was performed

The merged local branch `fix/tsafe-serviceability-retry` was removed before implementation.

## Implemented Coverage

### Schema and Data DSL

The schema matrix DSL now supports:

- StructArray declarations and qualified nested fields such as `attributes[score_sort]`
- StructArray scalar and vector sub-fields, nullable arrays, and deterministic element offsets
- Collection `properties` and validator-specific parameters
- Explicit index names, search parameters, and expected resolved AutoIndex type
- TEXT value profiles, MinHash document profiles, TIMESTAMPTZ values, and collection-level Entity TTL
- Fail-closed validation for unsupported Milvus 2.6 StructArray combinations and unknown validators

Deterministic data generation now covers StructArray scalar/vector elements, nullable vectors, TEXT LOB boundaries, MinHash documents, Geometry values, and TIMESTAMPTZ values.

### Milvus 2.6 Rollback-Safe Matrix

`schema_matrix_2_6.yaml` now contains seven schemas. The added rollback-safe coverage includes:

- StructArray scalar round trip and element-level vector search with exact `id + offset` validation
- All six nullable vector types: FLOAT, FLOAT16, BFLOAT16, INT8, BINARY, and SPARSE
- Geometry with RTREE, `ST_EQUALS`, and `ST_DWITHIN`
- Explicit legacy index families: FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, SCANN, HNSW_SQ, BIN_FLAT, and SPARSE_WAND

These schemas are consumed by the existing standalone and cluster `2.6 -> 3.0 -> 2.6` gate paths.

### Milvus 3.0 Matrix

`schema_matrix_3_0.yaml` now contains eight schemas. New coverage includes:

- StructArray nested scalar indexes:
  - FLOAT + STL_SORT
  - FLOAT + INVERTED
  - VARCHAR + INVERTED
  - VARCHAR + BITMAP
  - INT64 + STL_SORT
  - BOOL + BITMAP
- StructArray FLOAT_VECTOR with MAX_SIM_COSINE and exact element offset validation
- StructArray FLOAT16_VECTOR with DISKANN
- FAISS factories for dense and binary vectors
- MinHash function output with MINHASH_LSH and MHJACCARD ranking validation
- TIMESTAMPTZ Entity TTL with expired, future, and NULL visibility checks

Each nested scalar index is exercised with a real `MATCH_ANY` filter in addition to index metadata comparison.

### Storage V3 and Index Engine Matrices

Added `schema_matrix_3_0_storage_v3.yaml` for Loon/Vortex gates:

- TEXT NULL, empty, Unicode, 64 KiB boundary, over-64 KiB, and 1 MiB payloads
- Payload byte/character/hash verification without writing full LOB values to reports
- TEXT_MATCH, PHRASE_MATCH, and BM25 search

Added `schema_matrix_3_0_index_v10_v4.yaml`:

- Sparse SINDI, BLOCK_MAX_MAXSCORE, and BLOCK_MAX_WAND algorithms
- JSON BOOL BITMAP, DOUBLE STL_SORT, and VARCHAR NGRAM path indexes
- JSON AutoIndex with required resolved type `HYBRID`
- Runtime pod configuration validation for `targetVecIndexVersion=10` and `targetScalarIndexVersion=4`

### Gate Scenarios and Workflows

The gate manifest now contains 13 promoted scenarios and one negative scenario. Added promoted scenarios:

- `standalone-3-0-index-v10-v4-upgrade-rollback`
- `cluster-3-0-index-v10-v4-upgrade-rollback`

The standalone and cluster Loon/Vortex scenarios now create the Storage V3 TEXT collection after the target rollout and validate it again after rollback.

All three upgrade/rollback WorkflowTemplates now execute feature semantics at the required lifecycle points:

- Base data before upgrade
- Existing data after upgrade
- Forward data after upgrade
- Existing data after rollback
- Forward data after rollback when enabled and compatible

The final workflow report treats these feature result files as required inputs, so missing or failed feature checks cannot produce a successful gate report.

## PyMilvus Version

The test runtime and CI dependency were changed from `pymilvus==3.0.0` to `pymilvus==3.0.1`.

This is required because PyMilvus 3.0.0 does not expose `DataType.TEXT`, while the Storage V3 matrix must construct and validate real TEXT schemas. PyMilvus 3.0.1 successfully constructs all 19 schema/index definitions used by the five matrices.

## Verification Results

Verification used a temporary Python 3.9 virtual environment with `pymilvus==3.0.1`, `pytest==8.4.2`, `PyYAML`, `numpy`, and `ruff==0.15.22`.

### Focused New Validator Tests

```bash
cd milvus-bricks
PYTHONPATH=. /tmp/vectordb-testbricks-pymilvus301-py39/bin/python -m pytest -q \
  milvus_client/tests/test_validate_index_compatibility.py \
  milvus_client/tests/test_feature_validators.py \
  milvus_client/tests/test_validate_schema_features.py
```

Result: `28 passed, 1 warning`.

### Full Offline Unit Test Set

```bash
cd milvus-bricks
PYTHONPATH=. /tmp/vectordb-testbricks-pymilvus301-py39/bin/python -m pytest \
  milvus_client/tests -q \
  -k 'not test_rendered_cluster_helm_values_apply_metadata_to_chart_resources and not test_rendered_pulsar_broker_role_is_covered_by_workflow_manager_rbac'
```

Result: `295 passed, 2 deselected, 1 warning`.

The warning is the local Python 3.9 LibreSSL warning emitted by urllib3 and is unrelated to the implementation.

### Full Test Set Including Online Helm Tests

```bash
cd milvus-bricks
PYTHONPATH=. /tmp/vectordb-testbricks-pymilvus301-py39/bin/python -m pytest milvus_client/tests -q
```

Result: `295 passed, 2 failed, 1 warning`.

Both failures occurred in existing tests while running:

```text
helm repo add zilliztech https://zilliztech.github.io/milvus-helm/
```

The connection to GitHub Pages was reset. The failures were external network errors and were not assertion or implementation failures.

### Schema and Index Construction

The five matrices were loaded, validated against the feature/capability catalogs, and constructed with PyMilvus 3.0.1:

```text
schema_matrix_2_6.yaml: schemas=7
schema_matrix_3_0.yaml: schemas=8
schema_matrix_3_0_storage_v3.yaml: schemas=1
schema_matrix_3_0_index_v10_v4.yaml: schemas=2
schema_matrix_json_shredding.yaml: schemas=1
rendered_schemas=19
```

### Static and Workflow Validation

```bash
cd milvus-bricks
/tmp/vectordb-testbricks-pymilvus301-py39/bin/ruff check <21 affected Python files>
/tmp/vectordb-testbricks-pymilvus301-py39/bin/ruff format --check <21 affected Python files>
```

Result:

```text
All checks passed!
21 files already formatted
```

```bash
argo lint --offline milvus-bricks/argo
```

Result: `no linting errors found`.

```bash
git diff --check
```

Result: passed with no whitespace errors.

Whole-repository Ruff was not used as a completion gate because legacy scripts currently contain unrelated pre-existing Ruff violations. All changed Python files are clean.

## Remaining Environment Validation

No live Milvus, Kubernetes, Milvus Operator, Helm deployment, or Argo Workflow execution was performed in this implementation pass. The following behavior still requires execution on the QA cluster:

- Milvus 2.6.18 StructArray and six nullable-vector schemas can be created and reused after `2.6 -> 3.0 -> 2.6`
- StructArray nested scalar indexes execute successfully on Milvus 3.0 and remain usable after rollback to the pinned 3.0 baseline
- TEXT LOB values survive the Loon/Vortex upgrade/rollback path
- FAISS, MinHash, Entity TTL, SINDI/Block-Max, and JSON index engine v4 behave as expected on the target images
- Runtime pods expose and honor vector index version 10 and scalar index version 4
- The two Helm rendering integration tests pass when `https://zilliztech.github.io/milvus-helm/` is reachable

These are environment-level acceptance checks; the local schema construction, contract tests, workflow lint, and report fail-closed behavior are implemented and verified.
