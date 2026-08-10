# Milvus Upgrade/Rollback Type and Index Coverage Implementation Report

Date: 2026-08-07

## Scope

This change implements the supplementary coverage plan for Milvus `2.6.18 -> 3.0 -> 2.6` and `3.0.0 -> latest 3.0 -> 3.0.0` upgrade/rollback paths. The implementation adds deterministic data generation, schema/index metadata validation, runtime feature semantics checks, promoted gate scenarios, and WorkflowTemplate integration.

The work was implemented on:

- Branch: `feat/upgrade-rollback-type-index-coverage`
- Base commit after PR #24 merge: `bcafe367b80558fff064313e372b7bee8c9e0188`
- Pull request: `yanliang567/vectordb-testbricks#25`
- Initial PR commit: `ea45180ac4e828654d805d528a01bc21c21fd700`
- Initial CI run: `31231458563` (`295 passed, 2 deselected`)
- Review fixes: included in the PR after independent code review and local
  regression verification

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

- StructArray scalar round trip and element-level vector search; ordinary
  element search validates exact `id + offset`, while `MAX_SIM_*` uses an
  `EmbeddingList` row-level query and validates the expected PK
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
- StructArray FLOAT_VECTOR with MAX_SIM_COSINE row-level `EmbeddingList`
  validation
- StructArray FLOAT16_VECTOR with DISKANN
- FAISS factories for dense and binary vectors
- MinHash function output with strict exact self-search plus observational
  near-duplicate recall; ranking is checked only when both near-duplicate and
  unrelated rows are returned
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
- JSON AutoIndex with expected internal resolved type `HYBRID`
- Runtime pod target-configuration validation for
  `targetVecIndexVersion=10` and `targetScalarIndexVersion=4`

Milvus resolves target index versions against current/min/max engine versions.
The public SDK index metadata does not expose the exact version used by each
build. It also reports a scalar AutoIndex through the user-facing
`AUTOINDEX` type even when the internal index params resolve it to `HYBRID`.
When the matrix declares `expected_resolved_index_type`, the gate now requires
the public metadata to expose that resolved type and fails closed when it is
unavailable. It also validates the target configuration plus executable
SINDI/Block-Max and JSON index behavior. Real-cluster execution preserves
MixCoord/DataNode logs containing the internal `HYBRID` params,
`currentIndexVersion`, and `currentScalarIndexVersion` as supplementary evidence
for engine-version resolution; those logs do not turn unavailable required SDK
metadata into a passing gate.

### Gate Scenarios and Workflows

The gate manifest now contains 11 promoted gate scenarios, two pre-release
candidate scenarios, and one negative scenario. Added promoted scenarios:

- `standalone-3-0-index-v10-v4-upgrade-rollback`
- `cluster-3-0-index-v10-v4-upgrade-rollback`

The standalone and cluster Loon/Vortex scenarios now create the Storage V3 TEXT collection after the target rollout and validate it again after rollback.

The standalone and cluster 3.0 core scenarios also create the complete 3.0
matrix on the target image. This makes target-built scalar/vector index files a
first-class rollback input instead of relying on incidental pressure flushes or
explicit index rebuilds on the baseline collection.

All three upgrade/rollback WorkflowTemplates now execute feature semantics at the required lifecycle points:

- Base data before upgrade
- Existing data after upgrade
- Forward data after upgrade
- Existing data after rollback
- Forward data after rollback when enabled and compatible

The final workflow report treats these feature result files as required inputs, so missing or failed feature checks cannot produce a successful gate report.

### Review Hardening

PR review added the following fail-closed protections:

- Matrix index metadata is compared with actual index type, metric, explicit
  name, and compatibility-critical parameters before checkpointing.
- Vector search failures persist the returned PK, element offset, and score for
  every hit so server-side ranking/sign regressions remain diagnosable after
  workflow cleanup.
- Index search probes require the expected PK and an observable score/distance;
  BM25 function output indexes no longer pass on an unrelated non-empty hit.
- Index compatibility validates each collection while loaded, then performs a
  strict release/load cycle and repeats count/PK, vector search, and scalar
  index queries after both upgrade and rollback.
- Schema evolution validates deterministic evolved rows and searches, writes an
  after-upgrade checkpoint, and uses a read-only checkpoint validation mode
  after rollback. Top-level vector content, nullable vector state, StructArray
  payloads, and metric-specific self-match thresholds are part of this oracle.
- AutoID schema evolution inserts real rows, requires one unique returned PK
  per row, and validates the actual PK mapping through rollback query/search.
- TEXT_MATCH and PHRASE_MATCH compare `count(*)` with deterministic complete
  postings ground truth, then require the sample query to return its complete
  requested limit before validating PK and text content.
- Entity TTL probes use a reserved PK namespace outside continuous pressure
  insert/upsert/delete ranges.
- Registered scenarios reject WorkflowTemplate, schema matrix, index target,
  validator flag, storage feature, pressure policy, or data-scale drift before
  Milvus deployment.
- Missing matrix collections and validators with no applicable runtime target
  fail instead of reporting a zero-work pass.
- Required workflow validations must report `passed`; `skipped` is accepted
  only for optional validations.
- StructArray-only vector collections participate in rollout pressure, and
  nullable vector probes select a deterministic non-null row.
- Storage V3 capability activation, feature-to-brick catalog mappings, cluster
  post-config index target propagation, and CI Ruff scope are covered by
  regression tests.

## PyMilvus Version

The test runtime and CI dependency were changed from `pymilvus==3.0.0` to `pymilvus==3.0.1`.

This is required because PyMilvus 3.0.0 does not expose `DataType.TEXT`, while the Storage V3 matrix must construct and validate real TEXT schemas. PyMilvus 3.0.1 successfully constructs all 19 schema/index definitions used by the five matrices.

## Verification Results

Verification used a temporary Python 3.9 virtual environment with `pymilvus==3.0.1`, `pytest==8.4.2`, `PyYAML`, `numpy`, and `ruff==0.15.22`.

### Focused New Validator Tests

```bash
cd milvus-bricks
PYTHONPATH=. /tmp/vectordb-testbricks-pymilvus301-py39/bin/python -m pytest -q \
  milvus_client/tests/test_workload.py \
  milvus_client/tests/test_mixed_rw_pressure.py \
  milvus_client/tests/test_argo_template.py \
  milvus_client/tests/test_validate_index_compatibility.py \
  milvus_client/tests/test_validate_phase_dml_dql.py \
  milvus_client/tests/test_feature_validators.py \
  milvus_client/tests/test_validate_schema_features.py \
  milvus_client/tests/test_generate_workflow_report.py \
  milvus_client/tests/test_capability.py \
  milvus_client/tests/test_brick_catalog.py
```

Result: `160 passed, 1 warning`.

### Full Offline Unit Test Set

```bash
cd milvus-bricks
PYTHONPATH=. /tmp/vectordb-testbricks-pymilvus301-py39/bin/python -m pytest \
  milvus_client/tests -q \
  -k 'not test_rendered_cluster_helm_values_apply_metadata_to_chart_resources and not test_rendered_pulsar_broker_role_is_covered_by_workflow_manager_rbac'
```

Result: `311 passed, 2 deselected, 1 warning`.

The warning is the local Python 3.9 LibreSSL warning emitted by urllib3 and is unrelated to the implementation.

### Full Test Set Including Online Helm Tests

```bash
cd milvus-bricks
PYTHONPATH=. /tmp/vectordb-testbricks-pymilvus301-py39/bin/python -m pytest milvus_client/tests -q
```

One full run completed with `304 passed, 1 failed, 1 warning`. A second run
excluding that failed test reached the other online Helm test and completed
with `303 passed, 1 failed, 1 deselected, 1 warning`. Both online tests were
individually observed failing while running:

```text
helm repo add zilliztech https://zilliztech.github.io/milvus-helm/
```

The GitHub Pages connections were reset. These were external repository-access
errors before chart rendering, not assertion or implementation failures.

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
/tmp/vectordb-testbricks-pymilvus301-py39/bin/ruff check <23 affected Python files>
/tmp/vectordb-testbricks-pymilvus301-py39/bin/ruff format --check <23 affected Python files>
```

Result:

```text
All checks passed!
23 files already formatted
```

```bash
argo lint --offline argo
```

Result: `no linting errors found`.

```bash
git diff --check
```

Result: passed with no whitespace errors.

Whole-repository Ruff was not used as a completion gate because legacy scripts currently contain unrelated pre-existing Ruff violations. All changed Python files are clean.

### 2026-08-10 Final Review Verification

The final review hardening, including release/reload validation, schema
evolution rollback checkpoints, and explicit pre-release candidate eligibility
metadata, completed with:

```text
PYTHONPATH=. pytest -q milvus_client/tests: 415 passed
argo lint --offline argo: passed
ruff check on changed Python files: passed
ruff format --check on changed Python files: passed
git diff --check: passed
```

## Live Environment Validation

Live Milvus, Kubernetes, Helm, and Argo validation was completed on the QA
cluster after the implementation pass. The complete workflow matrix, metrics,
logs, blocker analysis, cleanup verification, and acceptance status are in:

[2026-08-09-upgrade-rollback-type-index-coverage-execution.md](2026-08-09-upgrade-rollback-type-index-coverage-execution.md)

Confirmed passing behavior:

- A 5000-row `2.6.18 -> newer 3.0 -> latest 2.6` run passed with the current
  PR commit and pinned image digests.
- The 2.6 rollback-safe StructArray, all six nullable vector types, Geometry,
  and 47 index definitions remained queryable after rollback without rebuild.
- StructArray nested FLOAT/VARCHAR/INT64/BOOL scalar indexes executed their
  target-side filter probes.
- Standalone JSON Shredding passed nested JSON paths, dynamic JSON checksums,
  six indexes, full DML pressure, and rollback reuse at 5000 rows.
- Cluster JSON Shredding passed the same format/index contract under read-only
  continuous pressure plus explicit phase insert/upsert/delete validation.
- Index engine versions `10/4` were confirmed through real DataNode/MixCoord
  5000-row build logs, including SINDI, Block-Max, JSON scalar indexes, and
  internal AutoIndex resolution to HYBRID.
- FAISS, MinHash, Entity TTL, Geometry, nullable vectors, and the common 3.0
  nested scalar index probes executed successfully outside the product defects
  listed below.

Confirmed Milvus blockers and support boundaries:

- StructArray FLOAT16 DISKANN returns a negative MAX_SIM_COSINE exact
  self-score.
- SINDI growing-index construction can crash QueryNode after selecting an
  unsupported effective index version.
- v3.0.0 cannot decode `vortex.variant` files written by the newer 3.0 target.
  Product follow-up in [#52340](https://github.com/milvus-io/milvus/issues/52340)
  established that v3.0.0 should not enable Vortex; supported rollback coverage
  starts at v3.0.1, with reviewed Vortex 0.75 branch builds used only as
  pre-release candidates.
- Continuous cluster DML across the Woodpecker rollout/rollback can lose reader
  temporary state and permanently stall selected channel tSafe values.

Each item has a reproduction and issue draft in this report directory. Active
blockers remain strict failures; the v3.0.0 Vortex result is preserved as an
unsupported-path boundary rather than converted to a warning.

The final 2026-08-10 offline unit test set completed with `415 passed`.
