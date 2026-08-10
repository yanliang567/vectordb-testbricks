# PR #25 Final Review And Reload Validation Implementation Plan

**Goal:** Close the remaining false-green paths in PR #25 and require every
upgrade/rollback index validation to prove that loaded collections still serve
query/search after an explicit release and reload cycle.

**Architecture:** Keep lifecycle validation in the existing bricks. Index
compatibility owns the release/load cycle and repeats serviceability probes.
Schema evolution writes an immutable after-upgrade checkpoint and uses a
read-only after-rollback mode. Registered gate scenarios are resolved from the
manifest at runtime and protected parameters are compared before deployment.

**Tech stack:** Python 3.11, PyMilvus 3.0.1, pytest, Argo WorkflowTemplate YAML.

**Status:** Implemented on PR #25. The final review scope also includes schema
evolution vector/null-state oracles, real AutoID evolution rows, complete TEXT
postings counts, and a reserved Entity TTL PK namespace. Final local unit tests:
`383 passed`; static and Argo validation results are recorded in the execution
reports.

---

### Task 1: Add regression tests for fail-closed validator behavior

**Files:**
- Modify: `milvus_client/tests/test_validate_index_compatibility.py`
- Modify: `milvus_client/tests/test_validate_phase_dml_dql.py`
- Modify: `milvus_client/tests/test_feature_validators.py`
- Modify: `milvus_client/tests/test_schema_evolution_workload.py`
- Modify: `milvus_client/tests/test_data_generation.py`
- Modify: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

**Steps:**
1. Reproduce rollback checkpoint overwrite and require the after-upgrade oracle
   to remain byte-for-byte unchanged.
2. Reproduce `NaN` scores and missing BM25 scores in both index and phase
   search validators.
3. Reproduce schema evolution passing with empty query/search and zero evolved
   rows.
4. Require MinHash exact rank and deterministic TEXT/BM25 result checks.
5. Require a real changed-field upsert projection for function-only and
   StructArray-only schemas.
6. Require StructArray payloads in default checksum fields.
7. Reject registered scenario parameter drift.
8. Require query/search calls both before and after release/load.

### Task 2: Harden Python validation and checkpoints

**Files:**
- Modify: `milvus_client/common/data.py`
- Modify: `milvus_client/common/schema.py`
- Modify: `milvus_client/common/feature_validators.py`
- Modify: `milvus_client/common/gates.py`
- Modify: `milvus_client/requests/schema_evolution_workload.py`
- Modify: `milvus_client/requests/validate_index_compatibility.py`
- Modify: `milvus_client/requests/validate_phase_dml_dql.py`

**Steps:**
1. Add finite metric validation and metric-specific score/distance bounds.
2. Preserve index compatibility checkpoints after upgrade; write no rollback
   observation over the oracle.
3. Repeat count/PK, vector search, and scalar-index queries after strict
   release/load.
4. Validate evolved PK-range count, evolved field values, StructArray payloads,
   and indexed vector searches; write a versioned checkpoint after upgrade.
5. Add read-only rollback validation for the schema-evolution checkpoint.
6. Generate and validate an explicit changed-field upsert projection.
7. Include StructArray top-level fields in checksum selection.
8. Strengthen lexical and MinHash feature ground truth.
9. Add registered-scenario protected-parameter comparison.

### Task 3: Wire workflow lifecycle contracts

**Files:**
- Modify: `argo/standalone-2-6-upgrade-rollback.yaml`
- Modify: `argo/standalone-3-0-upgrade-rollback.yaml`
- Modify: `argo/cluster-upgrade-rollback.yaml`
- Modify: `milvus_client/requests/generate_workflow_report.py`
- Modify: `milvus_client/tests/test_argo_template.py`
- Modify: `milvus_client/tests/test_generate_workflow_report.py`

**Steps:**
1. Validate registered scenario protected parameters before deployment.
2. Pass schema-evolution checkpoint paths during after-upgrade execution.
3. Add required after-rollback schema-evolution validation for existing and
   rollback-compatible forward collections.
4. Make the final report require those result files when the corresponding
   feature flags are enabled.
5. Keep target-only 3.0 forward schemas excluded from 2.6 rollback validation.

### Task 4: Verify and publish

**Commands:**

```bash
PYTHONPATH=. pytest -q milvus_client/tests/test_validate_index_compatibility.py
PYTHONPATH=. pytest -q milvus_client/tests/test_validate_phase_dml_dql.py
PYTHONPATH=. pytest -q milvus_client/tests/test_feature_validators.py
PYTHONPATH=. pytest -q milvus_client/tests/test_schema_evolution_workload.py
PYTHONPATH=. pytest -q milvus_client/tests
ruff check <changed-python-files>
ruff format --check <changed-python-files>
argo lint --offline argo
git diff --check
```

Commit and push only after all checks pass. Do not merge PR #25.
