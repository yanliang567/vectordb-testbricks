# Reload Pressure Maintenance Window Implementation Plan

**Goal:** Prevent intentional collection release/reload validation from producing false pressure-gate failures without weakening steady-state pressure coverage.

**Architecture:** Index and phase validators will record precise UTC maintenance windows around each `release_collection` and `load_collection` pair. The pressure aggregator will merge these brick-emitted windows with workflow rollout windows, exclude only `collection not loaded` Milvus failures that overlap a collection-reload window, and remove those windows from steady-state availability statistics.

**Technical stack:** Python, Pytest, Argo WorkflowTemplate YAML, PyYAML, Argo offline lint.

---

### Task 1: Define the reload-window contract

**Files:**
- Modify: `milvus_client/common/pressure_maintenance.py`
- Test: `milvus_client/tests/test_argo_template.py`

1. Add a context manager that records precise UTC start/end timestamps, duration, kind, label, source, and collection.
2. Add a parser for maintenance windows emitted in brick result metrics.
3. Add fail-closed classification for collection reload windows.
4. Exclude reload windows from steady-state availability while retaining them in overall observations.

### Task 2: Emit windows from validators

**Files:**
- Modify: `milvus_client/requests/validate_index_compatibility.py`
- Modify: `milvus_client/requests/validate_phase_dml_dql.py`
- Test: `milvus_client/tests/test_validate_index_compatibility.py`
- Test: `milvus_client/tests/test_validate_phase_dml_dql.py`

1. Record each index validation release/load pair.
2. Record phase existing/new collection reload pairs.
3. Record rollback checkpoint reload pairs.
4. Preserve windows on both successful and failed reload attempts.

### Task 3: Merge windows in all workflows

**Files:**
- Modify: `argo/standalone-2-6-upgrade-rollback.yaml`
- Modify: `argo/standalone-3-0-upgrade-rollback.yaml`
- Modify: `argo/cluster-upgrade-rollback.yaml`
- Test: `milvus_client/tests/test_argo_template.py`

1. Load completed brick result JSON from the shared results directory.
2. Merge brick windows with workflow-node rollout windows before pressure classification.
3. Keep all three templates byte-for-byte consistent in the embedded aggregation logic.

### Task 4: Verify

1. Run focused regression tests.
2. Run the full `milvus_client` test suite.
3. Run Argo offline lint for all three WorkflowTemplates.
4. Submit the standalone 2.6 -> fixed 3.0 -> 2.6 workflow and confirm reload failures are excluded while all validation gates remain strict.
