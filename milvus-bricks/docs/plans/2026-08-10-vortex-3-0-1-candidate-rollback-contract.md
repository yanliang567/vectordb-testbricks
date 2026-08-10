# Vortex 3.0.1 Candidate Rollback Contract Implementation Plan

**Goal:** Remove the unsupported v3.0.0 Vortex rollback path from promoted
gates, enforce the v3.0.1 compatibility boundary, and retain pre-release
coverage with pinned 3.0 branch candidate images that contain Vortex 0.75.

**Architecture:** Keep the ordinary v3.0.0 baseline for non-Vortex gates. Add a
separate candidate classification and immutable candidate image aliases with
reviewed storage compatibility metadata. Gate validation and all three Argo
templates fail closed for Vortex on versions below v3.0.1 unless the registered
scenario is an approved, image-locked pre-release candidate.

**Tech stack:** Python 3.11, PyYAML, pytest, Argo WorkflowTemplate YAML,
Milvus Operator, Harbor image digests.

---

### Task 1: Add version and candidate image contracts

**Files:**
- Modify: `milvus_client/common/version.py`
- Modify: `milvus_client/common/gates.py`
- Test: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

**Steps:**
1. Add numeric major/minor/patch parsing and `version_at_least`.
2. Treat `candidate` scenarios as immutable, strict lifecycle scenarios.
3. Require Vortex phases in release gates to use Milvus `>=3.0.1`.
4. Allow a pre-release candidate only when its alias is immutable, declares
   reviewed Vortex compatibility metadata, and runtime image/version overrides
   do not replace the reviewed alias.
5. Require rollback compatibility whenever target-phase Vortex data may exist,
   even if the rollback phase attempts to disable the Vortex config flag.

### Task 2: Replace unsupported v3.0.0 Vortex gates

**Files:**
- Modify: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- Test: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- Test: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

**Steps:**
1. Keep `milvus-3-0-baseline` pinned to v3.0.0 for non-Vortex gates.
2. Add reviewed candidate aliases:
   - baseline/rollback: `3.0-20260807-697431f2` at its manifest-list digest;
   - target: `3.0-20260807-1439dc7d` at its manifest-list digest.
3. Record full Milvus source commits and `milvus-storage 63c29c6` metadata.
4. Replace the two supported v3.0.0 Loon/Vortex gates with standalone and
   cluster `candidate` scenarios using the pinned aliases.
5. Keep strict data, serviceability, pressure, forward-schema, index, and
   release/reload validation. Do not count candidate outcomes as release gates.

### Task 3: Add direct-submission runtime guards

**Files:**
- Modify: `argo/standalone-2-6-upgrade-rollback.yaml`
- Modify: `argo/standalone-3-0-upgrade-rollback.yaml`
- Modify: `argo/cluster-upgrade-rollback.yaml`
- Test: `milvus_client/tests/test_argo_template.py`

**Steps:**
1. Parse phase versions as numeric major/minor/patch values in `resolve-inputs`.
2. Reject Vortex on versions below v3.0.1 before deployment.
3. Reject target Vortex data followed by rollback below v3.0.1.
4. Allow only the registered standalone/cluster candidate scenario IDs to use
   the pre-release exception; registered-scenario resolution must still prove
   exact candidate image identity.
5. Keep the existing explicit unsupported 2.6 negative coverage exception.

### Task 4: Update documentation and historical disposition

**Files:**
- Modify: `docs/upgrade-rollback-gates/README.md`
- Modify: `milvus_client/README.md`
- Modify: `milvus_client/docs/upgrade-rollback.md`
- Modify: `milvus_client/docs/reports/2026-08-09-upgrade-rollback-type-index-coverage-execution.md`
- Modify: `milvus_client/docs/reports/2026-08-09-upgrade-rollback-type-index-coverage-execution-zh.md`
- Modify: `milvus_client/docs/reports/2026-08-09-milvus-vortex-variant-rollback-incompatibility-issue-draft.md`

**Steps:**
1. Link Milvus issue #52340 and document that v3.0.0 Vortex is unsupported.
2. Preserve the historical failed executions as evidence, but remove them from
   the supported release-gate denominator.
3. Mark candidate runs as pre-release evidence, not v3.0.1 release acceptance.
4. Document that the candidate aliases must be refreshed through code review;
   runtime overrides are intentionally rejected.
5. Define the release transition: replace candidate aliases with a pinned
   v3.0.1 digest, change classification to `gate`, remove the candidate
   exception, and rerun standalone plus cluster workflows.

### Task 5: Verify

**Commands:**

```bash
PYTHONPATH=. pytest -q milvus_client/tests
ruff check <changed-python-files>
ruff format --check <changed-python-files>
argo lint --offline argo
git diff --check
```

After offline verification, submit a PR. Real candidate workflow execution is
a separate acceptance step and must record concrete workflow names, repository
commit, image digests, server versions, storage config, Vortex segment-load
logs, and post-rollback count/query/search/release-load results.
