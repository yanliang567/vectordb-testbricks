# Upgrade/Rollback Compatibility Scenario

The upgrade/rollback scenario validates rollback-safe Milvus data while cluster
lifecycle actions are owned by Argo, Helm, or an external operator.

## Flow

1. Run `precheck`.
2. Create rollback-safe compat collections from `schema_matrix_2_6.yaml`.
3. Seed deterministic compat data and write a checkpoint.
4. Validate compat data before upgrade.
5. Start background mixed read/write pressure on compat collections.
6. Start a background validator loop against the compat checkpoint.
7. For each cycle:
   - wait for the external upgrade signal;
   - observe the upgraded cluster;
   - validate compat data;
   - create, seed, and validate forward-only collections from `schema_matrix_3_0.yaml`;
   - wait for the external rollback signal;
   - observe the rolled-back cluster;
   - validate compat data only.
8. Stop background loops and run final compat validation.

Forward-only schema checkpoints are written under per-cycle checkpoint
directories, so they do not overwrite the rollback-safe compat checkpoint.

## Local Dry Run

```bash
PYTHONPATH=. python -m milvus_client.scenarios.upgrade_rollback_compatibility \
  --dry-run \
  --scenario-manifest milvus_client/manifests/scenario_upgrade_rollback.yaml \
  --uri http://localhost:19530 \
  --collection-prefix qa_upgrade \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/upgrade_rollback_dry_run.json
```

## Closed-Loop Execution

```bash
PYTHONPATH=. python -m milvus_client.scenarios.upgrade_rollback_compatibility \
  --scenario-manifest milvus_client/manifests/scenario_upgrade_rollback.yaml \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_upgrade \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/upgrade_rollback.json
```

When `actions.upgrade.wait_file` or `actions.rollback.wait_file` is empty, the
scenario continues immediately. Otherwise it waits until the configured file
exists.

## Argo

`argo/upgrade-rollback-compatibility.yaml` runs the scenario controller in one
closed-loop pod and exports:

- the scenario result JSON;
- all per-brick result JSON files;
- checkpoint state.

The template keeps the generic `brick-runner` for future direct composition of
independent bricks, but the Phase 2 upgrade/rollback loop uses
`scenario-runner` so background pressure and validator loops remain alive across
upgrade and rollback waits.

## 4am Standalone Upgrade/Rollback

`argo/standalone-2-6-upgrade-rollback.yaml` is the 4am workflow template for
the v2.6.18 rollback compatibility lane. Client pods run in `qa` with the
`milvus-upgrade-rollback-runner` ServiceAccount, while the Milvus Operator CR is
created in `qa-milvus`. The template creates a standalone CR with parameterized
2 CPU / 4 GiB requests and 4 CPU / 8 GiB limits, seeds only rollback-safe 2.6
schemas, upgrades to a configured 2.6 or master/3.0 target image, validates
existing data, rolls back to v2.6.18, and validates the same checkpoint again.
Client pods default to 1 CPU / 2 GiB requests and 2 CPU / 4 GiB limits.
Set `rollback-milvus-image` and `rollback-version` explicitly when a scenario
rolls back to a different 2.6 branch image, such as latest 2.6 instead of the
original v2.6.18 image.

By default this template does not create `schema_matrix_3_0.yaml` data. New 3.0
schema/data is upgrade-only when rolling back to 2.6 and is not expected to
survive that rollback. Set `forward-workload-enabled=true` only when the target
phase should exercise forward-only 3.0 schema; keep
`rollback-forward-validation-enabled=false` for 2.6 rollback runs.

`argo/standalone-3-0-upgrade-rollback.yaml` is the matching 3.0 baseline lane.
It defaults to `schema_matrix_3_0.yaml`, starts from
`3.0-20260701-d19d8484-47f6c14`, upgrades to a concrete master image tag, and
rolls back to the 3.0 baseline image. In this lane, 3.0 schema/data is the hard
compatibility gate across all phases.

Both templates expose the same configuration matrix parameters:

- `base-json-shredding-enabled`
- `target-json-shredding-enabled`
- `rollback-json-shredding-enabled`
- `target-loon-ffi-enabled`
- `post-upgrade-config-toggle-enabled`
- `post-upgrade-json-shredding-enabled`
- `forward-workload-enabled`
- `forward-schema-matrix`
- `forward-collection-prefix`
- `rollback-enabled`
- `rollback-version`
- `rollback-forward-validation-enabled`
- `observe-before-upgrade-sec`
- `observe-after-upgrade-sec`
- `observe-before-rollback-sec`
- `observe-after-rollback-sec`

The workflow writes `spec.config.common.storage.jsonShreddingEnabled` during
base deploy and image patch phases. It writes
`spec.config.common.storage.useLoonFFI` only for target/post-upgrade phases and
forces it back to false for 2.6 rollback. The workflow snapshots Milvus CR
config after base deploy, target upgrade, optional post-upgrade config toggle,
and rollback.

The standalone templates run pressure during fixed observe windows before and
after both upgrade and rollback. These observe windows default to 300 seconds so
client request traffic covers at least five minutes in each steady phase.
`observe-after-upgrade-sec` and `observe-before-rollback-sec` are sequential,
not overlapping. The workflow runs the after-upgrade observation first, then
upgrade-phase precheck/validation, existing schema evolution, optional
post-upgrade config patch, optional forward workload, and only then the
before-rollback observation.

Submit example:

```bash
kubectl apply -f argo/standalone-2-6-upgrade-rollback-rbac.yaml
argo submit -n qa --from workflowtemplate/milvus-standalone-2-6-upgrade-rollback \
  -p repo-revision=main \
  -p base-milvus-image=harbor.milvus.io/milvusdb/milvus:v2.6.18 \
  -p target-milvus-image=harbor.milvus.io/milvusdb/milvus:2.6-20260707-e9ee9a47 \
  -p rollback-milvus-image=harbor.milvus.io/milvusdb/milvus:v2.6.18 \
  -p rollback-version=2.6.18 \
  -p base-json-shredding-enabled=true \
  -p target-json-shredding-enabled=true \
  -p rollback-json-shredding-enabled=true \
  -p keep-milvus=false
```

3.0 baseline submit example:

```bash
kubectl apply -f argo/standalone-2-6-upgrade-rollback-rbac.yaml
argo submit -n qa --from workflowtemplate/milvus-standalone-3-0-upgrade-rollback \
  -p repo-revision=main \
  -p base-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14 \
  -p target-milvus-image=harbor.milvus.io/milvusdb/milvus:master-20260707-5617a46a \
  -p keep-milvus=false
```

When enabling target-only forward workload, set `forward-collection-prefix` to a
different value from `collection-prefix`. Forward create/seed/validate/schema
evolution steps use the forward prefix, while baseline compatibility and
pressure steps continue using `collection-prefix`.
Baseline seed/validate steps use
`/tmp/milvus-bricks/checkpoints/baseline/seed_data.json`; forward-only
seed/validate steps use `/tmp/milvus-bricks/checkpoints/forward/seed_data.json`.
This keeps forward-only checkpoints from overwriting the baseline checkpoint
used by rollback validation.

For `v2.6.18 -> master -> v2.6.18` runs, keep the 2.6 template and pass the
master image as `target-milvus-image`. The schema matrix remains
`schema_matrix_2_6.yaml` unless `forward-workload-enabled=true` is explicitly
set for target-only 3.0 coverage.

For upgrade-only target compatibility runs, such as `v2.6.18 -> master` with
LoonFFI/vortex and a post-upgrade jsonShredding toggle, set
`rollback-enabled=false`. The workflow still keeps strict pressure and
validation gates for the base and upgraded target phases, but it does not patch
back to a 2.6 image or run rollback validations. Keep
`rollback-forward-validation-enabled=false` on these runs because there is no
rollback phase.

`validate_forward_after_rollback` runs only when both `rollback-enabled=true`
and `forward-workload-enabled=true`; it is then controlled by
`rollback-forward-validation-enabled`. The final report requires this result
only for that same rollback-plus-forward combination.

The 4am template runs these strict foreground bricks:

- `precheck`
- `create_schema_matrix`
- `seed_data`
- `validate_data_integrity` before upgrade
- `validate_data_integrity` after upgrade
- `validate_data_integrity` after rollback

It also starts a daemon workload loop after the baseline validation:

- `search_pressure`
- `query_pressure`
- `query_iterator_scan`
- `count_pressure`
- `upsert_pressure`
- `delete_pressure`
- `mixed_rw_pressure`

For standalone upgrades, transient request failures can happen while the only
Milvus process restarts. The daemon loops default to
`pressure-fail-on-error=true`, so pressure failures fail the workflow after the
final report is generated. Override this only for exploratory standalone runs
where restart-window request failures should be recorded as warnings instead of
used as a regression gate. The final gate defaults to
`gate-allow-warning=false`; if an exploratory standalone run intentionally sets
`pressure-fail-on-error=false`, also set `gate-allow-warning=true` to let Argo
finish as succeeded while keeping the final report status as `warning`.

The pressure daemon does not mount the workflow state PVC while foreground
`run-brick` steps are running. It stores pressure attempts/results and the stop
signal in workflow-owned ConfigMaps in the Argo namespace, avoiding concurrent
ReadWriteOnce PVC mounts. After pressure stops, `check-pressure-results`
reconstructs pressure result artifacts and writes the pressure summary into the
workflow state PVC. The daemon receives the seeded row count as a parameter, so
`count_pressure` and the count operation inside `mixed_rw_pressure` can keep
checking row counts while upgrade and rollback are in progress.

The workflow emits `env_snapshot.json`, `flow_summary.json`,
`orchestrator_report.json`, `final_report.md`, foreground brick results,
checkpoints, pressure results, `pressure-summary.json`, and Kubernetes
snapshots. The final report merges validation results, pressure summary, target
metadata, config matrix parameters, and snapshot paths into one comparable
artifact. Strict pressure gating runs after final report generation, so
`pressure-fail-on-error` does not prevent report artifacts from being produced.
The final gate accepts only `passed`; pressure warnings do not pass the
regression workflow by default unless `gate-allow-warning=true` is explicitly
set.
`keep-milvus=false` is the default cleanup policy; set `keep-milvus=true` only
when preserving the generated Milvus CR for debugging.

The current 2.6 schema matrix keeps the default workflow at three collections,
but each collection is broader:

- `scalar_dynamic_partition_key`: explicit `INT64` PK, partition key with
  `num_partitions`, dynamic field, scalar types `INT8/INT16/INT32/INT64`,
  `FLOAT/DOUBLE`, `BOOL`, `VARCHAR`, `JSON`, and `ARRAY` with
  `INT64/FLOAT/BOOL/VARCHAR` elements.
- `vector_autoid_bm25`: `auto_id` PK, BM25 function, and vector types
  `FLOAT_VECTOR`, `FLOAT16_VECTOR`, `BFLOAT16_VECTOR`, `INT8_VECTOR`,
  `BINARY_VECTOR`, and `SPARSE_FLOAT_VECTOR`.
- `explicit_partitions_nullable`: explicit multi-partitions, `VARCHAR` PK,
  nullable scalar fields, JSON/array data, and HNSW vector search.

The matrix covers these vector indexes in rollback-safe 2.6 data:

- `HNSW`
- `IVF_RABITQ`
- `DISKANN`
- `AUTOINDEX`
- `BIN_IVF_FLAT`
- `SPARSE_INVERTED_INDEX`

It also covers scalar index families used by 2.6 compatibility workloads:

- `STL_SORT`
- `INVERTED`
- `BITMAP`
- `TRIE`
- `NGRAM`
- scalar `AUTOINDEX`

Nullable vector fields remain out of the 2.6 rollback-safe matrix because the
capability catalog treats `NullableVector` as a 3.0+ forward-only capability.

## Scenario Support Matrix

| Scenario | Workflow | Hard gate | Notes |
| --- | --- | --- | --- |
| 2.6.18 jsonShredding -> 2.6 latest jsonShredding -> 2.6.18 | `standalone-2-6-upgrade-rollback` | 2.6 schema/data and pressure | Strict rollback gate with default `rollback-enabled=true`. |
| 2.6.18 jsonShredding -> master jsonShredding -> latest 2.6 | `standalone-2-6-upgrade-rollback` | 2.6 schema/data and pressure | Strict rollback gate. Set `rollback-milvus-image` to latest 2.6 and `rollback-version` to the matching 2.6 operator version. |
| 2.6.18 jsonShredding disabled -> master LoonFFI/vortex -> jsonShredding enabled | `standalone-2-6-upgrade-rollback` | 2.6 schema/data, optional 3.0 target-only schema/data, and pressure | Strict upgrade-only gate. Set `rollback-enabled=false`; this scenario is not a 2.6 rollback gate. |
| 3.0 baseline -> master -> 3.0 baseline | `standalone-3-0-upgrade-rollback` | 3.0 schema/data and pressure | Fully supported. |
| 3.0 baseline jsonShredding -> master jsonShredding + LoonFFI/vortex -> 3.0 baseline | `standalone-3-0-upgrade-rollback` | 3.0 schema/data and pressure | Strict 3.0 rollback gate. |
