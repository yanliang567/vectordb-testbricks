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

## 4am Standalone 2.6 Upgrade/Rollback

`argo/standalone-2-6-upgrade-rollback.yaml` is the 4am workflow template for
the v2.6.18 rollback compatibility lane. Client pods run in `qa` with the
`milvus-upgrade-rollback-runner` ServiceAccount, while the Milvus Operator CR is
created in `qa-milvus`. The template creates a standalone CR with parameterized
4 CPU and 16 GiB defaults, seeds only rollback-safe 2.6 schemas, upgrades to a
configured 2.6 target image, validates existing data, rolls back to v2.6.18, and
validates the same checkpoint again. Client pods use the 4am default resources:
2 CPU / 8 GiB requests and 4 CPU / 16 GiB limits.

This template intentionally does not create `schema_matrix_3_0.yaml` data. New
3.0 schema/data is upgrade-only for this scenario and is not expected to survive
a rollback to 2.6.

Submit example:

```bash
kubectl apply -f argo/standalone-2-6-upgrade-rollback-rbac.yaml
argo submit -n qa --from workflowtemplate/milvus-standalone-2-6-upgrade-rollback \
  -p repo-revision=main \
  -p base-milvus-image=harbor.milvus.io/milvusdb/milvus:v2.6.18 \
  -p target-milvus-image=harbor.milvus.io/milvusdb/milvus:2.6-20260707-682ac8df \
  -p keep-milvus=false
```

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
- `upsert_pressure`
- `delete_pressure`
- `mixed_rw_pressure`

For standalone upgrades, transient request failures can happen while the only
Milvus process restarts. The daemon loops default to
`pressure-fail-on-error=true`, so any pressure failure fails the workflow instead
of being hidden behind foreground validation.

The pressure daemon intentionally does not mount the checkpoint PVC. That keeps
the read/write workload alive during foreground validation without relying on
concurrent ReadWriteOnce volume mounts across multiple pods.

The workflow emits `env_snapshot.json`, `flow_summary.json`,
`orchestrator_report.json`, foreground brick results, checkpoints, pressure
results, and Kubernetes snapshots. `keep-milvus=false` is the default cleanup
policy; set `keep-milvus=true` only when preserving the generated Milvus CR for
debugging.

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
