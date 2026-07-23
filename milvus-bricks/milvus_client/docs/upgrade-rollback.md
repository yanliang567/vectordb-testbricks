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
`milvus-upgrade-rollback-runner` ServiceAccount, while Milvus resources are
created in `qa-milvus`. The standalone templates create a Milvus Operator CR with parameterized
2 CPU / 4 GiB requests and 4 CPU / 8 GiB limits, seeds only rollback-safe 2.6
schemas, upgrades to a configured 2.6 or master/3.0 target image, validates
existing data, rolls back to a latest 2.6 image that contains #50792, and
validates the same checkpoint again.
Client pods default to 1 CPU / 2 GiB requests and 2 CPU / 4 GiB limits.
Set `rollback-milvus-image` and `rollback-version` explicitly when a scenario
rolls back to a different 2.6 branch image, such as latest 2.6 instead of the
original v2.6.18 image.

The officially supported `v2.6.18 -> latest 3.0 -> 2.6` rollback path has a
strict configuration constraint: do not enable storage v3 or vortex after
upgrading to 3.0. Enabling storage v3 or setting `dataNode.storage.format` to
`vortex` can make rollback to 2.6 unsafe and may trigger panic during rollback.
Keep these runs on the legacy storage format; use the 3.0 baseline rollback
lane for LoonFFI/vortex hard gates.

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
- `base-loon-ffi-enabled`
- `target-loon-ffi-enabled`
- `rollback-loon-ffi-enabled`
- `base-vortex-enabled`
- `target-vortex-enabled`
- `rollback-vortex-enabled`
- `post-upgrade-config-toggle-enabled`
- `post-upgrade-json-shredding-enabled`
- `forward-workload-enabled`
- `forward-schema-matrix`
- `forward-collection-prefix`
- `rollback-enabled`
- `rollback-version`
- `rollback-forward-validation-enabled`
- `index-compatibility-validation-enabled`
- `phase-dml-dql-validation-enabled`
- `phase-new-collection-rows`
- `phase-existing-dml-rows`
- `phase-existing-delete-rows`
- `allow-unsafe-negative-coverage` (negative scenario only; default `false`)
- `observe-before-upgrade-sec`
- `observe-after-upgrade-sec`
- `observe-before-rollback-sec`
- `observe-after-rollback-sec`
- `rollback-serviceability-timeout-sec`
- `rollback-serviceability-interval-sec`

The workflow writes `spec.config.common.storage.jsonShreddingEnabled` during
base deploy and image patch phases. Milvus 3.0 StorageV3 is represented by
`spec.config.common.storage.useLoonFFI`, exposed as `*-loon-ffi-enabled`; there
is no separate `storageV3Enabled` CR key. Image/config patch phases always write
the requested `useLoonFFI` value and `dataNode.storage.format` value, so a phase
can explicitly clear a previously enabled LoonFFI/vortex config. Vortex is
controlled by the separate `*-vortex-enabled` parameters and is not implicitly
enabled by LoonFFI. The workflow snapshots Milvus CR config after base deploy,
target upgrade, optional post-upgrade config toggle, and rollback.

The `allow-unsafe-negative-coverage` parameter exists only to run the
unsupported negative scenario that intentionally enables LoonFFI/vortex before a
2.6 rollback. Promoted gate scenarios keep it `false`; manifest validation
rejects enabling it on `classification: gate`, and the Workflow runtime also
requires an approved negative `scenario-id` before honoring the bypass.

The standalone templates run pressure during fixed observe windows before and
after both upgrade and rollback. These observe windows default to 300 seconds so
client request traffic covers at least five minutes in each steady phase.
`observe-after-upgrade-sec` and `observe-before-rollback-sec` are sequential,
not overlapping. The workflow runs the after-upgrade observation first, then
upgrade-phase precheck/validation, index compatibility validation, phase
DML/DQL validation, existing schema evolution, optional post-upgrade config
patch, optional forward workload, and only then the before-rollback observation.

After rollback, the templates run a data serviceability wait gate before strict
integrity validation. This gate repeatedly runs lightweight checkpoint count and
PK sample queries while pressure continues. It retries only known transient
query serviceability errors such as channel-not-available and no-shard-leader
responses. If the data becomes queryable again, the brick records
`recovered=true`, `recovery_duration_sec`, `attempts`, and
`transient_failure_attempts`; if the timeout expires, the workflow fails before
checksum validation. The default timeout is 900 seconds with a 10 second
interval.

When `index-compatibility-validation-enabled=true`, rollback workflows also
exercise index-version compatibility explicitly. After upgrade, the workflow
flushes and loads baseline collections, records the actual index metadata from
`list_indexes` / `describe_index`, runs indexed vector search, indexed scalar
filter queries, and checkpoint count/PK queries, and writes
`/tmp/milvus-bricks/checkpoints/index_compatibility.json`. After rollback, the
workflow reads that checkpoint, re-enumerates the actual indexes, compares
index name/field/type/metric metadata, and validates load/search/query again.
Scalar index validation selects a deterministic non-null probe row when
possible, runs a scalar-only query to prove the predicate has matches, then runs
`scalar predicate + primary-key predicate` so non-unique scalar conditions do not
depend on unordered query results. Deterministic vector self-search must return
the expected PK and a sane self-match distance/score when the metric supports
that assertion. L2/HAMMING/JACCARD self-search expects near-zero distance, while
COSINE/IP expects a high similarity score. AutoID checkpoints keep both the
deterministic data-generation id range and the actual Milvus-generated PKs, so
query vectors/filters are rebuilt from generation ids while expected hits use
actual PKs.
Promoted gates do not drop/recreate baseline indexes while the pressure daemon
is running; `--rebuild-index=true` remains available only for manual diagnostic
runs outside strict pressure.

When `phase-dml-dql-validation-enabled=true`, rollback workflows also exercise
active request compatibility at both phase boundaries:

- after upgrade: run insert/upsert/delete on baseline collections, create one
  `${collection-prefix}_after_upgrade` collection per schema, then query/search
  both old and new collections. The upsert/delete operations target the PK range
  inserted by this phase, not the original baseline seed rows. The workflow
  persists `/tmp/milvus-bricks/checkpoints/phase_dml_dql_after_upgrade.json`
  with the inserted PK ranges, deleted PKs, upsert sample values, and new
  collection row counts;
- after rollback: first validate the after-upgrade phase checkpoint, proving the
  baseline `50000000` range and upgrade-created `60000000` collections survived
  rollback. Only after that succeeds does the workflow run insert/upsert/delete
  on baseline collections again, carry the upgrade-created collections forward
  for another DML/DQL round, create one `${collection-prefix}_after_rollback`
  collection per schema, then query/search all of them. The carried collection
  upsert/delete operations also target rows inserted during the rollback phase.

Default deterministic data scale:

| Gate family | Schema count | Before upgrade baseline | After-upgrade phase validation | Before rollback after schema evolution | After-rollback phase validation |
| --- | --- | --- | --- | --- | --- |
| 2.6 rollback gate | 3 | baseline `15000` | baseline `17700`, upgrade-new `9000` | same as after-upgrade; schema evolution disabled | baseline `20400`, upgrade-new/carried `11700`, rollback-new `9000` |
| 3.0 branch gate | 4 | baseline `20000` | baseline `23600`, upgrade-new `12000` | baseline `43600`, upgrade-new `12000`; schema evolution adds `4 × 5000` | baseline `47200`, upgrade-new/carried `15600`, rollback-new `12000` |

Per collection, phase DML inserts `1000` rows, upserts the same PK range when
the schema has explicit PK, and deletes `100` rows, so the net row increase is
`900`. Auto-id collections skip upsert and still net `900` after delete. The
upsert check queries sample PKs and compares the updated field value generated
with `seed + 101`, so a no-op upsert is reported as a validation failure. The
rollback phase validates the after-upgrade phase checkpoint before any new
rollback writes, so losing all `50000000` / `60000000` phase data is reported
before the `70000000` / `80000000` ranges are inserted. The
table excludes background/foreground pressure workload writes because those are
not a stable row-count contract.

Submit example:

```bash
kubectl apply -f argo/standalone-2-6-upgrade-rollback-rbac.yaml
argo submit -n qa --from workflowtemplate/milvus-standalone-2-6-upgrade-rollback \
  -p repo-revision=main \
  -p base-milvus-image=harbor.milvus.io/milvusdb/milvus:v2.6.18 \
  -p target-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<sha> \
  -p target-version=3.0.0 \
  -p rollback-milvus-image=harbor.milvus.io/milvusdb/milvus:2.6-YYYYMMDD-<sha> \
  -p rollback-version=2.6.0 \
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

For `v2.6.18 -> master/3.0 -> latest 2.6` runs, keep the 2.6 template and pass
the master/3.0 image as `target-milvus-image`. Use a rollback 2.6 build that
contains #50792; `v2.6.18` is a diagnostic-only rollback target for #50694 and
is not a positive gate target with recent 3.0 images. The schema matrix remains
`schema_matrix_2_6.yaml` unless `forward-workload-enabled=true` is explicitly
set for target-only 3.0 coverage.

Code-managed gate definitions live in
`milvus_client/manifests/upgrade_rollback_gates.yaml`. This file is the source
of truth for gate branch/version paths. It intentionally separates reusable
definitions from scenario composition:

- `image_aliases`: concrete image + operator `version` pairs, such as
  `milvus-2-6-18`, `milvus-3-0-baseline`, or `milvus-3-0-latest`.
- `schema_matrices`: branch-level schema matrix paths.
- `deploy_profiles`: code-managed deployment topology. Standalone profiles render
  Milvus Operator CRs; cluster Pulsar/Woodpecker profiles render Helm chart
  values.
- `workflow_templates`: Argo WorkflowTemplate names.
- `scenarios`: gate or negative scenario composition using the refs above.

Formal gate submissions should resolve placeholder/latest images to concrete
Harbor tags before submission and should record the scenario id in the workflow
report. The parameter renderer rejects placeholder images for promoted gate
scenarios by default. Generate submit parameters from a scenario id after
resolving images instead of manually copying every `-p` flag:

```bash
PYTHONPATH=. python -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id standalone-2-6-18-to-3-0-latest-rollback-2-6-latest \
  --format argo-args
```

For dry-run/review output before replacing placeholders, pass
`--allow-placeholder`.

Copy the output after `argo submit -n qa`; do not wrap the command in shell
substitution because `pressure-modules` intentionally contains spaces. For
example:

```bash
argo submit -n qa \
  --from workflowtemplate/milvus-standalone-2-6-upgrade-rollback \
  -p scenario-id=standalone-2-6-18-to-3-0-latest-rollback-2-6-latest \
  -p base-milvus-image=harbor.milvus.io/milvusdb/milvus:v2.6.18 \
  -p target-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<sha> \
  -p rollback-milvus-image=harbor.milvus.io/milvusdb/milvus:2.6-YYYYMMDD-<sha> \
  -p base-loon-ffi-enabled=false \
  -p target-loon-ffi-enabled=false \
  -p rollback-loon-ffi-enabled=false \
  -p base-vortex-enabled=false \
  -p target-vortex-enabled=false \
  -p rollback-vortex-enabled=false \
  -p 'pressure-modules=search_pressure query_pressure query_iterator_scan count_pressure upsert_pressure delete_pressure mixed_rw_pressure' \
  -p keep-milvus=false
```

Use `--format json` for automation that wants to build the `argo submit`
command programmatically.

To switch a gate to a new branch/version, keep the change centralized:

1. Add or update one `image_aliases` entry with the concrete Harbor image and
   Milvus Operator `version`.
2. Add a `schema_matrices` entry only if the branch needs a new schema matrix,
   such as `schema_matrix_3_1.yaml` or `schema_matrix_4_0.yaml`.
3. Add or update a `scenario` by changing `base/target/rollback.image_ref`,
   `schema_matrix_ref`, and `workflow_template_ref`.
4. Add a deploy profile only when the topology changes, for example a new
   cluster CU shape, MQ, dependency, or component layout.
5. Do not change workflow YAML unless the DAG itself needs a new phase or a new
   runtime parameter. Branch/version changes should normally stay in the
   manifest.

For example, adding `3.1 baseline -> 3.1 latest -> 3.1 baseline` should only
need new `milvus-3-1-baseline` / `milvus-3-1-latest` aliases, a `3.1` schema
matrix ref if needed, and one new scenario. Adding `4.0` follows the same
pattern; create or update workflow templates only if 4.0 requires different
upgrade/rollback orchestration.

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
- `wait_data_serviceability` after upgrade for cluster gates
- `validate_data_integrity` after upgrade
- `wait_data_serviceability` after rollback
- `validate_data_integrity` after rollback
- `run-pressure-suite` before upgrade
- `run-pressure-suite` after upgrade
- `run-pressure-suite` before rollback
- `run-pressure-suite` after rollback

It also starts a daemon workload loop after the strict pre-upgrade pressure
suite:

- `search_pressure`
- `query_pressure`
- `query_iterator_scan`
- `count_pressure`
- `upsert_pressure`
- `delete_pressure`
- `mixed_rw_pressure`

## 4am Cluster Upgrade/Rollback

`argo/cluster-upgrade-rollback.yaml` is the cluster-mode counterpart. It reuses
the same schema, seed, validation, pressure, serviceability, reporting, and
cleanup bricks as the standalone templates, but deploys Milvus through the
Milvus Helm chart from the code-managed `cluster-pulsar-1cu` profile by
default:

- `deploy-profile=milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml`
- `deployer=helm`
- `chart=zilliztech/milvus`
- `chart_version=5.0.24`
- `mode=cluster`
- `msgStreamType=pulsar`
- `mixCoord/proxy/queryNode/dataNode/streamingNode` explicitly configured

The 2.6 -> 3.0 -> 2.6 cluster gates use Pulsar because 2.6 does not support the
external Woodpecker client topology used by current 3.0 Helm deployments. The
3.0 -> 3.0 cluster gate still uses `cluster-woodpecker-1cu` to keep external
Woodpecker covered on the branch that supports it.

The deploy profiles are stored under
`milvus_client/manifests/deploy_profiles/`. Cluster Workflow templates call
`milvus_client.requests.render_milvus_helm_values` to render Helm values and
write `deploy_topology.json`, so reports include the actual deployer, mode,
component replicas/resources, Helm chart version, image, version, and dependency
topology used for the run. Upgrade and rollback phases reuse the same
profile-defined Helm repo/chart/version instead of resolving the latest chart at
runtime. This avoids the current 4am Operator limitation where Woodpecker
cluster dependencies can fail before Milvus is deployed.

Use the cluster workflow when the test objective is rolling behavior under
distributed Milvus components. Use standalone workflows for compact data
compatibility gates where standalone restart-window request failures are
acceptable as warnings.

For standalone upgrades, transient request failures can happen while the only
Milvus process restarts. The daemon loop defaults to
`pressure-fail-on-error=false` and `gate-allow-warning=true`, so individual
daemon failures are preserved in the final report as warnings. Steady-state
traffic is still strict: the foreground `run-pressure-suite` runs every pressure
module against the baseline collections before upgrade, after the upgraded
target has passed its observe window and validation, before rollback, and after
rollback validation. Any foreground pressure failure fails the workflow.

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
metadata, rollback serviceability recovery metrics, config matrix parameters,
and snapshot paths into one comparable artifact. Background daemon pressure
warnings do not hide failed foreground validations or foreground pressure
suites; those steps fail before the final gate.
`keep-milvus=false` is the default cleanup policy; set `keep-milvus=true` only
when preserving the generated Milvus release for debugging. Cluster cleanup runs
`helm uninstall`, deletes label-selected workflow resources, and explicitly
deletes Helm StatefulSet PVCs named `data-<release>-etcd-N` and
`woodpecker-storage-<release>-woodpecker-N` because these PVC templates may not
carry the workflow labels.

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
| 2.6.18 -> latest 3.0 -> latest 2.6 | `milvus-standalone-2-6-upgrade-rollback` | 2.6 schema/data, serviceability, and pressure | Positive gate. Rollback target must contain #50792; storage v3 and vortex stay disabled after upgrade. |
| 3.0 baseline -> latest 3.0 -> 3.0 baseline | `milvus-standalone-3-0-upgrade-rollback` | 3.0 schema/data, serviceability, and pressure | Strict 3.0 branch rollback gate. |
| cluster 2.6.18 -> latest 3.0 -> latest 2.6 | `milvus-cluster-upgrade-rollback` | 2.6 schema/data, serviceability, pressure, and cluster topology | Positive cluster gate. Rollback target must contain #50792; storage v3 and vortex stay disabled after upgrade. |
| cluster 3.0 baseline -> latest 3.0 -> 3.0 baseline | `milvus-cluster-upgrade-rollback` | 3.0 schema/data, serviceability, pressure, and cluster topology | Strict 3.0 branch rollback gate under distributed components. |
| 2.6.18 -> latest 3.0 LoonFFI/vortex -> latest 2.6 | `milvus-standalone-2-6-upgrade-rollback` | Negative/observe-only | Not a promoted gate. This documents the unsafe rollback boundary and must not be treated as supported. |
