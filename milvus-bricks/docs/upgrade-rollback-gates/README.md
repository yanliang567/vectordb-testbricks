# Upgrade/Rollback Gate Scenarios

This guide explains the code-managed Argo upgrade/rollback gates under
`milvus_client/manifests/upgrade_rollback_gates.yaml`.

## Current scenario set

The manifest currently registers 26 scenarios:

- 21 promoted gate scenarios
- 2 pre-release candidate scenarios
- 2 known-limitation scenarios
- 1 negative coverage scenario

| Scenario ID | Mode | Classification | Path | Storage feature policy |
| --- | --- | --- | --- | --- |
| `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest` | standalone | gate | `2.6.18 -> 3.0 latest -> 2.6 latest` | LoonFFI/storage v3 and Vortex must stay disabled. |
| `standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest` | standalone | gate | `2.6.18 -> 3.0 latest + 3.0-only forward features -> 2.6 latest` | Forward 3.0 collections are required after upgrade but intentionally excluded from rollback validation. |
| `cluster-2-6-18-to-3-0-latest-rollback-2-6-latest` | cluster | gate | `2.6.18 -> 3.0 latest -> 2.6 latest` | LoonFFI/storage v3 and Vortex must stay disabled. |
| `cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest` | cluster | gate | `2.6.18 -> 3.0 latest + 3.0-only forward features -> 2.6 latest` | Forward 3.0 collections are required after upgrade but intentionally excluded from rollback validation. |
| `standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | standalone | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` | LoonFFI/storage v3 and Vortex disabled; target-created 3.0 collections and indexes are validated before and after rollback. |
| `cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` | LoonFFI/storage v3 and Vortex disabled; target-created 3.0 collections and indexes are validated before and after rollback. |
| `cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline` | cluster | known limitation | `3.0 baseline -> 3.0 latest + JSON Shredding -> 3.0 baseline + JSON Shredding` | JSON-heavy forward data and JSON path indexes remain required after rollback. |
| `cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` on Woodpecker 2CU | Proxy, QueryNode, DataNode, and StreamingNode must each keep at least two replicas. |
| `standalone-3-0-vortex-candidate-upgrade-rollback` | standalone | candidate | `earlier reviewed 3.0 candidate -> newer reviewed candidate + LoonFFI/Vortex -> earlier candidate + LoonFFI/Vortex` | Pre-release evidence for the v3.0.1 contract; not a release gate. |
| `cluster-3-0-vortex-candidate-upgrade-rollback` | cluster | candidate | `earlier reviewed 3.0 candidate -> newer reviewed candidate + LoonFFI/Vortex -> earlier candidate + LoonFFI/Vortex` | Distributed pre-release evidence; not a release gate. |
| `standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline` | standalone | known limitation | `3.0 baseline -> 3.0 latest + JSON Shredding -> 3.0 baseline + JSON Shredding` | JSON-heavy forward data and JSON path indexes remain required after rollback. |
| `standalone-3-0-index-v10-v4-upgrade-rollback` | standalone | gate | `3.0 baseline rollback-safe matrix -> 3.0 latest + target 10/4 -> 3.0 baseline rollback-safe matrix` | Target-only runtime config and SINDI/Block-Max plus JSON scalar index families are checked; forward collections are dropped before rollback. |
| `cluster-3-0-index-v10-v4-upgrade-rollback` | cluster | gate | `3.0 baseline rollback-safe matrix -> 3.0 latest + target 10/4 -> 3.0 baseline rollback-safe matrix` | Distributed equivalent of the target-only index engine gate. |
| `standalone-3-0-index-v11-v4-upgrade-rollback` | standalone | gate | `3.0 baseline rollback-safe matrix -> 3.0 latest + target 11/4 -> 3.0 baseline rollback-safe matrix` | Target-only v11/v4 index engine and algorithm coverage. |
| `cluster-3-0-index-v11-v4-upgrade-rollback` | cluster | gate | `3.0 baseline rollback-safe matrix -> 3.0 latest + target 11/4 -> 3.0 baseline rollback-safe matrix` | Distributed equivalent of the target-only v11/v4 gate. |
| `standalone-3-0-loon-vortex-to-2-6-negative` | standalone | negative | `2.6.18 -> 3.0 latest + LoonFFI/Vortex -> 2.6 latest` | Unsupported negative coverage only; not a promoted gate. |
| `standalone-3-0-1-vortex-self-compat-upgrade-rollback` | standalone | gate | `3.0.1 Vortex -> 3.0.1 Vortex -> 3.0.1 Vortex` | LoonFFI/Vortex enabled in every phase; Vortex TEXT LOB baseline survives the round trip. |
| `cluster-3-0-1-vortex-self-compat-upgrade-rollback` | cluster | gate | `3.0.1 Vortex -> 3.0.1 Vortex -> 3.0.1 Vortex` | Distributed Vortex baseline self-compatibility. |
| `standalone-3-0-0-to-3-0-1-vortex-enable-rollback` | standalone | gate | `3.0.0 legacy -> 3.0.1 + LoonFFI/Vortex -> 3.0.1 + LoonFFI/Vortex` | #52340 upgrade path; rollback stays on 3.0.1 Vortex. The 3.0.1 dual reader handles mixed legacy + Vortex segments. |
| `cluster-3-0-0-to-3-0-1-vortex-enable-rollback` | cluster | gate | `3.0.0 legacy -> 3.0.1 + LoonFFI/Vortex -> 3.0.1 + LoonFFI/Vortex` | Distributed equivalent of the #52340 upgrade path. |
| `standalone-3-0-1-json-shredding-vortex-rollback` | standalone | gate | `3.0.1 Vortex -> 3.0.1 + JSON Shredding + Vortex -> 3.0.1 + JSON Shredding + Vortex` | JSON Shredding and Vortex enabled together. |
| `cluster-3-0-1-json-shredding-vortex-rollback` | cluster | gate | `3.0.1 Vortex -> 3.0.1 + JSON Shredding + Vortex -> 3.0.1 + JSON Shredding + Vortex` | Distributed JSON Shredding plus Vortex coverage. |
| `standalone-3-0-1-loon-ffi-rollback` | standalone | gate | `3.0.1 legacy -> 3.0.1 + LoonFFI(storage v3) -> 3.0.1 legacy` | LoonFFI (storage v3) without Vortex; validates storage v3 -> v2 rollback readability under the dual-engine reader. |
| `cluster-3-0-1-loon-ffi-rollback` | cluster | gate | `3.0.1 legacy -> 3.0.1 + LoonFFI(storage v3) -> 3.0.1 legacy` | Distributed LoonFFI (storage v3) rollback coverage. |
| `standalone-3-0-1-vortex-disable-rollback` | standalone | gate | `3.0.1 legacy -> 3.0.1 + LoonFFI/Vortex -> 3.0.1 legacy` | Disabling Vortex at rollback; the 3.0.1 dual-format reader still reads Vortex segments. |
| `standalone-3-0-1-vortex-disable-keep-loon-rollback` | standalone | gate | `3.0.1 legacy -> 3.0.1 + LoonFFI/Vortex -> 3.0.1 + LoonFFI(no Vortex)` | Disabling Vortex at rollback while keeping LoonFFI (S4 -> S2); dual-format reader still reads Vortex segments. |

Milvus v3.0.0 is not a supported Vortex reader/writer baseline. The two
pre-release candidate scenarios (`standalone/cluster-3-0-vortex-candidate-*`)
use two immutable 3.0 branch images that both contain `milvus-storage 63c29c6`
and Vortex 0.75, and whose images are locked. They are excluded from the
promoted release-gate count.

Within a single 3.0.x version, binaries are dual readers/writers (storage v2 +
v3 engine, parquet + vortex format), so same-version LoonFFI/Vortex/JSON toggles
at rollback are supported positive gates; the only compatibility boundaries are
cross-version (2.6 cannot read storage v3/Vortex; v3.0.0 cannot read the
v3.0.1-upgraded Vortex encoding #52340).

Once v3.0.1 is released, replace the `milvus-3-0-1` placeholder with the
official manifest-list digest and rerun the standalone and cluster gates.

The standalone and cluster target-only feature gates use the 2.6 baseline
matrix for the rollback contract and the 3.0 matrix for forward collections
created only after upgrade. Those forward collections must pass data, index,
search/query, and schema evolution checks on the target version. They are not
part of the 2.6 rollback contract; requiring forward rollback validation for
either gate is rejected by manifest validation.

The v10/v4 and v11/v4 index engine gates follow the same target-only contract.
Milvus v3.0.0 does not contain the fix tracked by
[#52767](https://github.com/milvus-io/milvus/issues/52767), so it is not asked to
create or validate those dedicated index matrices. Base and rollback use the
rollback-safe matrix with default index engine selection; only the target phase
sets `dataCoord.targetVecIndexVersion` / `targetScalarIndexVersion`, creates the
dedicated forward collections, and runs strict index validation. The forward
collections are dropped before rollback and are intentionally excluded from
rollback validation, while the rollback-safe collections still validate the
upgrade and rollback lifecycle.

### Index engine compatibility contracts

Manifest v2 expresses index-engine lifecycle policy with one structured
`index_engine_contract` instead of independently maintained matrix, phase
version, drop, and rollback-validation fields. The compiler owns those derived
fields and rejects a scenario that also declares them directly.

| Contract | Baseline/rollback matrix | Index version phases | Target data after rollback |
| --- | --- | --- | --- |
| `target_only` | `rollback_safe_matrix_ref` | target only | dropped before rollback; absence enforced |
| `round_trip` | `matrix_ref` | base, target, rollback | retained and strictly validated |

Use `target_only` when the exact baseline image has not passed the capability.
Use `round_trip` only after `capability_qualifications` records `status: passed`
for the capability, the exact digest-pinned base/rollback image, and the
scenario's standalone or cluster topology. Evidence must be a stable
`argo://` workflow reference or `https://` run URL. The renderer fails closed
if an image override does not match that evidence. A v10 qualification never
authorizes v11.

A target-only `rollback_safe_matrix_ref` must contain only
`compat_mode: rollback_safe` schemas and must not require any `IndexEngine*`
capability. This prevents a matrix for another engine version from being
misclassified as baseline-safe.

The lifecycle contract applies to both explicit forward-matrix collections and
the base-matrix collections created by phase DML/DQL after upgrade. In
`target_only`, both groups finish their target-phase validation and are deleted
before rollback. The rollback phase still reloads and validates baseline
collections, requires the phase-new checkpoint group to be absent, skips
carried DML for that deleted group, and creates the normal after-rollback group.
In `round_trip` and `none`, phase-new collections remain present and retain the
existing reload/query and carried-DML validation path.

For a patch release, add a version-specific scenario, pin the baseline and
rollback aliases to the previous supported release digest, qualify each
capability/topology, and select the contract mode. Do not redefine the two
contract modes or reuse evidence from another image. Cross-minor releases such
as 3.0.x to 3.1.0 must additionally review schema/storage format, SDK/config,
and WorkflowTemplate branch constraints; add a new matrix only when those
capabilities change.

The renderer always emits protected metadata:
`index-engine-contract-mode`, `index-engine-capability`, and
`index-engine-qualification-status`. Non-index scenarios use
`none/none/not_applicable`. Environment snapshots, flow summaries, cleanup
fallbacks, and final JSON/Markdown reports preserve these values. The contract
mode also controls the phase-new cleanup node and rollback checkpoint behavior;
other DAG paths continue to use the compiled matrix and lifecycle parameters.

Every registered upgrade/rollback workflow renders `log.level: debug` by
default through the protected `milvus-log-level` parameter. Operator CR patches
and Helm `user.yaml` rewrites preserve the value across base, target, optional
config toggle, and rollback phases. Override it only for a deliberate diagnostic
run; the code-managed scenario default remains `debug`.

The standalone and cluster JSON Shredding gates both write JSON-heavy forward
data only after the post-upgrade configuration rollout has enabled JSON
Shredding. The rollback phase keeps the setting enabled and requires the
forward data, dynamic JSON fields, JSON path indexes, and filters to remain
usable.

The LoonFFI/Vortex candidate scenarios create forward collections from
`schema_matrix_3_0_storage_v3.yaml`. They validate TEXT payloads below, at, and
above 64 KiB plus a 1 MiB value, then rerun payload hash, lexical filter, BM25,
index, and feature-semantic checks after rollback.
Workflow clients use `pymilvus==3.0.1`; the 3.0.0 wheel predates the client-side
`DataType.TEXT` backport and cannot render this matrix.

The regular matrices include the promoted type/index coverage:

- `schema_matrix_2_6.yaml`: StructArray scalar round-trip and element search,
  all six nullable vector types, Geometry/RTREE, and explicit legacy indexes.
- `schema_matrix_3_0.yaml`: StructArray nested scalar indexes including
  `FLOAT + STL_SORT/INVERTED` and `VARCHAR + INVERTED/BITMAP`, EmbList DISKANN,
  FAISS, MinHash exact self-search with observational near-duplicate recall,
  and TIMESTAMPTZ entity TTL.
- `schema_matrix_3_0_index_v10_v4.yaml`: SINDI, Block-Max sparse algorithms,
  JSON scalar indexes, and resolved HYBRID AutoIndex with runtime target
  versions `10/4` configured. Public SDK index metadata does not expose the
  exact engine version selected by Milvus after its version resolution/clamp
  logic, so execution reports must preserve DataNode index-build logs as
  supplementary evidence rather than claim exact `10/4` builds automatically.
- `schema_matrix_3_0_index_v11_v4.yaml`: the corresponding target-only v11/v4
  matrix; it follows the same evidence and rollback-isolation policy.

The Woodpecker 2CU gate reuses the cluster Helm rolling upgrade workflow with
a multi-replica data plane. Its scenario contract rejects deploy-profile
overrides that reduce Proxy, QueryNode, DataNode, or StreamingNode below two
replicas. The existing pressure and serviceability gates remain strict outside
confirmed rollout connectivity windows, but this scenario does not define a
zero-request-failure availability SLO.

## Centralized change points

For normal branch or version updates, start here:

1. `milvus_client/manifests/upgrade_rollback_gates.yaml`
   - `image_aliases`: concrete image tags and logical versions.
   - `scenarios`: path definitions, workflow template selection, deploy profile,
     schema matrix, storage feature flags, and validation policy.
   - `defaults`: common workload sizes, validation toggles, and the default
     `milvus_log_level: debug` setting.
   - `index_engine_contract`: capability lifecycle policy for an index scenario.
   - `capability_qualifications`: immutable-image/topology evidence required by
     `round_trip`.
2. `milvus_client/manifests/deploy_profiles/*.yaml`
   - standalone or cluster deployment topology.
   - Helm chart repo/chart/version for cluster mode.
3. `milvus_client/manifests/schema_matrix_*.yaml`
   - schema/index coverage for each Milvus branch family.
4. `argo/*.yaml`
   - only when a new workflow parameter or DAG behavior is required.

The release baseline aliases are pinned to their official multi-arch
manifest-list digests:

```text
harbor.milvus.io/milvusdb/milvus:v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6
harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862
```

The tag remains in the reference for readability; the digest fixes the actual
image content even though Harbor does not mark the tag immutable. Do not use a
retention-limited `3.0-YYYYMMDD-<sha>` branch build as the long-lived baseline
alias. The reviewed candidate aliases are a temporary exception: they are
pinned by digest, record source/storage commits, and reject runtime overrides.
At the time of this change Harbor had no v3.0.1 release or previous-day
multi-arch 3.0 image, so the candidate path uses the previous available
`3.0-20260807-697431f2` build as base/rollback and
`3.0-20260807-1439dc7d` as target. Both resolve to `milvus-storage 63c29c6`
with Vortex 0.75.

If you add a new branch family such as `3.1` or `4.0`, add an image alias,
add or reuse a schema matrix, register the new scenario IDs, then update the
manifest and renderer tests. Patch releases normally require manifest-only
scenario data changes; changing Python or Argo is necessary only for a new
compatibility semantic or workflow branch behavior.

## Rendering Argo submit parameters

Run these commands from `milvus-bricks/`.

Standalone 3.0 LoonFFI/Vortex candidate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id standalone-3-0-vortex-candidate-upgrade-rollback \
  --format argo-args
```

Standalone 2.6 -> 3.0 target-only feature gate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest \
  --format argo-args \
  --allow-placeholder
```

Cluster 2.6 -> 3.0 target-only feature gate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest \
  --format argo-args \
  --allow-placeholder
```

Cluster JSON Shredding gate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline \
  --format argo-args \
  --allow-placeholder
```

Cluster 3.0 LoonFFI/Vortex candidate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-vortex-candidate-upgrade-rollback \
  --format argo-args
```

Cluster Woodpecker 2CU HA gate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline \
  --format argo-args \
  --allow-placeholder
```

`--allow-placeholder` is only for dry-run/review output. Before submitting a
formal gate, replace placeholder aliases such as `milvus-3-0-latest` and
`milvus-2-6-latest` with concrete image tags in
`upgrade_rollback_gates.yaml`.

## Submitting to Argo

Generate the arguments first:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-vortex-candidate-upgrade-rollback \
  --repo-revision <full-commit-sha> \
  --format argo-args
```

Copy the single argument line emitted by the renderer and paste it directly
after `argo submit -n qa`. Do not submit renderer output through shell command
substitution, `eval`, or `xargs`; those forms do not preserve the quoted
argument boundaries consistently across shells.

Do not pass `--allow-placeholder` for a real submit path. The renderer should
fail fast if any runnable image still contains `placeholder`. The checkout
logic accepts a branch, tag, or commit SHA, but formal gates and calibration
runs should pass a full commit SHA through `--repo-revision` so every workflow
step executes the same reviewed test implementation.

## Safety rules

- `2.6 -> 3.0 -> 2.6` promoted gates must keep LoonFFI/storage v3 and Vortex
  disabled in every phase.
- LoonFFI/storage v3 is represented by Milvus config key
  `common.storage.useLoonFFI`.
- Vortex is represented by Milvus config key `dataNode.storage.format=vortex`.
- Enabling Vortex requires LoonFFI/storage v3 to be enabled in the same phase
  (`vortex_enabled=true` implies `loon_ffi_enabled=true`); the manifest validator
  and the CR/Helm renderers both fail closed on `Vortex` without `useLoonFFI`.
- Promoted Vortex gates require Milvus v3.0.1 or later for every Vortex writer
  and for any rollback reader that may encounter Vortex data.
- Within a single 3.0.x version the binary is a dual reader/writer (storage v2 +
  v3 engine, parquet + vortex format), so toggling LoonFFI or Vortex on/off at
  rollback is a supported positive scenario. The compatibility boundaries are
  cross-version only: 2.6 cannot read storage v3/Vortex, and v3.0.0 cannot read
  the v3.0.1-upgraded Vortex encoding (#52340).
- Pre-release candidate aliases are immutable and locked in the manifest.
  Runtime image/version overrides are rejected; refresh the reviewed alias and
  storage/source commit metadata through a code change instead.
- `allow_unsafe_negative_coverage` is allowed only for explicitly registered
  negative scenarios.
- The WorkflowTemplate runtime guard still rejects unsafe 2.6 rollback storage
  flags for direct `argo submit` usage.

## Validation coverage

The current gates validate:

- baseline, target, and rollback storage config before data validation:
  - standalone checks Milvus CR `spec.config.common.storage.useLoonFFI` and
    `spec.config.dataNode.storage.format`, then checks the running Milvus pod's
    effective mounted config by merging `/milvus/configs/milvus.yaml` with
    `/milvus/configs/user.yaml`;
  - cluster checks Helm release values `extraConfigFiles.user.yaml`, then checks
    the running DataNode pod's effective mounted config by merging
    `/milvus/configs/milvus.yaml` with `/milvus/configs/user.yaml`;
  - disabled LoonFFI must be confirmed from the merged runtime config, because
    the explicit `false` override may be omitted from user config;
  - disabled Vortex accepts the Milvus default `dataNode.storage.format: parquet`
    from merged runtime config, while still rejecting `vortex` or invalid custom
    formats;
  - mismatched LoonFFI/storage v3 or Vortex settings fail the gate before
    baseline seed, precheck, DML/DQL, or index compatibility validation;
- schema feature semantics at base, after upgrade, and after rollback:
  - non-`MAX_SIM_*` StructArray element searches require the expected primary
    key and element offset;
  - `MAX_SIM_*` StructArray searches use PyMilvus `EmbeddingList` row-level
    queries and require the expected primary key without an element offset;
  - nested scalar indexes execute real `MATCH_ANY` filters;
  - unknown validator names fail manifest validation rather than silently pass;
  - index engine scenarios verify `dataCoord.targetVecIndexVersion` and
    `dataCoord.targetScalarIndexVersion` from merged runtime pod config and
    execute the matrix algorithms; this is target-configuration validation,
    not proof of an exact resolved build version;
- baseline seed data after upgrade and after rollback;
- phase checkpoints for data written after upgrade before rollback;
- new collections created after upgrade and after rollback;
- DML on carried collections in each phase:
  - insert 1000 rows;
  - upsert the inserted PK range;
  - delete 100 rows from that inserted PK range;
  - expected net increase: 900 rows per phase;
- DQL on old and new collections after each phase;
- index compatibility and loaded-state load/search/query probes, followed by a
  strict release/reload cycle and the same query/search probes again;
- forward collection index compatibility after upgrade, with a separate
  `/tmp/milvus-bricks/checkpoints/forward/index_compatibility.json` checkpoint;
- forward index compatibility after rollback only when
  `rollback-forward-validation-enabled=true`;
- schema-evolution checkpoints written after upgrade and validated read-only
  after rollback for existing and rollback-compatible forward collections;
- Woodpecker 2CU topology requirements at render time and again before Helm
  deployment for registered runtime scenarios;
- registered scenario schema/index/validator parameters and WorkflowTemplate
  topology revalidated immediately before deployment;
- continuous pressure workload, with rollout maintenance windows only excluding
  confirmed connectivity failures.
