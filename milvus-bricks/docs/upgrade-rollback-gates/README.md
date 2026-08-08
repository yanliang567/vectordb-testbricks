# Upgrade/Rollback Gate Scenarios

This guide explains the code-managed Argo upgrade/rollback gates under
`milvus_client/manifests/upgrade_rollback_gates.yaml`.

## Current scenario set

The manifest currently registers 14 scenarios:

- 13 promoted gate scenarios
- 1 negative coverage scenario

| Scenario ID | Mode | Classification | Path | Storage feature policy |
| --- | --- | --- | --- | --- |
| `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest` | standalone | gate | `2.6.18 -> 3.0 latest -> 2.6 latest` | LoonFFI/storage v3 and Vortex must stay disabled. |
| `standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest` | standalone | gate | `2.6.18 -> 3.0 latest + 3.0-only forward features -> 2.6 latest` | Forward 3.0 collections are required after upgrade but intentionally excluded from rollback validation. |
| `cluster-2-6-18-to-3-0-latest-rollback-2-6-latest` | cluster | gate | `2.6.18 -> 3.0 latest -> 2.6 latest` | LoonFFI/storage v3 and Vortex must stay disabled. |
| `cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest` | cluster | gate | `2.6.18 -> 3.0 latest + 3.0-only forward features -> 2.6 latest` | Forward 3.0 collections are required after upgrade but intentionally excluded from rollback validation. |
| `standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | standalone | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` | LoonFFI/storage v3 and Vortex disabled. |
| `cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` | LoonFFI/storage v3 and Vortex disabled. |
| `cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest + JSON Shredding -> 3.0 baseline + JSON Shredding` | JSON-heavy forward data and JSON path indexes remain required after rollback. |
| `cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` on Woodpecker 2CU | Proxy, QueryNode, DataNode, and StreamingNode must each keep at least two replicas. |
| `standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline` | standalone | gate | `3.0 baseline -> 3.0 latest + LoonFFI/Vortex -> 3.0 baseline + LoonFFI/Vortex` | Target and rollback both keep LoonFFI/storage v3 and Vortex enabled. |
| `cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest + LoonFFI/Vortex -> 3.0 baseline + LoonFFI/Vortex` | Target and rollback both keep LoonFFI/storage v3 and Vortex enabled. |
| `standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline` | standalone | gate | `3.0 baseline -> 3.0 latest + JSON Shredding -> 3.0 baseline + JSON Shredding` | JSON-heavy forward data and JSON path indexes remain required after rollback. |
| `standalone-3-0-index-v10-v4-upgrade-rollback` | standalone | gate | `3.0 baseline + index v10/v4 -> 3.0 latest + index v10/v4 -> 3.0 baseline + index v10/v4` | SINDI/Block-Max and scalar index v4 are validated against runtime config. |
| `cluster-3-0-index-v10-v4-upgrade-rollback` | cluster | gate | `3.0 baseline + index v10/v4 -> 3.0 latest + index v10/v4 -> 3.0 baseline + index v10/v4` | Distributed equivalent of the index engine version gate. |
| `standalone-3-0-loon-vortex-to-2-6-negative` | standalone | negative | `2.6.18 -> 3.0 latest + LoonFFI/Vortex -> 2.6 latest` | Unsupported negative coverage only; not a promoted gate. |

For the 3.0 LoonFFI/Vortex gates, the rollback phase uses the 3.0 baseline
image but keeps LoonFFI/storage v3 and Vortex enabled. This validates image
rollback compatibility after the upgraded version has written data and indexes
with the 3.0 storage features enabled.

The standalone and cluster target-only feature gates use the 2.6 baseline
matrix for the rollback contract and the 3.0 matrix for forward collections
created only after upgrade. Those forward collections must pass data, index,
search/query, and schema evolution checks on the target version. They are not
part of the 2.6 rollback contract; requiring forward rollback validation for
either gate is rejected by manifest validation.

The standalone and cluster JSON Shredding gates both write JSON-heavy forward
data only after the post-upgrade configuration rollout has enabled JSON
Shredding. The rollback phase keeps the setting enabled and requires the
forward data, dynamic JSON fields, JSON path indexes, and filters to remain
usable.

The LoonFFI/Vortex gates create forward collections from
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
  FAISS, MinHash, and TIMESTAMPTZ entity TTL.
- `schema_matrix_3_0_index_v10_v4.yaml`: SINDI, Block-Max sparse algorithms,
  JSON scalar indexes, and resolved HYBRID AutoIndex under runtime index
  versions `10/4`.

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
   - `defaults`: common workload sizes and validation toggles.
2. `milvus_client/manifests/deploy_profiles/*.yaml`
   - standalone or cluster deployment topology.
   - Helm chart repo/chart/version for cluster mode.
3. `milvus_client/manifests/schema_matrix_*.yaml`
   - schema/index coverage for each Milvus branch family.
4. `argo/*.yaml`
   - only when a new workflow parameter or DAG behavior is required.

The `milvus-3-0-baseline` alias is pinned to the official multi-arch `v3.0.0`
manifest-list digest:

```text
harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862
```

The tag remains in the reference for readability; the digest fixes the actual
image content even though Harbor does not mark the tag immutable. Do not use a
retention-limited `3.0-YYYYMMDD-<sha>` branch build as the long-lived baseline
alias; use those concrete build tags as explicit target or one-off phase
overrides instead.

If you add a new branch family such as `3.1` or `4.0`, add an image alias,
add or reuse a schema matrix, register the new scenario IDs, then update the
manifest and renderer tests.

## Rendering Argo submit parameters

Run these commands from `milvus-bricks/`.

Standalone 3.0 LoonFFI/Vortex gate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline \
  --format argo-args \
  --allow-placeholder
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

Cluster 3.0 LoonFFI/Vortex gate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline \
  --format argo-args \
  --allow-placeholder
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
  --scenario-id cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline \
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
  - StructArray searches require the expected primary key and element offset;
  - nested scalar indexes execute real `MATCH_ANY` filters;
  - unknown validator names fail manifest validation rather than silently pass;
  - index engine scenarios verify `dataCoord.targetVecIndexVersion` and
    `dataCoord.targetScalarIndexVersion` from merged runtime pod config;
- baseline seed data after upgrade and after rollback;
- phase checkpoints for data written after upgrade before rollback;
- new collections created after upgrade and after rollback;
- DML on carried collections in each phase:
  - insert 1000 rows;
  - upsert the inserted PK range;
  - delete 100 rows from that inserted PK range;
  - expected net increase: 900 rows per phase;
- DQL on old and new collections after each phase;
- index compatibility and load/search/query probes;
- forward collection index compatibility after upgrade, with a separate
  `/tmp/milvus-bricks/checkpoints/forward/index_compatibility.json` checkpoint;
- forward index compatibility after rollback only when
  `rollback-forward-validation-enabled=true`;
- Woodpecker 2CU topology requirements at render time and again before Helm
  deployment for registered runtime scenarios;
- continuous pressure workload, with rollout maintenance windows only excluding
  confirmed connectivity failures.
