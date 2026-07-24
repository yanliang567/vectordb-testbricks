# Upgrade/Rollback Gate Scenarios

This guide explains the code-managed Argo upgrade/rollback gates under
`milvus_client/manifests/upgrade_rollback_gates.yaml`.

## Current scenario set

The manifest currently registers 7 scenarios:

- 6 promoted gate scenarios
- 1 negative coverage scenario

| Scenario ID | Mode | Classification | Path | Storage feature policy |
| --- | --- | --- | --- | --- |
| `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest` | standalone | gate | `2.6.18 -> 3.0 latest -> 2.6 latest` | LoonFFI/storage v3 and Vortex must stay disabled. |
| `cluster-2-6-18-to-3-0-latest-rollback-2-6-latest` | cluster | gate | `2.6.18 -> 3.0 latest -> 2.6 latest` | LoonFFI/storage v3 and Vortex must stay disabled. |
| `standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | standalone | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` | LoonFFI/storage v3 and Vortex disabled. |
| `cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest -> 3.0 baseline` | LoonFFI/storage v3 and Vortex disabled. |
| `standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline` | standalone | gate | `3.0 baseline -> 3.0 latest + LoonFFI/Vortex -> 3.0 baseline + LoonFFI/Vortex` | Target and rollback both keep LoonFFI/storage v3 and Vortex enabled. |
| `cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline` | cluster | gate | `3.0 baseline -> 3.0 latest + LoonFFI/Vortex -> 3.0 baseline + LoonFFI/Vortex` | Target and rollback both keep LoonFFI/storage v3 and Vortex enabled. |
| `standalone-3-0-loon-vortex-to-2-6-negative` | standalone | negative | `2.6.18 -> 3.0 latest + LoonFFI/Vortex -> 2.6 latest` | Unsupported negative coverage only; not a promoted gate. |

For the 3.0 LoonFFI/Vortex gates, the rollback phase uses the 3.0 baseline
image but keeps LoonFFI/storage v3 and Vortex enabled. This validates image
rollback compatibility after the upgraded version has written data and indexes
with the 3.0 storage features enabled.

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

Cluster 3.0 LoonFFI/Vortex gate:

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline \
  --format argo-args \
  --allow-placeholder
```

`--allow-placeholder` is only for dry-run/review output. Before submitting a
formal gate, replace placeholder aliases such as `milvus-3-0-latest` and
`milvus-2-6-latest` with concrete image tags in
`upgrade_rollback_gates.yaml`.

## Submitting to Argo

Generate arguments, then submit in the 4am Argo namespace:

```bash
ARGS="$(PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline \
  --format argo-args)"

argo submit -n qa $ARGS
```

Do not pass `--allow-placeholder` for a real submit path. The renderer should
fail fast if any runnable image still contains `placeholder`.

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
- continuous pressure workload, with rollout maintenance windows only excluding
  confirmed connectivity failures.
