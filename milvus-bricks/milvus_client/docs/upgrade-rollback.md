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
