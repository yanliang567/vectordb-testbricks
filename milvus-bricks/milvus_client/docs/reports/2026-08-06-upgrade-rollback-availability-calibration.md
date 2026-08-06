# Upgrade/Rollback Availability Calibration Report

Date: 2026-08-06
Branch: `feat/availability-slo-calibration`

## Scope

Calibrate the observational upgrade/rollback availability metrics added in
PR #20 with real 4am cluster workflows. Availability remains non-gating while
the runs establish representative rollout and steady-state baselines.

Planned scenarios:

1. Woodpecker 2CU HA 3.0 upgrade and rollback.
2. Cluster 2.6 to 3.0 target-only feature upgrade and 2.6 rollback.
3. Cluster 3.0 JSON Shredding post-upgrade rollout and rollback.

All formal runs use concrete Harbor build tags, keep correctness,
serviceability, index, phase DML/DQL, and strict pressure gates enabled, and
leave `availability.gate_enforced=false`.

## Run Matrix

| Scenario | Workflow | Status | Duration | Availability result |
| --- | --- | --- | --- | --- |
| Woodpecker 2CU HA | `c30-2cu-ha-cjbmv` | Succeeded | 54m46s | Complete; 2 failed requests during upgrade rollout, zero during rollback |
| Cluster target-only | `c26to-sl6ds` | Failed | 51m51s | Complete; count aggregation failed during both rollouts, zero steady-state failures |
| Cluster JSON Shredding | `c30json-hxmjd` | Failed | 1h00m24s | Complete; closed-channel failures during rollouts and four count failures after rollback-ready |

## Woodpecker 2CU HA

Workflow: `c30-2cu-ha-cjbmv`

Images:

- Base/rollback: `harbor.milvus.io/milvusdb/milvus:3.0-20260805-ad3ba1ea`
- Target: `harbor.milvus.io/milvusdb/milvus:3.0-20260806-87ea4cac`
- Runner: `harbor.milvus.io/qa/fouram:2.1`

Effective data-plane topology:

- Proxy: 2 replicas
- QueryNode: 2 replicas
- DataNode: 2 replicas
- StreamingNode: 2 replicas
- Woodpecker: 4 Pods

Gate result:

- Argo status: `Succeeded`, 52/52 nodes completed.
- Correctness, serviceability, index compatibility, phase DML/DQL, schema
  evolution, and strict pressure checks passed.
- Workflow-owned Milvus resources, PVCs, Helm release secrets, and pressure
  ConfigMaps were independently verified as removed after `onExit`.

Availability summary:

| Scope | Samples | Operations | Failed requests | Success rate | Failure span | Impacted bricks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Overall | 186 | 1,043,178 | 2 | 0.999998 | 6.615528s | `delete_pressure` |
| Steady state | 150 | 837,716 | 0 | 1.0 | 0s | None |
| Upgrade rollout | 19 | 99,220 | 2 | 0.99998 | 6.615528s | `delete_pressure` |
| Rollback rollout | 17 | 106,242 | 0 | 1.0 | 0s | None |

Completeness:

- `incomplete_sample_count=0`
- `unassigned_sample_count=0`
- `complete=true`
- `calibration_eligible=true`

The two upgrade-window delete failures were in one pressure slice. Both were
Milvus error code 901 from `syncTimestamp`, with no available MixCoord while
the MixCoord deployment was rolling. No failures continued into post-upgrade
steady state, and the rollback rollout had no failed requests.

## Cluster Target-Only

Workflow: `c26to-sl6ds`

Images:

- Base: `harbor.milvus.io/milvusdb/milvus:v2.6.18`
- Target: `harbor.milvus.io/milvusdb/milvus:3.0-20260806-87ea4cac`
- Rollback: `harbor.milvus.io/milvusdb/milvus:2.6-20260805-4c2ef608`
- Runner: `harbor.milvus.io/qa/fouram:2.1`

Gate result:

- Argo status: `Failed` at `gate-final-status` because four non-excluded
  pressure result slices failed.
- Baseline data integrity, index compatibility, phase DML/DQL, and post-upgrade
  and post-rollback serviceability checks passed.
- The target-only 3.0 forward schema, data, indexes, filters, and schema
  evolution checks passed on the target version.
- After rollback, the workflow correctly skipped forward collection
  validation and revalidated only the 2.6 baseline contract. Baseline data
  written and updated during the target phase remained usable.
- Workflow-owned Milvus resources, PVCs, Helm release secrets, and pressure
  ConfigMaps were independently verified as removed after `onExit`.

Availability summary:

| Scope | Samples | Operations | Failed requests | Success rate | Failure span | Impacted bricks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Overall | 174 | 696,750 | 177 | 0.999746 | 1,384.940361s | `count_pressure`, `mixed_rw_pressure` |
| Steady state | 153 | 645,084 | 0 | 1.0 | 0s | None |
| Upgrade rollout | 11 | 26,428 | 18 | 0.999319 | 114.600232s | `count_pressure`, `mixed_rw_pressure` |
| Rollback rollout | 10 | 25,238 | 159 | 0.9937 | 58.513378s | `count_pressure`, `mixed_rw_pressure` |

Completeness:

- `incomplete_sample_count=0`
- `unassigned_sample_count=0`
- `complete=true`
- `calibration_eligible=true`

One upgrade-window upsert failure reporting no available MixCoord was excluded
by the existing rollout service-switch rule. The remaining failures were all
count aggregation errors, including `internal count result should only have
one column` and `reduce_by_groups` results with no field data. These are not
connection handoff failures and correctly remained strict pressure failures.

The overall `failure_span_sec` is the envelope from the first upgrade failure
to the last rollback failure; it does not represent a continuous outage. Both
rollout windows recovered to zero failed requests in their following
steady-state samples.

## Cluster JSON Shredding

Workflow: `c30json-hxmjd`

Images:

- Base/rollback: `harbor.milvus.io/milvusdb/milvus:3.0-20260805-ad3ba1ea`
- Target: `harbor.milvus.io/milvusdb/milvus:3.0-20260806-87ea4cac`
- Runner: `harbor.milvus.io/qa/fouram:2.1`

Gate result:

- Argo status: `Failed` at `gate-final-status` because three non-excluded
  pressure result slices failed.
- Baseline data integrity, index compatibility, phase DML/DQL, and
  serviceability checks passed before and after rollback.
- The post-upgrade Helm rollout enabled JSON Shredding and passed the runtime
  configuration assertion before forward data creation.
- JSON-heavy forward data, selected dynamic JSON keys, nested JSON filters,
  JSON path indexes, and index compatibility passed after upgrade and again
  after rollback.
- Workflow-owned Milvus resources, PVCs, Helm release secrets, and pressure
  ConfigMaps were independently verified as removed after `onExit`.

Availability summary:

| Scope | Samples | Operations | Failed requests | Success rate | Failure span | Impacted bricks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Overall | 202 | 879,171 | 695 | 0.999209 | 1,919.923454s | `count_pressure`, `query_iterator_scan`, `query_pressure`, `upsert_pressure` |
| Steady state | 151 | 685,057 | 4 | 0.999994 | 48.706234s | `count_pressure` |
| Upgrade rollout | 11 | 41,547 | 4 | 0.999904 | 101.927453s | `upsert_pressure` |
| Post-upgrade config rollout | 31 | 122,351 | 685 | 0.994401 | 1.664912s | `query_pressure` |
| Rollback rollout | 9 | 30,216 | 2 | 0.999934 | 11.645135s | `query_iterator_scan` |

Completeness:

- `incomplete_sample_count=0`
- `unassigned_sample_count=0`
- `complete=true`
- `calibration_eligible=true`

The upgrade rollout had one excluded MixCoord handoff failure and three
non-excluded `Cannot invoke RPC on closed channel` upsert failures. The
post-upgrade configuration rollout had 685 query failures with the same closed
channel error in one result slice. The rollback rollout had two excluded
QueryNode connection failures.

Four count requests started 14 seconds after `wait-rollback-ready` completed
and failed for about 49 seconds because the channel distribution was not yet
serviceable. They were therefore classified as steady-state failures instead
of rollout failures. This indicates that CR/Helm readiness and serviceability
readiness can diverge after a rolling rollback.

## Non-Calibrating Attempt

Workflow `c30-2cu-ha-dgg5z` failed during base deployment because the manifest
alias `milvus-3-0-baseline` still referenced removed Harbor tag
`3.0-20260723-77b26a50`. It produced no availability baseline and is excluded
from calibration. This also demonstrates that formal runs must override stale
aliases with existing concrete tags until the manifest alias is updated in a
separate change.

## Calibration Decision

Do not enable a hard availability SLO yet. The three scenarios demonstrate
that one global threshold would mix materially different rollout behavior:

| Scenario | Worst rollout success rate | Longest rollout failure span | Steady-state failed requests |
| --- | ---: | ---: | ---: |
| Woodpecker 2CU HA | 0.99998 | 6.615528s | 0 |
| Cluster target-only | 0.9937 | 114.600232s | 0 |
| Cluster JSON Shredding | 0.994401 | 101.927453s | 4 |

The first candidate policy for the Woodpecker 2CU HA promoted gate is:

- require `complete=true`, `calibration_eligible=true`, and zero unassigned
  samples;
- require zero failed requests in steady state;
- require each image rollout to have `success_rate >= 0.9999`;
- require each image rollout to have `failure_span_sec <= 15`;
- keep existing strict pressure and correctness gates independent and
  unchanged.

This candidate fits the one 2CU HA run and rejects the two degraded 1CU runs,
but it must not be enforced until repeated 2CU runs establish variance across
different builds and cluster load.

The current 10-second pressure slice setting does not provide a strict
request-level window boundary. A result slice that overlaps a rollout is
counted wholly inside that rollout, and RPC timeouts can make a nominal
10-second slice last substantially longer. `failure_span_sec`, which uses
per-failure timestamps, is more precise than rollout `success_rate`, but both
should remain observational until repeated runs or time-bucketed operation
metrics confirm the threshold.

`overall.failure_span_sec` must not be used as a gate because it is the envelope
between failures in separate rollout windows. Post-upgrade configuration
rollouts also need a separate policy from image rollouts.
