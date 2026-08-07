# Upgrade/Rollback Availability Repeat Calibration Report

Date: 2026-08-07

## Scope

Repeat the Woodpecker 2CU HA upgrade/rollback gate with reproducible repository
and Milvus image inputs before enabling the candidate availability SLO. These
runs remain observational and do not change the existing correctness,
serviceability, or strict pressure gates.

## Test Contract

- Repository: `https://github.com/yanliang567/vectordb-testbricks.git`
- WorkflowTemplate: `milvus-cluster-upgrade-rollback`
- Scenario ID:
  `cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline`
- Deploy profile:
  `milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml`
- Base and rollback image:
  `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
- Runner: `harbor.milvus.io/qa/fouram:2.1`
- SDK: `pymilvus==3.0.0`
- Topology: 2 Proxy, 2 QueryNode, 2 DataNode, 2 StreamingNode, 1 MixCoord,
  Woodpecker storage
- LoonFFI, Vortex, and JSON Shredding: disabled in every phase

## Run Matrix

| Workflow | Repository revision | Target image | Status | Duration | Calibration eligibility |
| --- | --- | --- | --- | --- | --- |
| `c30-2cu-ha-7jf5g` | `efef556c96a570f01dd5737807d5cb8fc430e270` | `3.0-20260806-bb41eb52@sha256:56638970...d83bd6` | Failed at rollback serviceability | 50m35s | Ineligible |
| `c30-2cu-ha-q8rn2` | `d178bb5dc33534068ff6fbc3222bf54679998f50` | `3.0-20260807-697431f2@sha256:e29d3275...d6db5a3` | Failed at rollback serviceability | 49m18s | Ineligible |

Both runs used full repository commit SHAs. The second revision is the merge
commit of PR #23 and includes the reviewed commit-SHA checkout path.

## Common Gate Result

Both runs completed the following required checks before rollback:

- base deployment, configuration assertion, schema creation, seed data, and
  checkpoint validation;
- strict pressure before upgrade;
- image rollout to the target and runtime version/configuration validation;
- post-upgrade data serviceability and integrity validation;
- index metadata compatibility;
- phase DML/DQL validation and existing-collection schema evolution;
- strict pressure after upgrade and before rollback.

The rollback Helm rollout and readiness checks also completed. The first
baseline count/PK serviceability probe after rollback then failed.

## Rollback Serviceability Failure

`c30-2cu-ha-7jf5g` recorded 12 query failures across three collections.
`c30-2cu-ha-q8rn2` recorded 16 query failures across four collections. Every
failure was Milvus code 505 with `channel tsafe stalled`; observed channel lag
in the second run ranged from about 5 to 13 minutes.

Example:

```text
failed to search/query delegator 21 for channel ...:
LBPolicyImpl.Execute: lag(10m7.228s) max(3s):
channel tsafe stalled[channel=by-dev-rootcoord-dml_0_...]
```

The failure reproduced with two different target builds while the exact same
`v3.0.0` manifest digest was used for base and rollback. An earlier calibration
using recent 3.0 branch builds for both base and rollback succeeded, so these
runs expose a rollback compatibility/service-recovery difference rather than
normal availability variance.

Milvus issue: https://github.com/milvus-io/milvus/issues/52297

## Test Harness Finding

The serviceability classifier recognized unavailable channel distribution and
missing shard leaders, but not `channel tsafe stalled`. It therefore treated
the error as a non-transient correctness failure after one attempt. That first
attempt also continued probing every collection and PK sample, accumulating
multiple approximately 28-second RPC timeouts instead of using the configured
900-second recovery window efficiently.

The follow-up change classifies only the explicit tSafe-stalled query error as
transient and ends an attempt after its first transient RPC failure. Count
drift, missing PK, checksum mismatch, and an unrecovered serviceability timeout
remain hard failures.

## Cleanup

Both `onExit` reports recorded:

- `cleanup_attempted=true`
- `cleanup_status=completed`
- empty `cleanup_error`
- empty `kept_resources`

The second run was also checked by workflow UID after completion; no owned
Pods, Services, Deployments, StatefulSets, PVCs, or pressure ConfigMaps
remained.

## Decision

Do not add these runs to the availability baseline and do not enable the hard
SLO. They ended before the post-rollback observation and final pressure
aggregation, so they are not complete calibration samples.

Keep the candidate Woodpecker 2CU HA policy observational until:

1. the bounded tSafe serviceability retry is runtime-validated;
2. the Milvus rollback issue is fixed or explicitly dispositioned;
3. at least two additional complete 2CU runs establish rollout variance with
   the pinned `v3.0.0` baseline.
