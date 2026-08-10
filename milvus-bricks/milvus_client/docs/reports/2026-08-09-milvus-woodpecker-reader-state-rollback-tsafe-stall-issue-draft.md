# Milvus Issue Draft: Woodpecker Reader State Loss Stalls Channel tSafe After Rollback

## Suggested Title

`[Bug]: Woodpecker reader state loss after a 3.0 rollback permanently stalls channel tSafe under DML`

## Environment

- Target Milvus image:
  `harbor.milvus.io/milvusdb/milvus:3.0-20260807-1439dc7d@sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64`
- Rollback Milvus image:
  `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
- Test repository commit:
  `a5d91d69d48672832b9bf075c1dee0ba0a664720`
- PyMilvus: `3.0.1`
- Deployment: Milvus cluster, one QueryNode, one DataNode, one MixCoord,
  one StreamingNode, and four Woodpecker Pods
- WAL: Woodpecker `v0.1.33`
- Storage V2 with JSON Shredding enabled only after the target rollout

All rollback Pods were `Running`, `Ready`, and had zero container restarts when
the serviceability failure was captured.

## Problem

With continuous insert/upsert/delete pressure, a `3.0.0 -> newer 3.0 ->
3.0.0` rollout can leave a subset of pre-existing collection channels
permanently stalled after rollback. Proxy reports a tSafe lag that keeps
growing beyond 16 minutes:

```text
lag(16m43.836s) max(3s): channel tsafe stalled[
  channel=by-dev-rootcoord-dml_14_468259685039343966v0
]
```

The affected collection remains loaded but count/query requests time out. Other
collections in the same deployment remain serviceable, so the failure is not a
general Proxy or QueryNode outage.

The StreamingNode logs on the same physical channels show that the Woodpecker
scanner cannot update or read its persisted reader state:

```text
reader temp info not found for logId:9
readerName:by-dev-rootcoord-dml_14-r-by-dev-rootcoord-dml_14/6/3-1786270658809831864

update reader info failed
pendingReadSegmentId=117 nextReadSegmentId=117

direct read batch failed
segId=117 from=0 error="no record extract"
```

The second stalled channel has the same pattern:

```text
logName=by-dev-rootcoord-dml_0
logId=16
reader temp info not found
pendingReadSegmentId=3524 nextReadSegmentId=3524
direct read batch failed error="no record extract"
```

MixCoord consequently keeps the delegator non-serviceable and cannot advance
the readable distribution version:

```text
delegator is not serviceable
channel=by-dev-rootcoord-dml_14_468259685039343966v0

before shard delegator update it's readable version, skip release segment
leaderVersion=1786270653123090701
currentVersion=1786270723116811700
```

## Reproduction

### Full-DML Reproduction

- Argo workflow: `pr25-cl30-json-r3-vw87h`
- Scenario:
  `cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`
- Base rows: 100 rows per rollback-safe collection
- Pressure modules:
  `search_pressure query_pressure query_iterator_scan count_pressure upsert_pressure delete_pressure mixed_rw_pressure`
- Target phase: enable JSON Shredding, continue pressure, and create the
  forward JSON collection
- Rollback phase: restore the pinned v3.0.0 image and wait for every baseline
  collection to answer count queries

The rollback readiness check and storage configuration assertion passed. The
serviceability brick then retried for 909.259 seconds and failed after 24
attempts:

```json
{
  "collection": "qa_gate_cluster_30_json_shredding_legacy_index_rollback_safe",
  "type": "QUERY_FAILED",
  "error": "lag(16m43.836s) max(3s): channel tsafe stalled"
}
```

Collection and channel mapping captured through PyMilvus before cleanup:

```text
qa_gate_cluster_30_json_shredding_legacy_index_rollback_safe
  collectionID=468259685039343966
  channel=by-dev-rootcoord-dml_14_468259685039343966v0

qa_gate_cluster_30_json_shredding_scalar_dynamic_partition_key
  collectionID=468259685039343507
  channel=by-dev-rootcoord-dml_0_468259685039343507v0
```

Direct `count(*)` probes with a 3-second timeout confirmed that both collections
timed out, while the other baseline collections, every target-created phase
collection, and the forward JSON collection still returned counts.

### Independent Reproduction

An earlier full-DML run, `pr25-cl30-json-r3-77cdt`, failed with the same tSafe
stall after rollback. In that run the first collection reported by the
serviceability brick was
`qa_gate_cluster_30_json_shredding_geometry_rtree_rollback_safe`, with a lag of
approximately 14 minutes.

The affected collection varies between runs. This rules out Geometry, JSON
Shredding, or a specific scalar/vector index as the sole trigger.

## Control Runs

### Read-Only Pressure Control

Workflow `pr25-cl30-json-ro-r1-r56q6` used the same images, deployment profile,
schema matrices, JSON Shredding rollout, rollback, and phase DML/DQL checks. It
removed only the continuous write pressure modules and kept:

```text
search_pressure query_pressure query_iterator_scan count_pressure
```

Result:

- workflow status: passed
- rollback serviceability: passed on the first attempt
- all data, index, JSON path, dynamic JSON, and phase DML/DQL checks: passed
- steady-state pressure: `452408 / 452408`, success rate `1.0`
- no `reader temp info not found`, `no record extract`, or tSafe stall logs

This isolates continuous cluster DML across the component rollouts as a
required trigger.

### Standalone Control

Workflow `pr25-st30-json-f5000-r1-c5kcs` ran 5000 rows with full write pressure
on standalone RocksMQ and passed upgrade, JSON Shredding enablement, rollback,
all validations, and steady-state pressure.

The failure is therefore specific to the cluster Woodpecker reader/recovery
path, not to the JSON collection format or generic rollback validation.

## Root-Cause Hypothesis

The evidence points to a Woodpecker reader lifecycle mismatch across the
Milvus component rollouts:

1. Full DML creates active reader progress and many WAL segments on baseline
   collection channels.
2. Target/config/rollback rollouts recreate StreamingNode and consumers.
3. A recreated scanner resumes with a reader name and pending segment whose
   temporary reader metadata no longer exists in Woodpecker metadata.
4. `ReadNext` repeatedly fails with `reader temp info not found` and then
   `no record extract` for the same pending segment.
5. The consumer checkpoint cannot advance, QueryNode tSafe remains frozen, and
   the delegator never becomes serviceable.

This is a root-cause hypothesis. The ownership of reader temp metadata between
Milvus StreamingNode and Woodpecker should be verified, especially around
scanner recreation, reader cleanup, and generation changes during a rollout.

## Expected Behavior

After a supported same-minor rollback, every loaded channel must resume WAL
consumption and advance tSafe. Missing temporary reader metadata must either be
reconstructed from durable progress or cause a bounded, explicit recovery
operation. It must not leave selected collections permanently unavailable while
all Pods remain ready.

## Requested Fix and Tests

1. Make Woodpecker reader progress needed for restart/rollout recovery durable,
   or recreate missing temporary reader metadata from the committed reader
   position.
2. Treat `reader temp info not found` as a recoverable scanner-state transition
   instead of repeatedly retrying the same unreadable segment.
3. Add structured logs and metrics for reader generation, committed position,
   pending segment, metadata cleanup, and recreation decisions.
4. Add an integration test that continuously inserts, upserts, and deletes on
   multiple loaded collections while StreamingNode/QueryNode are rolled through
   upgrade, runtime config rollout, and rollback.
5. Assert that every channel advances tSafe and all collections answer count,
   query, and search requests after rollback.
6. Add a health signal for a loaded channel whose tSafe has not advanced for a
   bounded interval, even when all Kubernetes workloads are Ready.
