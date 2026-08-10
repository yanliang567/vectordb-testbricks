# Milvus Issue Draft: Vortex Variant Encoding Breaks 3.0 Rollback

## Disposition Update

Milvus issue [#52340](https://github.com/milvus-io/milvus/issues/52340) was
submitted from this draft. Product follow-up on 2026-08-10 established that
v3.0.0 is not recommended or supported with Vortex enabled; supported rollback
coverage starts at v3.0.1. The reproduction below remains valid historical
evidence for that boundary. The test repository now rejects v3.0.0 Vortex
release gates and uses reviewed Vortex 0.75 branch images only as pre-release
candidate coverage until a pinned v3.0.1 release image is available.

## Suggested Title

`[Bug]: v3.0.0 cannot load Vortex segments written by a newer 3.0 build because vortex.variant is not registered`

## Environment

- Target Milvus image:
  `harbor.milvus.io/milvusdb/milvus:3.0-20260807-1439dc7d@sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64`
- Target source commit from the image tag: `1439dc7de8b198a01c2afa0ae20c0c473e0e1abc`
- Rollback Milvus image:
  `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
- Rollback source tag commit: `f46a0328558be155d11266a1a2b90602ccc9b366`
- PyMilvus: `3.0.1`
- Deployments reproduced:
  - Milvus Operator standalone with RocksMQ
  - Milvus Operator cluster with Woodpecker WAL
- Storage configuration in target and rollback phases:

  ```yaml
  common:
    storage:
      useLoonFFI: true
  dataNode:
    storage:
      format: vortex
  ```

## Problem

After the target build writes and flushes data with Vortex enabled, rolling
back to the v3.0.0 baseline leaves the Milvus CR and Pod healthy but the
collections never become serviceable. The v3.0.0 QueryNode cannot load the
target-written segments because its Vortex registry does not contain the
encoding used by the newer build:

```text
IOError: Failed to open vortex file: Registry missing encoding with id vortex.variant
```

The segment load is retried continuously. QueryCoord cannot establish shard
leaders, and client queries fail with:

```text
failed to query: no available shard leaders: channel not available
```

The Milvus CR simultaneously reports `Healthy`, `MilvusReady=True`, and the
expected v3.0.0 image. Deployment readiness therefore does not expose the data
format incompatibility.

## Upgrade/Rollback Reproduction

- Argo workflow: `pr25-st30-loon-r4-jsw7x`
- Diagnostic workflow: `pr25-loon-diag-vkmnd`
- Cluster Argo workflow: `pr25-cl30-loon-r1-jfkzb`
- Cluster diagnostic workflow: `pr25-cl-loon-diag-5skkn`
- Test repository commit: `8bb689c89aaad42e41b8bfe764ae1641163f21dc`
- Scenario:
  `standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline`
- Seed scale: 100 rows per base collection and 100 rows in the Storage V3
  forward collection
- Target and rollback both enabled LoonFFI and Vortex
- All target-phase data, index, schema-feature, DML/DQL, runtime-config, and
  strict-pressure validations passed
- Failure phase: rollback serviceability wait

The v3.0.0 Pod repeatedly logged:

```text
failed to load some segments
At LoadSegment: At Load: Assert "chunk_reader_result.ok()"
get chunk reader failed, segment 468257428861791175, column group index 0
IOError: Failed to open vortex file: Registry missing encoding with id vortex.variant
```

The same error was emitted from both growing and sealed segment load paths,
including `SegmentGrowingImpl.cpp` and `ChunkedSegmentSealedImpl.cpp`.

The cluster reproduction also showed the v3.0.0 DataNode retrying failed mix
compactions once per minute:

```text
compact wrong, fail to merge sort segments
failed to get record batch reader
IOError: Failed to open vortex file: Registry missing encoding with id vortex.variant
```

MixCoord recorded persistent failed segment-load tasks for multiple base and
forward collections, while QueryNode failed both growing and sealed loads with
the same missing encoding. This confirms the incompatibility is not specific
to standalone mode, RocksMQ, or a single query path.

## Expected Behavior

For a supported same-minor rollback path, one of the following must hold:

1. The newer 3.0 target writes a Vortex format and encoding set readable by
   the pinned 3.0 rollback baseline.
2. The rollback baseline contains compatible decoders for every encoding the
   supported target may persist.
3. Milvus rejects the incompatible rollout configuration before target data is
   written and clearly reports that rollback is unsupported.

It must not report the deployment healthy while all affected collections are
permanently unavailable.

## Root-Cause Hypothesis

The newer target selects the `vortex.variant` encoding when serializing one or
more column groups. The v3.0.0 binary links an older Vortex implementation or
registry that does not register that encoding ID. The metadata and object files
remain discoverable, but the baseline fails while constructing the chunk
reader and therefore never loads the segment.

This appears to be a persisted-format compatibility issue rather than an
Operator rollout race: the v3.0.0 Pod was ready, the CR observed generation 5,
the expected image and storage flags were active, and the same segment load
error continued across retries.

## Requested Fix and Tests

1. Define and enforce the supported Vortex persisted-format compatibility
   contract for 3.0 patch-to-patch rollback.
2. Add format-version or required-encoding metadata so an incompatible reader
   reports a direct compatibility error instead of leaving collections without
   shard leaders.
3. Backport the `vortex.variant` decoder to supported rollback baselines, or
   make newer writers use a baseline-readable encoding while rollback support
   is required.
4. Add an E2E test that writes and flushes nullable/variant-capable fields with
   the latest 3.0 build, rolls back to v3.0.0 with Vortex still enabled, loads
   every collection, and performs count/query/search assertions.
5. Make health reporting account for persistent segment-load failures that
   prevent all shard leaders from becoming available.
