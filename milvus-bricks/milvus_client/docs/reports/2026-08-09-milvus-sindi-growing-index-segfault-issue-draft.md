# Milvus Issue Draft: SINDI Growing Index Build Failure Crashes QueryNode

## Suggested Title

`[Bug]: SINDI growing sparse index on index version 8 crashes QueryNode after reporting unsupported algorithm`

## Environment

- Milvus image:
  `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
- Milvus source tag commit: `f46a0328558be155d11266a1a2b90602ccc9b366`
- PyMilvus: `3.0.1`
- Deployment: Milvus Operator cluster, Woodpecker WAL, one QueryNode
- Configured target versions:
  `dataCoord.targetVecIndexVersion=10` and
  `dataCoord.targetScalarIndexVersion=4`
- Sparse index: `SPARSE_INVERTED_INDEX`
- Sparse algorithm: `SINDI`
- Metric: `IP`

## Problem

After a loaded collection with a SINDI sparse index receives enough writes to
build a growing-segment sparse index, QueryNode first reports that SINDI is not
supported by index version 8 and then terminates with `SIGSEGV`:

```text
[KNOWHERE][ValidateInvertedIndexAlgo] Unsupported sparse inverted index
algorithm SINDI for index version 8

[SERVER][AppendSegmentIndexSparse] growing sparse index build error:
failed to build index, knowhere inner error at
../internal/core/src/index/VectorMemIndex.cpp:501

SIGNAL CATCH BY NON-GO SIGNAL HANDLER
SIGNO: 11; SIGNAME: Segmentation fault; SI_CODE: 1; SI_ADDR: (nil)

knowhere::sparse::inverted::GrowableInvertedIndex<float, float>::add(...)
```

The Pod restarts repeatedly. During the restart loop, Proxy operations fail
with errors including:

```text
complexDelete: node not found
channel tsafe stalled
dial tcp <query-node-ip>:21123: connect: connection refused
```

An unsupported algorithm/version combination must be rejected or handled as a
normal index-build error. It must not crash QueryNode.

## Upgrade/Rollback Gate Reproduction

- Argo workflow: `pr25-cl30-idx104-r1-gptfx`
- Test repository commit: `0914249b78f92e3346084c7892ebb9cee3c5aab6`
- Scenario: `cluster-3-0-index-v10-v4-upgrade-rollback`
- Failed node: `strict-pressure-before-upgrade`
- Failure phase: v3.0.0 baseline, before any upgrade
- Seed rows: 100 per collection
- Pressure modules included insert, upsert, delete, search, query, and count

Before write pressure, all of these steps passed:

- collection and SINDI index creation
- seed and flush
- data-integrity validation
- SINDI search validation
- schema-feature validation

The crash began after pressure created a growing segment with approximately
6,680 sparse-vector rows. The QueryNode log showed:

```text
origin_index_type=SPARSE_INVERTED_INDEX
inverted_index_algo=SINDI
use key SPARSE_INVERTED_INDEX_CC_sparse_u32_f32 ... with version 8
Unsupported sparse inverted index algorithm SINDI for index version 8
```

## Additional Server Evidence

MixCoord accepted the user index definition:

```text
IndexParams:
  index_type=SPARSE_INVERTED_INDEX
  inverted_index_algo=SINDI
  metric_type=IP
```

The Milvus CR contained:

```yaml
dataCoord:
  targetVecIndexVersion: 10
  targetScalarIndexVersion: 4
```

However, the growing QueryNode index factory still selected Knowhere index
version 8. The version/configuration boundary is therefore not enforced before
the growing-index path attempts to build SINDI.

## Source Analysis and Root-Cause Hypothesis

In the v3.0.0 source, `VectorFieldIndexing::AppendSegmentIndexSparse()` calls
`BuildWithDataset()`. On `SegcoreError`, it logs the failure, calls
`recreate_index(...)`, and returns. The crash backtrace is in an asynchronous
Knowhere `GrowableInvertedIndex::add()` task immediately after the failed
build.

This is consistent with an error-path lifetime/race defect: the unsupported
build causes the sparse index object to be recreated while an asynchronous add
callback still references the failed or replaced index. This is a hypothesis;
the exact ownership transition should be confirmed with ASAN or a focused C++
test.

## Expected Behavior

One of the following is acceptable:

1. Reject SINDI at `create_index` when the effective growing index version
   cannot support it.
2. Defer growing-index construction and continue serving from raw growing
   data.
3. Return a controlled build/load error and mark the segment unavailable.

In all cases, QueryNode must remain alive and serviceable.

## Requested Fix and Tests

1. Validate SINDI against the effective growing-index version before starting
   the Knowhere build.
2. Make the sparse growing-index failure path ownership-safe; do not recreate
   or destroy an index while asynchronous build/add work can still access it.
3. Add a C++ regression test for `SINDI + effective index version 8` that
   asserts a normal error and no process crash.
4. Add an E2E test that creates and loads a SINDI collection, inserts enough
   rows to cross the growing-index build threshold, and verifies QueryNode
   remains healthy.

