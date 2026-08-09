# Milvus Upgrade/Rollback Type and Index Coverage Execution Report

Date: 2026-08-09

## Executive Summary

PR `yanliang567/vectordb-testbricks#25` was reviewed, corrected, and exercised
against real Milvus deployments on the QA Kubernetes cluster. The runtime code
under test was commit:

```text
a5d91d69d48672832b9bf075c1dee0ba0a664720
```

The test framework behavior is accepted. It now distinguishes passing
upgrade/rollback coverage from Milvus product blockers without suppressing or
misclassifying failures.

Confirmed passing coverage:

- 5000-row `2.6.18 -> newer 3.0 -> latest 2.6` standalone rollback-safe gate
- 5000-row cluster index engine version `10/4` gate
- 5000-row standalone JSON Shredding gate with full read/write pressure
- cluster JSON Shredding gate with read-only continuous pressure and explicit
  phase DML/DQL
- data integrity, index metadata, filter/search execution, phase DML/DQL,
  serviceability, and steady-state pressure reporting

Confirmed Milvus blockers:

- StructArray `FLOAT16_VECTOR + DISKANN + MAX_SIM_COSINE` returns a negative
  exact self-similarity score
- SINDI growing sparse index failure can crash QueryNode
- newer Vortex `vortex.variant` files cannot be read by the pinned v3.0.0
  rollback image
- cluster Woodpecker reader state can be lost across rollout/rollback under
  continuous DML, permanently stalling selected channel tSafe values

The blockers remain strict failures. They were not converted to warnings or
excluded from the corresponding feature contract.

## Pull Request and Environment

- Pull request: `https://github.com/yanliang567/vectordb-testbricks/pull/25`
- Branch: `feat/upgrade-rollback-type-index-coverage`
- Code-under-test commit: `a5d91d69d48672832b9bf075c1dee0ba0a664720`
- PyMilvus: `3.0.1`
- Kubernetes namespace: `qa-milvus`
- Argo namespace: `qa`

Milvus images:

| Phase | Image |
|---|---|
| 2.6.18 base | `harbor.milvus.io/milvusdb/milvus:v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6` |
| newer 3.0 target | `harbor.milvus.io/milvusdb/milvus:3.0-20260807-1439dc7d@sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64` |
| latest 2.6 rollback | `harbor.milvus.io/milvusdb/milvus:2.6-20260807-d85dc945@sha256:2051a754368d70f589a281fa301a12128d058e531bd6e5d82583e588bccd961e` |
| 3.0 baseline/rollback | `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862` |

## Workflow Result Matrix

| Coverage | Workflow | Scale | Result | Conclusion |
|---|---|---:|---|---|
| 2.6 rollback-safe types/indexes | `pr25-st26-current-r1-g7wxd` | 5000 rows/schema | Passed | Current PR SHA, pinned images, full pressure, DML/DQL, and rollback reuse passed. |
| Index engine v10/v4 | `pr25-cl30-idx104-f5000-r1-ff9dq` | 5000 rows/schema | Passed | SINDI/Block-Max and JSON scalar index files were built with versions 10/4 and reused after rollback. |
| JSON Shredding standalone | `pr25-st30-json-f5000-r1-c5kcs` | 5000 rows | Passed | Nested/dynamic JSON, six indexes, full pressure, and rollback reuse passed. |
| JSON Shredding cluster read-only control | `pr25-cl30-json-ro-r1-r56q6` | 100 rows/schema | Passed | Same config rollout and phase DML/DQL pass without continuous write pressure. |
| JSON Shredding cluster full DML | `pr25-cl30-json-r3-77cdt` | 100 rows/schema | Failed | Rollback channel tSafe stalled for about 14 minutes. |
| JSON Shredding cluster full DML reproduction | `pr25-cl30-json-r3-vw87h` | 100 rows/schema | Failed | Reproduced a 16+ minute tSafe stall with matching Woodpecker reader-state errors. |
| StructArray FLOAT16 DISKANN | `pr25-st30-forward5000-r1-tl6pm` | 5000 rows | Failed | Exact self-query returned approximately `-1.0` for MAX_SIM_COSINE. |
| SINDI growing index | `pr25-cl30-idx104-r1-gptfx` | write pressure | Failed | QueryNode crashed after an unsupported growing index version path. |
| Loon/Vortex standalone | `pr25-st30-loon-r4-jsw7x` | 100 rows/schema | Failed | v3.0.0 rollback cannot decode `vortex.variant`. |
| Loon/Vortex cluster | `pr25-cl30-loon-r1-jfkzb` | 100 rows/schema | Failed | Same persisted-format incompatibility reproduced with Woodpecker. |

## Passing Evidence

### 2.6.18 to 3.0 to Latest 2.6

Workflow `pr25-st26-current-r1-g7wxd` used the exact PR commit and all three
immutable image references.

Result summary:

```text
status: passed
rows per collection: 5000
collections checked after rollback: 7
actual indexes checked after rollback: 47
vector searches after rollback: 29
scalar index queries after rollback: 18
indexes rebuilt: 0
indexes dropped: 0
```

Feature semantics after rollback:

```text
StructArray scalar round-trip checks: 3 passed
StructArray element search checks: 1 passed
nullable vector type checks: 6 passed
Geometry filters: ST_EQUALS and ST_DWITHIN passed
```

The six nullable vector checks cover:

- `FLOAT_VECTOR`
- `FLOAT16_VECTOR`
- `BFLOAT16_VECTOR`
- `INT8_VECTOR`
- `BINARY_VECTOR`
- `SPARSE_FLOAT_VECTOR`

The rollback-safe StructArray contains FLOAT, VARCHAR, INT64, BOOL, and nested
FLOAT_VECTOR values. Its scalar values survived upgrade and rollback, and the
nested vector search returned the expected PK/offset contract.

Phase DML/DQL after rollback:

```text
existing collections: 7
rows inserted into existing collections: 7000
rows upserted: 6000
rows deleted: 700
new collections: 7
rows inserted into new collections: 21000
checkpoint searches validated: 56
total phase searches: 84
```

Pressure result:

```text
steady-state operations: 377631 / 377631
steady-state success rate: 1.0
strict failures outside maintenance windows: 0
```

Upgrade and rollback serviceability both passed on the first attempt.

### Index Engine Versions 10 and 4

Workflow `pr25-cl30-idx104-f5000-r1-ff9dq` passed on the current PR commit.

The final report confirmed:

```text
collections checked: 2
actual indexes checked: 9
scalar index queries: 4
vector/sparse searches: 5
indexes rebuilt: 0
indexes dropped: 0
steady-state operations: 345140 / 345140
```

DataNode and MixCoord logs captured real 5000-row builds:

```text
DataNode building index ... numRows=5000 current_index_version=10
Successfully prepare indexBuildTask ...
  currentIndexVersion=10 currentScalarIndexVersion=4
Successfully build index ... currentIndexVersion=10
```

The logs include successful builds for:

- `SINDI`
- `BLOCK_MAX_MAXSCORE`
- `BLOCK_MAX_WAND`
- JSON `BITMAP`
- JSON `STL_SORT`
- JSON `NGRAM`
- dense `HNSW`

The JSON user-facing AutoIndex was resolved internally to `HYBRID`:

```text
UserIndexParams.index_type=AUTOINDEX
IndexParams.index_type=HYBRID
json_path=json_auto['score']
json_cast_type=DOUBLE
```

After rollback, all data, metadata, scalar filters, sparse searches, and phase
DML/DQL checks passed without rebuilding the indexes.

Continuous pressure was intentionally read-only for this scenario because the
full write path independently reproduces the SINDI growing-index QueryNode
crash. The phase DML/DQL checks still performed deterministic insert, upsert,
delete, query, and search operations.

### JSON Shredding Standalone

Workflow `pr25-st30-json-f5000-r1-c5kcs` passed with full read/write pressure.

The forward checkpoint checksum includes declared and dynamic JSON fields:

```text
id
tenant
category
json_profile
json_nested
tags
dyn_bucket
dyn_text
dyn_json
```

After rollback:

```text
checkpoint rows: 5000
actual indexes checked: 6
scalar index queries: 5
vector searches: 1
indexes rebuilt: 0
indexes dropped: 0
```

The preserved index metadata includes both nested JSON paths:

```text
json_nested['nested']['score'] -> INVERTED, DOUBLE
json_profile['bucket'] -> INVERTED, DOUBLE
```

It also includes tenant/category/tags scalar indexes and the vector HNSW index.
All data, index, runtime config, phase DML/DQL, forward rollback, and
serviceability results passed.

Pressure result:

```text
steady-state operations: 469844 / 469844
steady-state success rate: 1.0
strict failures outside maintenance windows: 0
```

### JSON Shredding Cluster Isolation

The first full-DML cluster run failed after rollback. To separate JSON data
compatibility from cluster WAL recovery, workflow `pr25-cl30-json-ro-r1-r56q6`
kept the same images, schema matrices, JSON Shredding config rollout, rollback,
and phase DML/DQL checks, but limited continuous pressure to search, query,
iterator, and count.

It passed every validation:

```text
rollback serviceability: first attempt
forward rollback serviceability: first attempt
steady-state operations: 452408 / 452408
steady-state success rate: 1.0
```

This proves the JSON data and index formats are compatible with the tested
rollback. The remaining cluster failure requires continuous DML and is tracked
as a Woodpecker reader recovery issue.

## Confirmed Product Blockers

### StructArray FLOAT16 DISKANN

An exact stored FLOAT16 vector searched through StructArray DISKANN returns the
correct PK but a MAX_SIM_COSINE score close to `-1.0`. The defect reproduces on
the target before rollback, so it is not caused by rollback metadata or the
workflow.

Issue draft:
[2026-08-09-milvus-struct-array-diskann-max-sim-negative-score-issue-draft.md](2026-08-09-milvus-struct-array-diskann-max-sim-negative-score-issue-draft.md)

### SINDI Growing Index QueryNode Crash

The configured sealed index builds use versions 10/4, but QueryNode's growing
SINDI path selects index version 8, reports the algorithm as unsupported, and
can terminate with `SIGSEGV`. The test framework therefore keeps the SINDI
feature strict but uses read-only continuous pressure for the promoted index
version gate until Milvus is fixed.

Issue draft:
[2026-08-09-milvus-sindi-growing-index-segfault-issue-draft.md](2026-08-09-milvus-sindi-growing-index-segfault-issue-draft.md)

### Vortex Persisted-Format Rollback

The newer 3.0 target writes `vortex.variant`; v3.0.0 lacks the corresponding
decoder. Standalone and cluster rollback both leave collections without shard
leaders while the deployment still appears healthy.

Issue draft:
[2026-08-09-milvus-vortex-variant-rollback-incompatibility-issue-draft.md](2026-08-09-milvus-vortex-variant-rollback-incompatibility-issue-draft.md)

### Woodpecker Reader State and tSafe Stall

Two independent full-DML cluster JSON runs failed after rollback. The second
run captured matching evidence across layers:

- Proxy: two channels stalled with more than 16 minutes of tSafe lag
- MixCoord: delegator not serviceable and readable version not advanced
- StreamingNode/Woodpecker: `reader temp info not found` and
  `no record extract` for the exact stalled physical logs
- client probe: only the affected base collections timed out; other base,
  phase-created, and forward collections remained queryable
- all current Pods: Ready with zero restarts

The read-only pressure control passed and emitted none of these reader errors.

Issue draft:
[2026-08-09-milvus-woodpecker-reader-state-rollback-tsafe-stall-issue-draft.md](2026-08-09-milvus-woodpecker-reader-state-rollback-tsafe-stall-issue-draft.md)

## Fixes and Optimizations Made During Execution

Real execution exposed and drove the following PR improvements:

- preserved pressure error details instead of reporting only brick-level
  failures
- separated maintenance-window failures from steady-state gate failures
- made approximate MinHash recall observational while keeping exact
  self-search strict; ranking remains strict only when both the near-duplicate
  and unrelated rows are returned
- made phase searches filter and assert the phase-written PK, StructArray
  offset, and self-search score/distance instead of accepting any old hit
- rebuilt existing-collection search probes from the post-upsert seed and
  persisted that seed for rollback checkpoint validation
- required AutoID insert responses to return one unique PK per row and stored
  both generation PKs and actual Milvus PKs for rollback search probes
- made declared `expected_resolved_index_type` fail closed when public index
  metadata cannot expose the resolved type
- corrected FAISS search parameter scoping
- used typed TIMESTAMPTZ filter probes
- added visibility waits for phase DML/DQL
- made TEXT MATCH configuration explicit through `enable_match`
- isolated JSON Shredding and Loon/Vortex specialty gates onto a stable
  rollback-safe 2.6-compatible base matrix
- kept Vortex, DISKANN score, SINDI crash, and Woodpecker tSafe failures strict

Final review also found two test files that did not match the configured Ruff
formatter. They were mechanically formatted; no runtime behavior changed.

The phase-search, upsert-seed, AutoID, resolved-index, and MinHash coverage
changes above were post-execution review hardening. They were covered by offline
regression tests; the historical Kubernetes runs in this report were not rerun
for these test-framework-only changes.

## Local and CI Verification

Offline unit tests:

```text
344 passed, 2 deselected
```

The two deselected tests require the external Helm GitHub Pages repository.
They were run separately and both failed before chart rendering because
`https://zilliztech.github.io/milvus-helm/` reset the network connection.

Static validation:

```text
argo lint --offline argo: passed
ruff check on changed Python files: passed
ruff format --check on changed Python files: passed after formatting
git diff --check: passed
```

GitHub Actions for commit `a5d91d69...` passed before the final documentation
and formatting-only commit. CI is required again on the final PR head.

## Resource Cleanup

All retained test releases were removed after reports and logs were captured.
Cleanup verification found no remaining owned deployments, StatefulSets,
services, PVCs, ConfigMaps, or secrets for:

- `pr25-cl30-idx104-f5000-r1-ff9dq`
- `pr25-cl30-json-ro-r1-r56q6`
- `pr25-st26-current-r1-g7wxd`
- `pr25-cl30-json-r3-vw87h`

There were no running `pr25-*` Argo workflows at report completion.

## Acceptance Status

| Area | Status |
|---|---|
| Test framework implementation | Accepted |
| 2.6 rollback-safe type/index coverage | Passed |
| JSON Shredding standalone upgrade/rollback | Passed |
| JSON Shredding cluster format compatibility | Passed with read-only continuous pressure |
| JSON Shredding cluster full-DML rollback | Blocked by Woodpecker reader/tSafe issue |
| Index engine v10/v4 sealed index compatibility | Passed |
| SINDI growing-index write pressure | Blocked by QueryNode crash |
| StructArray FLOAT/VARCHAR nested scalar indexes | Implemented and target-side probes passed |
| StructArray FLOAT16 DISKANN score contract | Blocked by negative MAX_SIM_COSINE score |
| Loon/Vortex rollback | Blocked by persisted-format decoder incompatibility |

PR #25 is ready for review as a test-infrastructure change. The confirmed
Milvus defects should be resolved or explicitly accepted by product owners
before the affected feature scenarios are treated as release-green gates.
