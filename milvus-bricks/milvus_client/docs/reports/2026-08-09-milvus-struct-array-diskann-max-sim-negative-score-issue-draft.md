# Milvus Issue Draft: StructArray FLOAT16 DISKANN Returns Negative MAX_SIM_COSINE Self Score

## Suggested Title

`[Bug]: StructArray FLOAT16 DISKANN returns a negative MAX_SIM_COSINE self-similarity score after index build`

## Environment

- Target Milvus image:
  `harbor.milvus.io/milvusdb/milvus:3.0-20260807-1439dc7d@sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64`
- Target source commit from the image tag: `1439dc7de8b198a01c2afa0ae20c0c473e0e1abc`
- Comparison image:
  `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
- PyMilvus: `3.0.1`
- Deployment: Milvus Operator standalone, RocksMQ, Storage V2
- Vector field: StructArray `ArrayOfVector<FLOAT16_VECTOR>`, dimension `64`
- Index: `DISKANN`
- Metric: `MAX_SIM_COSINE`

## Problem

After a real DISKANN index is built for a StructArray FLOAT16 vector sub-field,
searching with the exact stored vector returns the correct primary key but a
negative score close to `-1.0`:

```json
{
  "collection": "qa_gate_30_to_30latest_forward_struct_array_float16_diskann",
  "distance": -0.999978244304657,
  "expected_pk": 0,
  "field": "embeddings[vector]",
  "index_type": "DISKANN",
  "metric_type": "MAX_SIM_COSINE"
}
```

`MAX_SIM_COSINE` is positively related, so an exact self-match is expected to
be close to `+1.0`. The returned sign is inconsistent with the metric contract
and can also invert document ranking during TokenANN reranking.

## Clean Reproduction From the Upgrade/Rollback Gate

The cleanest reproduction creates a new 3.0 collection on the target image,
inserts 5000 rows, waits for indexes, and validates it before any rollback:

- Argo workflow: `pr25-st30-forward5000-r1-tl6pm`
- Test repository commit: `142d93457e6dfc38b18dea772387bde0c005ce91`
- Failed node: `validate-forward-indexes-after-upgrade`
- Failure occurred on the target image before rollback.
- Every other collection and index in the 3.0 matrix passed.
- The only failure was the StructArray FLOAT16 DISKANN self-score shown above.

This excludes rollback, phase DML, schema evolution, and delete pressure as
required triggers.

## Minimal PyMilvus Reproduction

```python
import random

import numpy as np
from pymilvus import DataType, MilvusClient
from pymilvus.client.embedding_list import EmbeddingList

client = MilvusClient(uri="http://localhost:19530")
name = "struct_float16_diskann_maxsim"

if client.has_collection(name):
    client.drop_collection(name)

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
struct = client.create_struct_field_schema()
struct.add_field("vector", DataType.FLOAT16_VECTOR, dim=64)
struct.add_field("label", DataType.VARCHAR, max_length=64)
schema.add_field(
    "embeddings",
    DataType.ARRAY,
    element_type=DataType.STRUCT,
    struct_schema=struct,
    max_capacity=4,
)
client.create_collection(name, schema=schema)


def vector(pk: int) -> np.ndarray:
    rng = random.Random(pk)
    value = np.asarray([rng.random() for _ in range(64)], dtype=np.float32)
    value /= np.linalg.norm(value)
    return value.astype(np.float16)


for start in range(0, 5000, 100):
    rows = []
    for pk in range(start, start + 100):
        rows.append(
            {
                "id": pk,
                "embeddings": [
                    {"vector": vector(pk), "label": f"label_{pk}"}
                ],
            }
        )
    client.insert(name, rows)

client.flush(name)
indexes = client.prepare_index_params()
indexes.add_index(
    field_name="embeddings[vector]",
    index_name="embeddings_vector",
    index_type="DISKANN",
    metric_type="MAX_SIM_COSINE",
    params={"search_list_size": 64},
)
client.create_index(name, indexes)
client.load_collection(name)

query = EmbeddingList()
query.add(vector(0))
result = client.search(
    collection_name=name,
    data=[query],
    anns_field="embeddings[vector]",
    filter="id == 0",
    limit=5,
    search_params={
        "metric_type": "MAX_SIM_COSINE",
        "params": {"search_list_size": 64},
    },
)
print(result)
```

Expected: PK `0` with a score close to `+1.0`.

Observed on the target image: PK `0` with a score close to `-1.0`.

## Additional Controls

| Workflow | Variable isolated | Result |
|---|---|---|
| `pr25-st30-baseprobe-r1-97wzm` | v3.0.0, no Pod restart, 100 rows | `42/42 Succeeded` |
| `pr25-st30-reload-r1-58gmd` | v3.0.0, same digest forced reload and rollback | `56/56 Succeeded` |
| `pr25-st30-xver-r1-gx2jp` | latest -> v3.0.0 rollback, no write pressure | `56/56 Succeeded` |
| `pr25-st30-phasedml-r1-4tw27` | target phase DML only | `56/56 Succeeded` |
| `pr25-st30-indexedbase-r1-lrtct` | v3.0.0, 5000 rows, index version 8 | `42/42 Succeeded` |
| `pr25-st30-pressure-r1-hzp79` | target write pressure builds a 33,560-row DISKANN segment | failed with the same negative score after rollback |
| `pr25-st30-forward5000-r1-tl6pm` | target-only 5000-row index, no write pressure | failed with the same negative score before rollback |

The last workflow proves the problem is present on the target image before
rollback. The pressure workflow shows how normal insert/upsert traffic exposes
the same defect in an upgrade/rollback gate once a target-side segment receives
a real index.

## Index Build Evidence

The pressure reproduction built the affected target-side index with these
server log fields:

```text
collectionID=468254536246757024
segmentID=468254631521571551
fieldID=102
fieldType=ArrayOfVector
element_type=Float16Vector
index_type=DISKANN
metric_type=MAX_SIM_COSINE
numRows=33560
currentIndexVersion=8
currentScalarIndexVersion=3
buildID=468254631522081645
```

The DataNode log also recorded:

```text
DataNode building index ... numRows=33560 current_index_version=8
Successfully prepare indexBuildTask ... currentIndexVersion=8 currentScalarIndexVersion=3
create index ... index_type:"DISKANN" metric_type:"MAX_SIM_COSINE"
```

## Source Analysis and Root-Cause Hypothesis

Milvus v3.0.0 test coverage explicitly lists DISKANN support for StructArray
FLOAT16 vectors with `MAX_SIM_COSINE`.

In Knowhere `v3.0.6`, TokenANN reranking treats COSINE as
`larger_is_closer=true`. It calls `IndexNode::CalcDistByIDs()` and aggregates
the returned distances as similarity scores. The DISKANN implementation accepts
an `is_cosine` argument but currently discards it:

```cpp
DiskANNIndexNode<DataType>::CalcDistByIDs(..., const bool is_cosine, ...) const {
    (void)bitset;
    (void)is_cosine;
    ...
    pq_flash_index_->calc_dist_by_ids(..., p_dist_ptr + index * labels_len);
}
```

The exact `-0.999978` result is consistent with an underlying negative cosine
distance being passed into a larger-is-better MaxSim aggregation without sign
conversion. This is a root-cause hypothesis, not yet a confirmed ownership
boundary: the packaged target image and the v3.0.0 comparison image should be
checked for their exact Knowhere build revisions and compile options.

## Requested Fix

1. Make DISKANN `CalcDistByIDs()` return COSINE similarity with the same sign
   and ordering contract used by HNSW/IVF TokenANN reranking, or convert the
   value before MaxSim aggregation.
2. Add a Knowhere unit test using an exact self-query for
   `FLOAT16 + DISKANN + MAX_SIM_COSINE` and assert a score near `+1.0`.
3. Add a Milvus Python E2E assertion on score/ranking, not only non-empty
   results, after the DISKANN index is fully built and loaded.
4. Verify index files built by the fixed target remain readable after rollback
   to the supported 3.0 baseline, if that compatibility is part of the release
   contract.

