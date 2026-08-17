# Milvus 持久化索引升级/回滚兼容性补充测试执行报告

日期：2026-08-10

> 2026-08-14 更新：Milvus #52359 修复已进入 3.0 分支镜像，并完成 standalone 与 cluster Pulsar 1CU 的完整升级/回滚复验。最终结论见 [2026-08-14 #52359 修复镜像升级/回滚复验报告](2026-08-14-52359-fixed-image-upgrade-rollback-validation-zh.md)。本报告后续关于 blocker 的描述保留为修复前历史执行记录。

## 1. 结论

本轮围绕“向量、标量数据及索引文件的 writer/reader 兼容性”完成了测试审计、实现、PR 多轮自审、真实 K8s 执行和 Milvus issue 复现。

- 测试实现位于 [vectordb-testbricks PR #27](https://github.com/yanliang567/vectordb-testbricks/pull/27)。
- 最终执行 commit：`625df1493fc2f01c2e12f8a104a4e71dc2e90c49`。
- 最新 PR 实现单元测试：`470 passed`。
- Ruff、Argo offline lint、`git diff --check` 均通过。
- PR CI 通过、状态可合并，无未处理 review comment。
- 2.6 baseline 的数据完整性、feature validation 和升级前压力测试通过。
- 新增 explicit scalar format collection 已确认真实生成并上传索引文件。
- 3.0 target 上，除两个 StructArray scalar AutoIndex blocker collection 外，其余 9 个 schema 在显式 release/load 后完成 31 次向量搜索和 35 次标量索引查询，全部通过。
- Milvus [#52359](https://github.com/milvus-io/milvus/issues/52359) 已稳定复现，并确认同一根因同时影响 VARCHAR 和 FLOAT/INT64 nested HYBRID reader。
- 因 target serviceability fail-closed，真实 rollback 阶段未执行；该项明确记录为 Milvus blocker，不计为测试通过。

## 2. 固定输入

| 项目 | 值 |
|---|---|
| PR | [yanliang567/vectordb-testbricks#27](https://github.com/yanliang567/vectordb-testbricks/pull/27) |
| 最终 commit | `625df1493fc2f01c2e12f8a104a4e71dc2e90c49` |
| Base | `v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6` |
| Target | `3.0-20260807-1439dc7d@sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64` |
| Rollback | `2.6-20260807-d85dc945@sha256:2051a754368d70f589a281fa301a12128d058e531bd6e5d82583e588bccd961e` |
| 部署 | standalone、Rocksmq、Milvus Operator |
| SDK | PyMilvus 3.0.1 |
| Base rows/schema | 1500 |
| Shards/schema | 1 |
| R6 phase rows | new 1500、existing 1100、delete 100 |

## 3. 为什么以前没有测出 #52359

此前存在三个叠加缺口：

1. 2.6 rollback-safe matrix 的 StructArray 只有 nested vector HNSW，没有 scalar sub-field AutoIndex。
2. StructArray scalar indexes 只存在于 3.0 matrix，且使用显式 STL_SORT/INVERTED/BITMAP，未覆盖 `2.6 AUTOINDEX -> HYBRID writer -> 3.0 nested reader`。
3. 真实运行使用默认 16 shards 或多 partition，小 segment 低于 index build threshold。SDK 仍报告 index `Finished`，但 DataCoord 日志实际是 `segment does not need index really`，因此 metadata 检查会假绿。

R3b 已证明 6 到 13 行的小 segment 不生成索引文件。后续将 schema 固定为单 shard 后，又发现 partition-key schema 仍被拆为 94/187 行、四分区 schema 为 375 行，因此进一步新增了无 partition 的 explicit-format owner schema。

## 4. 本轮实现

### 4.1 Schema coverage

| Schema | 主要覆盖 |
|---|---|
| `scalar_dynamic_partition_key` | primitive scalar、JSON、ARRAY、dynamic field、partition key 语义 |
| `scalar_autoindex_formats_rollback_safe` | INT64/FLOAT/BOOL/VARCHAR、JSON double/bool/varchar path、ARRAY<INT64/FLOAT/BOOL/VARCHAR> AutoIndex |
| `scalar_explicit_index_formats_rollback_safe` | 无 partition 的 BITMAP/INVERTED/STL_SORT/TRIE/NGRAM、JSON/ARRAY explicit persisted formats |
| `vector_autoid_bm25` | AutoID、BM25、FLOAT/FLOAT16/BFLOAT16/INT8/BINARY/SPARSE vectors |
| `explicit_partitions_nullable` | VARCHAR PK、显式 partitions、nullable scalar |
| `struct_array_element_rollback_safe` | StructArray scalar round-trip、nested FLOAT_VECTOR HNSW、PK/offset |
| `struct_array_varchar_autoindex_rollback_safe` | StructArray VARCHAR sub-field AutoIndex，#52359 reproducer |
| `struct_array_numeric_autoindex_rollback_safe` | StructArray FLOAT/INT64/BOOL sub-field AutoIndex |
| `nullable_vectors_all` | 六类 nullable vector 及 validity data |
| `geometry_rtree_rollback_safe` | GEOMETRY、RTREE、spatial filter |
| `legacy_index_rollback_safe` | FLAT/IVF/SCANN/HNSW_SQ/BIN_FLAT/SPARSE_WAND |

### 4.2 数据类型

- Scalar：INT8、INT16、INT32、INT64、FLOAT、DOUBLE、BOOL、VARCHAR。
- Compound：JSON、ARRAY<INT64/FLOAT/BOOL/VARCHAR>、dynamic fields、partition key。
- StructArray sub-fields：FLOAT、VARCHAR、INT64、BOOL、FLOAT_VECTOR。
- Vector：FLOAT_VECTOR、FLOAT16_VECTOR、BFLOAT16_VECTOR、INT8_VECTOR、BINARY_VECTOR、SPARSE_FLOAT_VECTOR。
- 其他：GEOMETRY、BM25 function output、AutoID、nullable fields。

### 4.3 索引类型

- Scalar explicit：BITMAP、INVERTED、STL_SORT、TRIE、NGRAM、RTREE。
- Scalar AutoIndex：primitive、JSON path、ARRAY、StructArray sub-field HYBRID。
- Dense vector：FLAT、IVF_FLAT、IVF_SQ8、IVF_PQ、SCANN、HNSW、HNSW_SQ、IVF_RABITQ、DISKANN、AUTOINDEX。
- Binary：BIN_FLAT、BIN_IVF_FLAT。
- Sparse：SPARSE_INVERTED_INDEX、SPARSE_WAND、BM25。

### 4.4 验证点

- collection schema、index metadata、JSON path/cast params。
- insert、flush、load、count、PK sample、scalar/StructArray checksum。
- 向量 self-search：目标 PK、StructArray offset、finite score/distance、metric-specific threshold。
- 标量 index query：broad filter + PK-constrained filter。
- NGRAM 使用 `LIKE "%value%"`，不再用 equality 代替。
- target/rollback 首轮验证通过后严格 `release_collection -> load_collection`，再重复 count、PK、upsert、vector/scalar query。
- target phase insert/upsert/delete、新 collection、scalar/vector query 和 checkpoint。
- per-collection index/query/reload/AutoIndex metrics。
- FLOAT、ARRAY<FLOAT>、StructArray FLOAT、FLOAT_VECTOR 统一按 float32 生成；DOUBLE 和 JSON number 保持 float64。

## 5. 真实运行记录

| Run | 结果 | 分析与处理 |
|---|---|---|
| `pr27-idxfmt-r1-4hphd` | Framework false-negative | StructArray FLOAT_VECTOR float64/float32 checksum 边界，已从生成源头规范 float32 |
| `pr27-idxfmt-r2-ztxhm` | Framework false-negative | 2.6 不支持 MATCH_ANY，已按真实 server version 分层并 fail-closed |
| `pr27-idxfmt-r3-ml49t` | 无效运行 | full SHA 手工输入错误，未进入 Milvus 结果统计 |
| `pr27-idxfmt-r3b-57qcv` | 覆盖无效 | target 查询通过，但 16 shards 下 segment 仅 6 到 13 行，未生成 persisted index file；另发现 phase vector oracle 使用旧向量 |
| `pr27-idxfmt-r4-7kphq` | Framework false-negative | 1500 rows 首次触发 StructArray FLOAT scalar `129.1` checksum 边界，已统一 FLOAT float32 |
| [`pr27-idxfmt-r5-qpnfw`](https://argo-workflows.zilliz.cc/workflows/qa/pr27-idxfmt-r5-qpnfw) | Milvus blocker reproduced | base 全通过；target 的 VARCHAR 和 numeric StructArray HYBRID load 失败 |
| [`pr27-idxfmt-r6-bnz7g`](https://argo-workflows.zilliz.cc/workflows/qa/pr27-idxfmt-r6-bnz7g) | 最终实现验证 | 新 explicit formats 真实 build/upload；9 个非 blocker schema target reader 验证通过；两个 blocker 使 workflow fail-closed |

## 6. R6 通过项

### 6.1 Base 2.6

- `validate-before-upgrade`：通过。
- `validate-schema-features-base`：通过。
- `strict-pressure-before-upgrade`：通过。
- 新 explicit-format collection：`collectionID=468284933719983752`，`segmentID=468284933721945625`，`numRows=1500`。
- 已确认上传文件包括：
  - `milvus_packed_bitmap_index.v3`
  - `milvus_packed_inverted_index.v3`
  - `milvus_packed_stlsort_index.v3`
  - `milvus_packed_marisa_index.v3`
  - `milvus_packed_ngram_index.v3`
  - `HNSW`

这证明新增 explicit scalar coverage 已越过 build threshold，不是 metadata-only。

### 6.2 Target 3.0 非 blocker schema

使用 PyMilvus 3.0.1，对两个已确认 blocker collection 之外的 9 个 schema 执行严格 release/load 后验证：

| 指标 | 结果 |
|---|---:|
| schema | 9/9 passed |
| vector searches | 31 passed |
| scalar index queries | 35 passed |
| query/count failures | 0 |

其中新增 explicit schema 单独完成 11 个 scalar index query 和 1 个 vector search；普通 scalar AutoIndex schema 完成 6 个 scalar query 和 1 个 vector search。两者 loaded/reloaded 两轮均通过。

## 7. Confirmed Milvus blocker

Milvus issue：[milvus-io/milvus#52359](https://github.com/milvus-io/milvus/issues/52359)

补充证据：[issue comment](https://github.com/milvus-io/milvus/issues/52359#issuecomment-5240286296)

### 7.1 VARCHAR nested AutoIndex

```text
collection 468284729278859008
pr27_idxfmt_r5_struct_array_varchar_autoindex_rollback_safe
segment 468284729281900736
At LoadSegment: At Load: Assert "meta_json_.contains(key)"
=> Meta key not found: version
```

### 7.2 FLOAT/INT64/BOOL nested AutoIndex

```text
collection 468284729278859042
pr27_idxfmt_r5_struct_array_numeric_autoindex_rollback_safe
segment 468284729282060776
At LoadSegment: At Load: Assert "meta_json_.contains(key)"
=> Meta key not found: index_length
```

2.6 已将这些 nested scalar AutoIndex 写成 `milvus_packed_hybrid_index.v3`。3.0 `CreateNestedIndex()` 未保留 HYBRID wrapper，错误选择具体 scalar reader；VARCHAR reader 需要 `version`，numeric reader 需要 `index_length`，因此出现不同缺失 key，但根因相同。

R6 `wait-upgrade-serviceability` 在 60 秒后严格失败：

```text
QUERY_FAILED: pr27_idxfmt_r6_struct_array_numeric_autoindex_rollback_safe
SERVICEABILITY_TIMEOUT: elapsed_sec=65.325
```

同时，waiter 已记录 scalar、vector、nullable、geometry、legacy 和 explicit-format 等其他 collection 的 `serviceable_count=1500`。

## 8. 未执行项与边界

- Rollback 2.6 未执行，因为 target 3.0 无法把两个 StructArray scalar AutoIndex collection 恢复为 serviceable。
- `3.0 writer -> 2.6 reader` 的真实 phase collection rollback 验证因此被同一 blocker 阻断。
- 代码和单元测试已实现 phase collection 的 release/load、search、scalar query、upsert probe 和 checkpoint；真实环境结论必须在 #52359 修复镜像可用后补跑，不能将本轮记为通过。
- Partition-key 和显式 multi-partition schema 仍用于语义覆盖；持久化格式证据由无 partition owner schema承担，避免低于 index threshold。
- R6 后的 PR 复审补充了同阶段严格 reload 二次验证，以及 JSON bool/varchar、ARRAY INT64/FLOAT/BOOL AutoIndex。该补强已由单元测试覆盖，但尚未冒充 R6 的真实 K8s 结果；需要在 #52359 修复镜像可用后随完整 workflow 补跑。
- phase checkpoint 现在强制验证版本、生成阶段和 existing/new schema 完整性；严格 release/load 使用默认 120 秒的显式 PyMilvus timeout，避免空 checkpoint false-green 或 load progress 永久等待。
- checkpoint 进一步绑定 seed collection、target-written prefix 和 workflow 行数参数；existing/new collection 必须互斥，每个 entry 必须提供非零 rows、精确 count 和有效 search probe，不能再以同一 collection 或零载荷跳过验证。
- existing/new checkpoint entry 现在分别强制显式 PK range/sample/delete/upsert oracle 或 AutoID returned-ID oracle；缺失字段不能再通过空列表默认值跳过 count、PK、delete 和 upsert 验证。

## 9. 最终 Review 与验证

- PR diff 已完成 matrix、validator、lifecycle、report 和 Argo 五轮以上自审。
- GitHub PR 无未处理 review comments。
- GitHub CI：passed。
- `PYTHONPATH=. pytest -q milvus_client/tests`：`470 passed`。
- changed Python files Ruff check/format：passed。
- `argo lint --offline milvus-bricks/argo`：passed。
- `git diff --check`：passed。

结论：测试框架补充、false-green 修复和 #52359 真实复现均已完成。当前剩余失败由 confirmed Milvus blocker 导致，证据和 issue 已固定；待修复镜像发布后，应使用同一 PR gate 继续执行 target phase 和 rollback 闭环。
