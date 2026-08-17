# Milvus #52359 修复镜像升级/回滚复验报告

日期：2026-08-14

## 1. 结论

Milvus [#52359](https://github.com/milvus-io/milvus/issues/52359) 的修复已通过 standalone 和 cluster Pulsar 1CU 两条完整升级/回滚路径验证。

- Base：Milvus 2.6.18。
- Target：包含 #52359 修复的 3.0 分支 daily image。
- Rollback：最新 2.6 分支 daily image。
- 两次 workflow 均为 `Succeeded 57/57`，最终报告 `status=passed`、`release_gate_eligible=true`。
- 11 个 collection、77 个 persisted indexes 在 target 和 rollback 阶段均通过 metadata、search/query 和严格 release/load 复验。
- Target 阶段完成存量 collection insert/upsert/delete、新建 11 个 collection，并在 rollback 后通过 checkpoint、count、PK、delete/upsert oracle、vector/scalar query 和 reload 验证。
- `indexes_dropped=0`、`indexes_rebuilt=0`，验证读取的是跨版本保留的索引，不是回滚阶段重建后的替代结果。
- PR #29 收紧后的 pressure gate 在两次真实运行中均为 `failed=0`；所有被排除的 `collection not loaded` 明细都匹配同一 collection 的零 padding reload 窗口。
- 未再复现 StructArray VARCHAR/numeric nested HYBRID 的 `version` 或 `index_length` metadata 缺失错误。

结论只适用于下表固定的 daily image digest。不能据此宣称正式 `v3.0.0` tag 已包含修复，也不能替代未来 3.0.1 正式版本的 release gate。

## 2. 固定输入

| 项目 | Standalone | Cluster |
|---|---|---|
| Scenario | `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest` | `cluster-2-6-18-to-3-0-latest-rollback-2-6-latest` |
| Workflow | [`pr29-reload-strict-qv5gr`](https://argo-workflows.zilliz.cc/workflows/qa/pr29-reload-strict-qv5gr) | [`c26rb-fixed-fqg5f`](https://argo-workflows.zilliz.cc/workflows/qa/c26rb-fixed-fqg5f) |
| 测试代码 | `7b969207ceaf9b4593a7abeced19c0b5d718f06c` | `12415e6a46a0d3e1d5a1545d3ab163318f50a20d` |
| PR | [vectordb-testbricks#29](https://github.com/yanliang567/vectordb-testbricks/pull/29) | PR #29 merge commit |
| Deploy profile | `standalone-rocksmq.yaml` | `cluster-pulsar-1cu.yaml` |
| Base image | `harbor.milvus.io/milvusdb/milvus:v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6` | 同左 |
| Target image | `harbor.milvus.io/milvusdb/milvus:3.0-20260813-7606f8e3@sha256:5e88b9df376f9682765932e810e453031f9cf21f67f628ae40350a63ec3449a2` | 同左 |
| Rollback image | `harbor.milvus.io/milvusdb/milvus:2.6-20260813-75c41815@sha256:a0e42c26f7aecf754b8b4ac6759a824e86ccddd14303d9639c4e8fa532f0c15c` | 同左 |
| SDK | PyMilvus 3.0.1 | PyMilvus 3.0.1 |

正式 gate 参数：

- `rows-per-collection=5000`
- `phase-existing-dml-rows=1000`
- `phase-existing-delete-rows=100`
- `phase-new-collection-rows=3000`
- `pressure-fail-on-error=true`
- `gate-allow-warning=false`
- `index-compatibility-validation-enabled=true`
- `phase-dml-dql-validation-enabled=true`
- LoonFFI、Vortex、storage v3、JSON Shredding 均未启用

## 3. 数据与索引覆盖

本轮使用 `schema_matrix_2_6.yaml` 的 11 个 rollback-safe schema。

| Schema | 核心覆盖 |
|---|---|
| `scalar_dynamic_partition_key` | primitive scalar、JSON、ARRAY、dynamic fields、partition key |
| `scalar_autoindex_formats_rollback_safe` | primitive、JSON path、ARRAY AutoIndex |
| `scalar_explicit_index_formats_rollback_safe` | BITMAP、INVERTED、STL_SORT、TRIE、NGRAM persisted formats |
| `vector_autoid_bm25` | AutoID、BM25、dense/binary/sparse vectors |
| `explicit_partitions_nullable` | VARCHAR PK、显式 partition、nullable scalar |
| `struct_array_element_rollback_safe` | StructArray round-trip、nested FLOAT_VECTOR HNSW、offset |
| `struct_array_varchar_autoindex_rollback_safe` | StructArray VARCHAR sub-field AutoIndex |
| `struct_array_numeric_autoindex_rollback_safe` | StructArray FLOAT/INT64/BOOL sub-field AutoIndex |
| `nullable_vectors_all` | 六类 nullable vectors 与 validity data |
| `geometry_rtree_rollback_safe` | GEOMETRY、RTREE、spatial filter |
| `legacy_index_rollback_safe` | legacy dense/binary/sparse vector indexes |

覆盖的数据类型：

- Scalar：INT8、INT16、INT32、INT64、FLOAT、DOUBLE、BOOL、VARCHAR。
- Compound：JSON、ARRAY<INT64/FLOAT/BOOL/VARCHAR>、dynamic fields、partition key。
- StructArray sub-fields：FLOAT、VARCHAR、INT64、BOOL、FLOAT_VECTOR。
- Vector：FLOAT_VECTOR、FLOAT16_VECTOR、BFLOAT16_VECTOR、INT8_VECTOR、BINARY_VECTOR、SPARSE_FLOAT_VECTOR。
- 其他：GEOMETRY、BM25 function output、AutoID、nullable fields。

覆盖的索引类型：

- Scalar explicit：BITMAP、INVERTED、STL_SORT、TRIE、NGRAM、RTREE。
- Scalar AutoIndex：primitive、JSON path、ARRAY、StructArray sub-field HYBRID。
- Dense vector：FLAT、IVF_FLAT、IVF_SQ8、IVF_PQ、SCANN、HNSW、HNSW_SQ、IVF_RABITQ、DISKANN、AUTOINDEX。
- Binary：BIN_FLAT、BIN_IVF_FLAT。
- Sparse：SPARSE_INVERTED_INDEX、SPARSE_WAND、BM25。

每个 topology 实际检查 11 个 collection 和 77 个 index。

## 4. 核心验证点

### 4.1 升级前

- 创建 schema、插入 5000 rows/schema、flush、load。
- collection schema、index metadata、JSON path/cast params。
- count、PK sample、scalar/StructArray checksum。
- 向量 self-search：目标 PK、StructArray offset、finite distance、metric-specific threshold。
- 标量 index query：broad filter 与 PK-constrained filter。
- schema feature validator 和升级前 strict pressure。

### 4.2 升级后

- 校验真实 server image/version 和配置约束。
- 不 drop、不 rebuild 读取 2.6 persisted indexes。
- 11 个 collection 首轮 index query/search 后执行严格 `release_collection -> load_collection` 并复验。
- 对存量 collection 每个写入 1000 行、删除 100 行并执行 upsert；合计 insert 11000、delete 1100、upsert 10000。
- 创建 11 个 target phase collection，每个写入 3000 行，合计 33000 行。
- 对 22 个 existing/new collection 执行 phase reload 后的 vector/scalar query。

### 4.3 回滚后

- 等待 2.6 所有组件 Ready，并验证 11 个 baseline collection serviceable。
- 使用升级前 index checkpoint 验证 77 个 persisted indexes，禁止 drop/rebuild。
- 验证 target 阶段 existing/new checkpoint 的 collection 身份、行数、PK/AutoID、delete/upsert oracle 和 search probe。
- 对 checkpoint 的 22 个 collection 执行 release/load 复验。
- 对 rollback current phase 的 33 个 collection 执行 release/load、search 和 scalar index query。
- 执行 rollback strict pressure 和最终 pressure aggregation。

## 5. Standalone 结果

Workflow：[`pr29-reload-strict-qv5gr`](https://argo-workflows.zilliz.cc/workflows/qa/pr29-reload-strict-qv5gr)

| 指标 | 结果 |
|---|---:|
| Workflow | `Succeeded 57/57` |
| Final report | `passed` |
| Indexes | 77/77 |
| Index drop/rebuild | 0/0 |
| Upgrade/rollback index reload | 11/11 |
| Upgrade phase reload | 22 |
| Rollback checkpoint reload | 22 |
| Rollback current-phase reload | 33 |
| Pressure attempts | 144 |
| Pressure passed/failed/excluded slices | 123/0/21 |
| Steady-state operations | 528492 |
| Steady-state failures | 0 |
| Steady-state success rate | 1.0 |

Pressure 精确性复核：

- collection reload windows：99。
- 检查 `collection not loaded` 失败明细：880。
- 同 collection 且零 padding 精确重叠：880。
- 未匹配或跨 collection：0。

## 6. Cluster Pulsar 1CU 结果

Workflow：[`c26rb-fixed-fqg5f`](https://argo-workflows.zilliz.cc/workflows/qa/c26rb-fixed-fqg5f)

### 6.1 Index compatibility

| 指标 | Upgrade 后 | Rollback 后 |
|---|---:|---:|
| Collections checked | 11 | 11 |
| Actual indexes | 77 | 77 |
| Indexes dropped | 0 | 0 |
| Indexes rebuilt | 0 | 0 |
| Initial vector searches | 33 | 33 |
| Initial scalar index queries | 44 | 40 |
| Reload cycles | 11 | 11 |
| Reload vector searches | 33 | 33 |
| Reload scalar index queries | 44 | 40 |

Rollback 的 StructArray scalar sub-field index query 在 2.6 按 capability contract 标记为 unsupported，不冒充自动化通过；对应数据 round-trip、collection load 和其他 vector/scalar query 仍严格执行。

### 6.2 Phase DML/DQL

| 指标 | 结果 |
|---|---:|
| Existing collections | 11 |
| Target new collections | 11 |
| Existing inserted | 11000 |
| Existing deleted | 1100 |
| Existing upserted | 10000 |
| New collection inserted | 33000 |
| Upgrade phase reload | 22/22，failure 0 |
| Rollback checkpoint validated | true |
| Rollback checkpoint collections | 11 existing + 11 new |
| Rollback checkpoint reload | 22/22，failure 0 |
| Rollback current-phase reload | 33/33，failure 0 |
| Rollback reload vector searches | 99 |
| Rollback reload scalar index queries | 120 |

### 6.3 Pressure 与 cleanup

| 指标 | 结果 |
|---|---:|
| Workflow | `Succeeded 57/57` |
| Final report | `passed` |
| Pressure attempts | 155 |
| Pressure passed/failed/excluded slices | 135/0/20 |
| Steady-state samples | 85 |
| Steady-state operations | 397157 |
| Steady-state failures | 0 |
| Steady-state success rate | 1.0 |
| Cleanup | `completed` |

Pressure 精确性复核：

- collection reload windows：99。
- 检查 `collection not loaded` 失败明细：919。
- 同 collection 且零 padding 精确重叠：919。
- 未匹配或跨 collection：0。

## 7. #52359 修复结论

修复前，2.6 写出的 StructArray nested HYBRID index 在 3.0 reader 中会错误选择具体 scalar reader：

- VARCHAR 路径缺少 `version` metadata。
- FLOAT/INT64/BOOL 路径缺少 `index_length` metadata。

本次修复镜像在两种 topology 上均完成：

1. 2.6 writer 创建并持久化 nested HYBRID index。
2. 3.0 target reader 首次 load、search/query 和 release/load 二次读取。
3. 3.0 阶段继续写入、upsert、delete 和创建新 collection。
4. 回滚 2.6 后重新读取 baseline 与 target-written 数据。
5. 全程未 drop/rebuild index。

因此，在固定 image digest 和非 Vortex 配置下，#52359 原始 blocker 已不再复现，standalone 与 cluster 的支持路径均通过。

## 8. 边界与后续

- 本轮验证的是 2.6.18 -> 3.0 branch fixed image -> 2.6 branch image，不是正式 3.0.1 release。
- Vortex/LoonFFI/storage v3 按支持合同关闭；不覆盖 3.0.1 Vortex candidate rollback 场景。
- 不覆盖 `target-only-features` gate；3.0-only forward schema 仍需使用独立 scenario 验证。
- 不覆盖 Woodpecker 1CU/2CU HA、JSON Shredding、index v10/v4 等其他注册 gate。
- 3.0.1 RC 或正式镜像发布后，应使用相同 full-SHA 参数再执行 standalone 和 cluster 两条 release gate，避免 daily image 结论被错误继承。

建议后续顺序：

1. 将本报告链接和 standalone/cluster 通过证据补充到 Milvus #52359。
2. 使用同一修复镜像执行 `cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest`，补齐 3.0-only forward collection 的 cluster 证据。
3. 3.0.1 RC 可用后执行 standalone/cluster Vortex candidate gate，保持 `release_gate_eligible=false`，直到正式支持合同确认。

## 9. 证据与清理

Cluster 原始 artifacts 已保存到本地 `outputs`：

- `c26rb-fixed-fqg5f-orchestrator-report.tgz`
- `c26rb-fixed-fqg5f-final-report-md.tgz`
- `c26rb-fixed-fqg5f-flow-summary.tgz`
- `c26rb-fixed-fqg5f-pressure-summary.tgz`

两次 workflow 的 `flow_summary` 均记录：

```json
{
  "cleanup_attempted": true,
  "cleanup_status": "completed",
  "cleanup_error": "",
  "kept_resources": []
}
```

Artifacts 固定后已删除临时 Workflow CR 和 WorkflowTemplate，并确认无同名 Milvus、PVC、ConfigMap、Service 或 Pod 残留。
