# Milvus Persisted Index Writer/Reader Compatibility Implementation Plan

**目标：** 将升级/回滚覆盖从“类型或索引是否出现过”提升为“writer version × persisted format × reader version × load/query”，补齐 Milvus #52359 暴露的 StructArray scalar AutoIndex/HYBRID 缺口，并审计同类向量、标量和复合字段索引。

**架构：** 继续使用现有 schema matrix、index checkpoint 和三阶段 WorkflowTemplate，不新增独立测试框架。2.6 rollback-safe matrix 负责验证 `2.6 writer -> 3.0 reader -> 2.6 reader`，phase-created collections 负责验证 `3.0 writer -> 2.6 reader`。新增 per-collection index metrics 和 manifest contract test，避免聚合指标掩盖某个 schema 的零索引覆盖。

**技术栈：** Python、PyMilvus、YAML schema matrix、Argo WorkflowTemplate、Milvus Operator、pytest。

---

## 1. 背景与根因

Milvus issue [#52359](https://github.com/milvus-io/milvus/issues/52359) 的触发路径是：

1. Milvus 2.6.x 为 StructArray VARCHAR scalar sub-field 创建 `AUTOINDEX`。
2. 2.6 将其持久化为 `HYBRID` scalar index wrapper。
3. 升级到 3.0 后重新加载 collection。
4. 3.0 `CreateNestedIndex()` 未处理 `HYBRID_INDEX_TYPE`，错误创建 `StringIndexSort`。
5. `StringIndexSort::LoadEntries()` 读取 hybrid packed metadata 时找不到 `version`，segment load 失败。

此前没有发现该问题不是 validator 接受了错误结果，而是 matrix 根本没有创建该 index：

- `schema_matrix_2_6.yaml` 的 StructArray 只为普通 vector 和 nested vector 创建 HNSW。
- StructArray scalar index 全部位于 `schema_matrix_3_0.yaml`，由 3.0 写出，且使用显式 `STL_SORT/INVERTED/BITMAP`。
- JSON `AUTOINDEX -> HYBRID` 使用 JSON index factory，不经过 nested StructArray loader。
- 真实 K8s 运行使用 commit `a5d91d6`；显式 release/load 二次验证在后续 commit `6f9d827` 才加入，最终代码没有重新执行完整真实环境矩阵。

## 2. Writer/Reader 审计结果

### 2.1 已覆盖的数据兼容性

| 数据族 | 2.6 writer -> 3.0 reader | 3.0 writer -> 2.6 reader | 当前结论 |
|---|---|---|---|
| INT8/16/32/64、FLOAT、DOUBLE、BOOL、VARCHAR | baseline checksum/query | phase existing/new DML/DQL | 已覆盖 |
| JSON、ARRAY、dynamic field、partition key | baseline checksum/filter | phase existing/new DML/DQL | 已覆盖 |
| StructArray FLOAT/VARCHAR/INT64/BOOL | scalar round-trip | phase DML/checkpoint | 数据已覆盖，scalar index format 未覆盖 |
| FLOAT/FLOAT16/BFLOAT16/INT8/BINARY/SPARSE vector | search + null semantics | phase new/existing search | 已覆盖 |
| StructArray FLOAT_VECTOR | element search + PK/offset | phase search | 已覆盖 |
| Geometry | RTREE + spatial filter | phase query | 已覆盖 |
| TEXT、TIMESTAMPTZ、extended StructArray | 不属于 2.6 baseline | 3.0 -> newer 3.0 -> 3.0 | 已覆盖对应 3.0 path |

### 2.2 已覆盖的 persisted vector index formats

2.6 baseline 已包含并在 3.0 target load/search：

- FLAT、IVF_FLAT、IVF_SQ8、IVF_PQ、SCANN
- HNSW、HNSW_SQ、IVF_RABITQ、DISKANN
- BIN_FLAT、BIN_IVF_FLAT
- SPARSE_INVERTED_INDEX、SPARSE_WAND、BM25
- FLOAT_VECTOR 和 INT8_VECTOR `AUTOINDEX`
- StructArray nested FLOAT_VECTOR HNSW
- nullable FLOAT/FLOAT16/BFLOAT16/INT8/BINARY/SPARSE indexes

这些 index 同时由 target-side phase collections 创建，并在 2.6 rollback 后 query/search。向量侧暂未发现与 #52359 同等级的 writer-reader 空白；AutoIndex 的所有算法选择并未穷举，但对应 concrete format 已有显式覆盖。

### 2.3 Scalar/compound AutoIndex format 缺口

| Field family | 当前 baseline | 缺口 | 优先级 |
|---|---|---|---|
| Primitive INT64 | `AUTOINDEX` 已存在 | 缺少 per-field resolved/loaded evidence | P1 |
| Primitive FLOAT | 仅显式 STL_SORT | 缺少 2.6 HYBRID writer -> 3.0 reader | P1 |
| Primitive VARCHAR | TRIE/NGRAM 显式 index | 缺少 string HYBRID compatibility | P1 |
| Primitive BOOL | 显式 BITMAP | 缺少 low-cardinality HYBRID compatibility | P1 |
| ARRAY | 显式 INVERTED | 缺少 composite HYBRID compatibility | P1 |
| JSON path | 显式 INVERTED | 缺少 2.6 AutoIndex resolver output 和 persisted-format compatibility | P1 |
| StructArray scalar | 没有 scalar index | 缺少 nested HYBRID；直接对应 #52359 | P0 |

## 3. 选型

### 方案 A：只增加 #52359 VARCHAR reproducer

优点是改动小、复现快。缺点是仍然保留 FLOAT/BOOL/ARRAY/JSON 等相同 wrapper format 的盲区，下一次只能继续按 issue 补洞。

### 方案 B：扩展 2.6 rollback-safe matrix，并增加 coverage contract 与 per-collection metrics

这是推荐方案。它复用现有 workflow 和 validator，同时覆盖 primitive、compound、JSON、StructArray 四种 scalar AutoIndex factory 路径。新增 contract test 可以阻止后续 matrix 重构时再次丢失 writer-reader 组合。

### 方案 C：新增通用 persisted-format DSL 和单独 workflow generator

长期表达力最好，但会重复现有 schema/index DSL，并显著增加 PR blast radius。本轮不采用。

## 4. Schema Matrix 修改

### 任务 1：补齐 primitive/compound scalar AutoIndex

**文件：**

- 修改：`milvus_client/manifests/schema_matrix_2_6.yaml`
- 测试：`milvus_client/tests/test_schema_manifest.py`

新增独立的 `scalar_autoindex_formats_rollback_safe` schema，避免现有显式
scalar index 先失败而掩盖 AutoIndex writer/reader 路径，也使 per-collection
metrics 能单独表达该组 persisted format：

```yaml
- name: scalar_autoindex_formats_rollback_safe
fields:
  - {name: id, dtype: INT64, primary: true, auto_id: false}
  - {name: int64_auto, dtype: INT64}
  - {name: float_auto, dtype: FLOAT}
  - {name: bool_auto, dtype: BOOL}
  - {name: varchar_auto, dtype: VARCHAR, max_length: 256}
  - {name: json_auto, dtype: JSON}
  - {name: arr_varchar_auto, dtype: ARRAY, element_type: VARCHAR, max_capacity: 16, max_length: 128}
  - {name: embedding, dtype: FLOAT_VECTOR, dim: 64}
indexes:
  - {field: int64_auto, index_type: AUTOINDEX}
  - {field: float_auto, index_type: AUTOINDEX}
  - {field: bool_auto, index_type: AUTOINDEX}
  - {field: varchar_auto, index_type: AUTOINDEX}
  - {field: json_auto, index_type: AUTOINDEX, params: {json_cast_type: double, json_path: "json_auto['bucket']"}}
  - {field: arr_varchar_auto, index_type: AUTOINDEX}
  - {field: embedding, index_type: HNSW, metric_type: COSINE}
```

保留现有 `int64_category + AUTOINDEX` 作为历史 control；新 schema 独立覆盖
integral、floating、bool、string、JSON 和 ARRAY factory 路径。

### 任务 2：补齐 StructArray nested scalar AutoIndex

**文件：**

- 修改：`milvus_client/manifests/schema_matrix_2_6.yaml`
- 测试：`milvus_client/tests/test_schema_manifest.py`
- 测试：`milvus_client/tests/test_feature_validators.py`

不要把所有 nested scalar AutoIndex 放入同一 schema。否则 FLOAT/BOOL index
可能先于 VARCHAR 失败，无法稳定证明 #52359 的 string loader 路径。新增两个
独立 schema：

```yaml
- name: struct_array_varchar_autoindex_rollback_safe
  validators:
    - count
    - pk_sample
    - struct_array_scalar_round_trip
    - struct_array_scalar_index_queries
  validator_params: {min_struct_scalar_index_queries: 1}
  fields:
    - {name: id, dtype: INT64, primary: true, auto_id: false}
    - {name: normal_vector, dtype: FLOAT_VECTOR, dim: 64}
  struct_arrays:
    - name: items
      max_capacity: 8
      fields:
        - {name: category, dtype: VARCHAR, max_length: 128}
  indexes:
    - {field: normal_vector, index_type: HNSW, metric_type: COSINE, params: {M: 8, efConstruction: 64}}
    - {field: "items[category]", index_type: AUTOINDEX}
```

第二个 schema 隔离 numeric/boolean companion paths：

```yaml
- name: struct_array_numeric_autoindex_rollback_safe
  validator_params: {min_struct_scalar_index_queries: 3}
  validators:
    - count
    - pk_sample
    - struct_array_scalar_round_trip
    - struct_array_scalar_index_queries
  struct_arrays:
    - name: items
      max_capacity: 8
      fields:
        - {name: score, dtype: FLOAT}
        - {name: rank, dtype: INT64}
        - {name: enabled, dtype: BOOL}
  indexes:
  - {field: "items[score]", index_type: AUTOINDEX}
  - {field: "items[rank]", index_type: AUTOINDEX}
  - {field: "items[enabled]", index_type: AUTOINDEX}
```

VARCHAR schema 是 #52359 的确定性 reproducer；FLOAT、INT64、BOOL schema
验证 nested factory 对不同 internal scalar implementations 的兼容性。保留原
`struct_array_element_rollback_safe` 作为 scalar data + nested vector control。

## 5. Validator 与报告增强

### 任务 3：输出 per-collection index coverage metrics

**文件：**

- 修改：`milvus_client/requests/validate_index_compatibility.py`
- 测试：`milvus_client/tests/test_validate_index_compatibility.py`

每个 collection 必须输出：

```text
<collection>.actual_indexes_total
<collection>.vector_searches_total
<collection>.scalar_index_queries_total
<collection>.reload_cycles_total
<collection>.reload_vector_searches_total
<collection>.reload_scalar_index_queries_total
<collection>.declared_autoindexes_total
```

这样报告不能再用全局 `scalar_index_queries_total=18` 掩盖 StructArray schema 实际为 0。

### 任务 4：增加 persisted-format coverage contract

**文件：**

- 修改：`milvus_client/tests/test_schema_manifest.py`

参数化检查 `schema_matrix_2_6.yaml`：

- 所有六种 vector data types 至少有一个 persisted index。
- explicit vector format inventory 保持完整。
- primitive scalar AutoIndex 至少覆盖 INT64、FLOAT、BOOL、VARCHAR。
- ARRAY 和 JSON path 至少各有一个 AutoIndex。
- StructArray scalar AutoIndex 至少覆盖 FLOAT、VARCHAR、INT64、BOOL。
- StructArray scalar schema 必须启用 `struct_array_scalar_index_queries`，且 minimum 等于实际 indexed scalar field 数。

### 任务 5：强制 reload target-written phase collections

**文件：**

- 修改：`milvus_client/requests/validate_phase_dml_dql.py`
- 测试：`milvus_client/tests/test_validate_phase_dml_dql.py`

after-upgrade 创建新 collection 后继续 flush/load 并执行 search。after-rollback
读取 phase checkpoint 时，在 count/query/search 前对 existing 和 new collections
执行严格的：

```text
release_collection -> load_collection -> count/query/search
```

新增 metrics：

```text
phase_checkpoint_reload_collections_total
phase_checkpoint_reload_failures_total
phase_checkpoint_scalar_index_queries_total
```

release 或 load 失败必须进入 validation failures，不能 best-effort 忽略。该步骤
负责证明 target 3.0 写出的 index 文件可由 rollback 2.6 reader 加载。reload 后
还必须复用 index compatibility validator，对每个 scalar index 执行 broad filter
和 PK-constrained filter，并输出逐 collection scalar query 指标，避免只用 count
或向量 search 代表全部 index 可用。

## 6. Workflow 验证路径

现有正式 2.6 gates 已满足所需生命周期，不增加 DAG task：

1. Base 2.6 创建 collection/index，insert、flush、load、query/search。
2. Target 3.0 rollout 后执行 index compatibility load/query。
3. 显式 release collection，再 load collection，再次 query/search。
4. Target phase 创建同 schema 的新 collection/index，并 flush/load。
5. Target phase 对新写 collection 执行 vector search 和全部 scalar index filter；
   对发生 upsert 的 collection 跳过被修改字段，验证其余 scalar indexes。
6. Rollback 2.6 后显式 release/load baseline 和 target-written collections，并重复
   vector/scalar index query。
7. 全流程禁止 rebuild index。

若 #52359 存在，步骤 2 或步骤 3 应产生：

```text
At LoadSegment: At Load: Assert "meta_json_.contains(key)"
=> Meta key not found: version
```

## 7. 单元测试与静态验证

```bash
PYTHONPATH=. pytest -q milvus_client/tests
ruff check <changed-python-files>
ruff format --check <changed-python-files>
argo lint --offline argo
git diff --check
```

## 8. PR Review 策略

提交 PR 后至少执行以下自审轮次：

1. Matrix review：确认每个新增 field 都有 deterministic data、index 和 executable filter。
2. Validator review：确认 query 返回目标 PK，reload metrics 不是简单累加请求数。
3. Lifecycle review：确认 index 在 base 2.6 创建，target 不 rebuild。
4. Report review：确认 per-collection metrics 能区分 StructArray 和普通 scalar collections。
5. Argo review：确认 standalone/cluster 均消费同一 rollback-safe matrix。

发现 blocking/P1/P2 后修复并重新执行完整验证，直到没有阻塞 merge 的 comments。

## 9. 真实 K8s 复现与验收

### 9.1 固定输入

```text
base image:
harbor.milvus.io/milvusdb/milvus:v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6

target image:
harbor.milvus.io/milvusdb/milvus:3.0-20260807-1439dc7d@sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64

rollback image:
harbor.milvus.io/milvusdb/milvus:2.6-20260807-d85dc945@sha256:2051a754368d70f589a281fa301a12128d058e531bd6e5d82583e588bccd961e
```

### 9.2 第一阶段：最小规模复现

- 使用 PR full commit SHA。
- standalone、100 rows/schema、无观察等待。
- 保留完整 index compatibility 和 release/load validation。
- 预期 workflow 在 after-upgrade load 阶段失败。
- 固定 result JSON、QueryNode/standalone log、index metadata checkpoint 和 pod events。

### 9.3 第二阶段：相关路径验证

- 若 primitive/ARRAY/JSON AutoIndex 能通过，记录每个 collection 的 index/query/reload metrics。
- StructArray nested scalar failure必须能定位到具体 collection/field。
- 必要时运行 cluster smoke，确认不是 standalone-only 行为。

### 9.4 验收标准

- #52359 在 pinned target image 上稳定复现。
- 新增 test 在缺陷存在时严格失败，不允许 warning 或 skip。
- 现有 vector/scalar coverage 不回退。
- 报告明确区分数据兼容、explicit index compatibility、AutoIndex wrapper compatibility。
- 生成中文执行报告，包含 workflow、commit、镜像 digest、失败阶段、日志证据和 issue link。

## 10. 文档 Review 记录

### Round 1

- **[P1] 单一 StructArray schema 不能保证命中 VARCHAR loader。** 已拆分
  `struct_array_varchar_autoindex_rollback_safe` 和 numeric companion schema。
- **[P1] target-created phase collection 缺少显式 rollback reload。** 已新增任务 5，
  要求 after-rollback checkpoint validation 严格 release/load。
- **[P2] 公共 SDK 不一定暴露 2.6 scalar AutoIndex 的 internal HYBRID type；JSON AutoIndex 也可能由 coordinator 先解析为显式 INVERTED，而不是写出 HYBRID wrapper。**
  自动 gate 不在 metadata 不可观测时伪造 resolved type；以 declared AUTOINDEX、
  persisted load、真实 filter 和保存的 DataNode/standalone build/load 日志作为证据。
- **[P2] 聚合 scalar query 指标会掩盖 schema 零覆盖。** 已新增 per-collection
  index/query/reload/AutoIndex metrics。

### Round 2

- **[P1] target-written collection 只有 reload 和 vector search，scalar index 可在
  postings/query 损坏时假绿。** 已复用统一 scalar index validator，在 target phase
  写入后以及 rollback checkpoint reload 后执行 broad filter 与 PK-constrained
  filter，并输出总量和逐 collection 指标。
- **[P1] 直接跳过 upsert 修改字段会漏掉 scalar index 增量更新。** 已使用 phase
  checkpoint 中保存的 `upsert_samples.expected` 构造 probe override；普通 scalar、
  ARRAY、JSON 和 StructArray sub-field 均使用更新后的真实期望值验证。
- **[P2] 两个旧单测按 matrix 数组下标定位 `vector_autoid_bm25`。** 新增 schema
  后会验证错误对象；已改为按 schema name 定位，避免 manifest 扩展再次触发。

### 提交前验证

```text
PYTHONPATH=. pytest -q milvus_client/tests: 441 passed
ruff check: passed
ruff format --check: passed
argo lint --offline argo: no linting errors
git diff --check: passed
```

### Round 3：真实 K8s 首轮反馈

- **[P1] base 2.6 的 StructArray nested FLOAT_VECTOR checksum 在 seed=0 下假失败。**
  Python 生成值为 float64，Milvus 持久化并返回 float32；个别分量跨过五位小数
  rounding 边界。已将 `stable_float_vector()` 的输出从源头规范为 IEEE-754
  float32 Python 值，保留 nested vector 全量 checksum，同时不降低 DOUBLE scalar
  的 checksum 精度。
- 已增加 seed=0、100 rows 的 StructArray vector float32 round-trip 回归测试；该
  输入与 workflow `pr27-idxfmt-r1-4hphd` 的失败 hash 完全一致。
