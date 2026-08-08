# Milvus 2.6.18/3.0.0 升级回滚类型与索引补充测试实现计划

**目标：** 补齐 Milvus `2.6.18 -> 3.0 -> 2.6` 与 `3.0.0 -> latest 3.0 -> 3.0.0` 路径中的数据类型、StructArray、索引类型和索引版本覆盖，确保每个新增 collection 都经过数据、索引 metadata、查询/search 和回滚后复用验证。

**架构：** 继续复用现有 3 条参数化 WorkflowTemplate。schema matrix 负责声明 collection 形状和索引，`common/schema.py` 与 `common/data.py` 扩展 StructArray、collection properties 和特殊数据生成能力，`validate_index_compatibility.py` 负责索引 metadata 与可用性，新增统一的 feature semantics validator 负责 StructArray offset、TEXT LOB、MinHash、Geometry 和 Entity TTL 等无法由通用 count/checksum 覆盖的语义。

**技术栈：** Python、PyMilvus 2.6/3.0、PyYAML、pytest、Argo WorkflowTemplate、Milvus Operator/Helm、Milvus 2.6.18/3.0.0 server configuration。

---

## 权威依据

- Milvus `v2.6.18` release note：Nullable Vector、StructArray element-level search。
- Milvus `v3.0.0` release note：StructArray nullable/bitmap/partial update、TEXT LOB、FAISS、MinHash、EmbList DISKANN、SINDI/Block-Max sparse algorithms。
- `v2.6.18` 源码：StructArray scalar 子字段支持 `BOOL`、`INT8/16/32/64`、`FLOAT`、`DOUBLE`、`VARCHAR`；ArrayOfVector 仅支持非 nullable `FLOAT_VECTOR`。
- `v3.0.0` 源码：ArrayOfVector 扩展到全部 fixed-dimension vector；Struct scalar 子字段可建立 `INVERTED`、`STL_SORT`、`BITMAP`；新 sparse algorithms 需要提高 index engine target version。

不根据 proto/client 枚举单独判定正式覆盖范围。`Mol`、`Date`、`Time`、`IVF_HNSW` 暂不进入 promoted gate；GPU index 放入独立 GPU lane，不阻塞 CPU upgrade gate。

## 覆盖原则

1. `2.6.18` 已支持的能力必须进入 rollback-safe baseline，并经过 standalone、cluster 两条 `2.6 -> 3.0 -> 2.6` 路径。
2. 3.0 才支持或会改变持久化格式的能力，不要求回滚到 2.6；必须进入 `3.0 -> latest 3.0 -> 3.0` 路径。
3. 只创建 schema 或 index 不算覆盖。每个 index 必须同时验证 describe metadata 和真实 filter/search。
4. `AUTOINDEX` 不替代显式 index coverage；只有断言 resolved index type 后才能计入底层 index 覆盖。
5. StructArray scalar index 必须覆盖常用类型，最低要求如下：

| Struct sub-field | 类型 | 索引 | 必须执行的断言 |
|---|---|---|---|
| `score_sort` | `FLOAT` | `STL_SORT` | `MATCH_ANY` 等值/范围过滤返回指定 PK |
| `score_inverted` | `FLOAT` | `INVERTED` | 等值过滤返回指定 PK |
| `category_inverted` | `VARCHAR` | `INVERTED` | 字符串等值过滤返回指定 PK |
| `tag_bitmap` | `VARCHAR` | `BITMAP` | 低基数字符串过滤返回指定 PK |
| `rank_sort` | `INT64` | `STL_SORT` | 数值范围过滤返回指定 PK |
| `enabled_bitmap` | `BOOL` | `BITMAP` | bool 过滤返回指定 PK |

`FLOAT` 和 `VARCHAR` 是 promoted gate 的硬要求，不允许只覆盖 INT/BOOL。

## 目标 Matrix

### 1. `schema_matrix_2_6.yaml` rollback-safe 增量

新增以下 schemas，所有现有 `2.6 -> 3.0 -> 2.6` gate 自动执行：

#### `struct_array_element_rollback_safe`

```yaml
- name: struct_array_element_rollback_safe
  compat_mode: rollback_safe
  feature_tags: [struct_array]
  validators:
    - count
    - pk_sample
    - struct_array_scalar_round_trip
    - struct_array_element_search
  fields:
    - {name: id, dtype: INT64, primary: true}
    - {name: normal_vector, dtype: FLOAT_VECTOR, dim: 64}
  struct_arrays:
    - name: items
      max_capacity: 8
      fields:
        - {name: embedding, dtype: FLOAT_VECTOR, dim: 64}
        - {name: score, dtype: FLOAT}
        - {name: category, dtype: VARCHAR, max_length: 128}
        - {name: rank, dtype: INT64}
        - {name: enabled, dtype: BOOL}
  indexes:
    - {field: normal_vector, index_type: HNSW, metric_type: COSINE,
       params: {M: 8, efConstruction: 64}, search_params: {ef: 32}}
    - {field: "items[embedding]", index_type: HNSW, metric_type: COSINE,
       params: {M: 8, efConstruction: 64}, search_params: {ef: 32}}
```

约束：2.6 baseline 不给 Struct scalar 子字段建索引。2.6 的核心验证是 nested scalar 数据升级/回滚不丢失；普通 element-level search 返回正确 `id + offset`，`MAX_SIM_*` 使用 `EmbeddingList` 做 row-level search，只校验正确 PK。

#### `nullable_vectors_all`

覆盖全部六种 nullable vector：

| 字段 | 类型 | 显式索引 |
|---|---|---|
| `nullable_float` | `FLOAT_VECTOR` | `HNSW` |
| `nullable_float16` | `FLOAT16_VECTOR` | `HNSW_SQ` |
| `nullable_bfloat16` | `BFLOAT16_VECTOR` | `DISKANN` |
| `nullable_int8` | `INT8_VECTOR` | `HNSW` |
| `nullable_binary` | `BINARY_VECTOR` | `BIN_FLAT` |
| `nullable_sparse` | `SPARSE_FLOAT_VECTOR` | `SPARSE_WAND` |

每种类型都插入 mixed null/non-null 数据，并在 base、after-upgrade、after-rollback 验证：NULL 不参与 ANN、非 NULL 自搜索命中、query 输出 null 状态保持一致。

#### `geometry_rtree_rollback_safe`

将 Geometry 从纯 3.0 forward-only coverage 中拆出一个 2.6 rollback-safe case。除 RTREE metadata 外，必须执行 `ST_EQUALS` 和 `ST_DWITHIN`。

#### `legacy_index_rollback_safe`

显式覆盖 `FLAT`、`IVF_FLAT`、`IVF_SQ8`、`IVF_PQ`、`SCANN`、`BIN_FLAT`、`SPARSE_WAND`、`HNSW_SQ`。使用小维度和小参数，控制 gate 时长；不依赖 `AUTOINDEX` 推断底层实现。

### 2. `schema_matrix_3_0.yaml` 常规 3.0 增量

#### `struct_array_scalar_indexes`

```yaml
- name: struct_array_scalar_indexes
  compat_mode: forward_only
  feature_tags: [struct_array_element_hybrid_search]
  validators:
    - count
    - pk_sample
    - struct_array_scalar_round_trip
    - struct_array_scalar_index_queries
    - struct_array_element_search
  fields:
    - {name: id, dtype: INT64, primary: true}
    - {name: normal_vector, dtype: FLOAT_VECTOR, dim: 64}
  struct_arrays:
    - name: attributes
      max_capacity: 8
      nullable: true
      fields:
        - {name: embedding, dtype: FLOAT_VECTOR, dim: 64}
        - {name: score_sort, dtype: FLOAT}
        - {name: score_inverted, dtype: FLOAT}
        - {name: category_inverted, dtype: VARCHAR, max_length: 128}
        - {name: tag_bitmap, dtype: VARCHAR, max_length: 64}
        - {name: rank_sort, dtype: INT64}
        - {name: enabled_bitmap, dtype: BOOL}
  indexes:
    - {field: "attributes[embedding]", index_type: HNSW,
       metric_type: MAX_SIM_COSINE,
       params: {M: 8, efConstruction: 64}, search_params: {ef: 32}}
    - {field: "attributes[score_sort]", index_type: STL_SORT}
    - {field: "attributes[score_inverted]", index_type: INVERTED}
    - {field: "attributes[category_inverted]", index_type: INVERTED}
    - {field: "attributes[tag_bitmap]", index_type: BITMAP}
    - {field: "attributes[rank_sort]", index_type: STL_SORT}
    - {field: "attributes[enabled_bitmap]", index_type: BITMAP}
```

另增加一个小 schema 覆盖 `ArrayOfVector<FLOAT16_VECTOR> + DISKANN + MAX_SIM_COSINE`，验证 EmbList DISKANN metadata、release/load、upgrade 和 rollback 后 search。

#### `faiss_float_binary`

- `FLOAT_VECTOR + FAISS + IVF64,Flat + L2`
- `FLOAT_VECTOR + FAISS + OPQ16,IVF64,PQ16x4 + COSINE`
- `BINARY_VECTOR + FAISS + BFlat + HAMMING`

checkpoint 必须比较 `faiss_index_name`，search 使用 matrix 中显式的 `search_params`。

#### `minhash_lsh`

- 输入：analyzer-enabled `VARCHAR`
- 输出：`BINARY_VECTOR`
- Function：`MINHASH`，`num_hashes=128`、`shingle_size=3`
- Index：`MINHASH_LSH`、`MHJACCARD`、`mh_lsh_band=8`
- 断言：近重复文档排在无关文档之前；upgrade 和 rollback 后结果关系保持。

#### `timestamptz_entity_ttl`

- collection properties：`ttl_field=event_time`、`timezone=UTC`
- `event_time`：`TIMESTAMPTZ + STL_SORT`
- baseline rows 使用远未来时间，避免 gate 运行期间自然过期。
- feature validator 额外插入 expired、future、NULL 三类 TTL row，验证可见性；临时 row 不进入 baseline checkpoint PK 范围。

### 3. 新增 `schema_matrix_3_0_storage_v3.yaml`

仅供 Loon/Vortex scenarios 使用：

- `TEXT` small、empty、NULL、Unicode。
- 64 KB 以下、等于 64 KB、超过 64 KB、1 MiB 四个边界。
- TEXT analyzer、`text_match`、`phrase_match`、BM25。
- payload checkpoint 不直接保存大文本，保存 `state/bytes/chars/prefix/suffix/sha256`。
- after-upgrade、after-rollback 都执行 payload hash、BM25 ranking 和 release/load。

Loon/Vortex scenarios 必须设置 `forward_schema_matrix_ref: 3.0_storage_v3`、`forward_workload_enabled: true`、`rollback_forward_validation_enabled: true`。

### 4. 新增 `schema_matrix_3_0_index_v10_v4.yaml`

仅供 index engine version scenarios 使用：

- Sparse IP：`SPARSE_INVERTED_INDEX + SINDI`
- Sparse BM25：`SPARSE_INVERTED_INDEX + BLOCK_MAX_MAXSCORE`
- Sparse IP：`SPARSE_INVERTED_INDEX + BLOCK_MAX_WAND`
- JSON BOOL path：`BITMAP`
- JSON DOUBLE path：`STL_SORT`
- JSON VARCHAR path：`NGRAM`
- JSON AutoIndex：断言 resolved type 为 `HYBRID`

base、target、rollback 三阶段都固定：

```yaml
target_vec_index_version: 10
target_scalar_index_version: 4
```

不在 `2.6 -> 3.0 -> 2.6` promoted gate 中启用这些 index versions。

## 实施任务

### 任务 1：扩展 schema matrix DSL

**文件：**

- 修改：`milvus_client/common/schema.py`
- 修改：`milvus_client/tests/test_schema_manifest.py`
- 修改：`milvus_client/tests/test_create_schema_matrix.py`

**数据结构：**

```python
@dataclass(frozen=True)
class StructArraySpec:
    name: str
    fields: list[FieldSpec]
    max_capacity: int
    nullable: bool = False

@dataclass(frozen=True)
class IndexSpec:
    field: str
    index_type: str
    metric_type: str | None = None
    index_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    search_params: dict[str, Any] = field(default_factory=dict)
```

`SchemaSpec` 增加：

```python
struct_arrays: list[StructArraySpec]
properties: dict[str, Any]
```

**步骤：**

1. 先增加失败测试，解析包含 `struct_arrays`、`properties`、`index_name`、`search_params` 的临时 matrix。
2. 增加 qualified field resolver，使 `attributes[score_sort]` 能解析到所属 StructArray 和 sub-field。
3. matrix validation 校验 StructArray field name、max_capacity、sub-field 类型、重复名称和 index field reference。
4. 对 version `2.6` fail-closed：ArrayOfVector 只允许 `FLOAT_VECTOR`，不允许 nullable StructArray/sub-field。
5. `build_milvus_schema()` 使用 `MilvusClient.create_struct_field_schema()` 创建 nested schema。
6. `create_collection_kwargs()` 将 `properties` 传给 `client.create_collection()`。
7. `build_index_params()` 传递 `index_name` 和 `search_params` 之外的 build params。

**验证：**

```bash
cd milvus-bricks
PYTHONPATH=. uv run pytest -q \
  milvus_client/tests/test_schema_manifest.py \
  milvus_client/tests/test_create_schema_matrix.py
```

预期：StructArray matrix dry-run 通过；非法 2.6 nullable/binary EmbList matrix fail-closed。

### 任务 2：实现 StructArray 确定性数据生成

**文件：**

- 修改：`milvus_client/common/data.py`
- 修改：`milvus_client/tests/test_data_generation.py`

**步骤：**

1. 写失败测试，要求同一个 `seed/pk` 生成完全相同的 StructArray。
2. 每行生成 `1 + pk % 4` 个元素，保证数组长度非固定。
3. nested vector 使用 `pk * 1000 + offset` 作为 seed，避免同一 row 内重复。
4. `FLOAT` 使用可精确重建的数值；`VARCHAR` 使用低基数和高基数两类值；BOOL/INT 使用确定性模式。
5. nullable StructArray 每 10 行生成一个 NULL；非 NULL 行仍包含 mixed scalar values。
6. generic checksum 不包含 nested vector；feature validator 对固定 PK sample 比较 scalar sub-fields，并通过 self-search 验证 vector。
7. 增加 TEXT `value_profile: text_lob_boundary`，只在少量固定 PK 生成 64 KB/1 MiB payload，避免 pressure 产生大量 LOB。

**验证：**

```bash
PYTHONPATH=. uv run pytest -q milvus_client/tests/test_data_generation.py
```

### 任务 3：扩展 index compatibility validator

**文件：**

- 修改：`milvus_client/requests/validate_index_compatibility.py`
- 修改：`milvus_client/common/workload.py`
- 修改：`milvus_client/tests/test_validate_index_compatibility.py`
- 修改：`milvus_client/tests/test_workload.py`

**步骤：**

1. qualified field resolver 同时返回 top-level 与 Struct sub-fields。
2. `_indexed_vector_fields()` 支持 `items[embedding]`。
3. nested vector probe 从确定性 StructArray 的指定 element 取 query vector。普通 element search 检查返回 `id` 和 `offset`；`MAX_SIM_*` 构造 `EmbeddingList` 并检查 row-level PK，不要求 offset。
4. `_indexed_scalar_indexes()` 支持 Struct scalar sub-field。
5. Struct scalar filter 使用以下形式：

```text
MATCH_ANY(attributes, $[score_sort] >= 10.0)
MATCH_ANY(attributes, $[category_inverted] == "category_3")
MATCH_ANY(attributes, $[tag_bitmap] == "tag_1")
MATCH_ANY(attributes, $[enabled_bitmap] == true)
```

6. `MAX_SIM_COSINE/IP/L2` 映射到对应 score/distance 断言。
7. `search_params_for_field()` 优先使用 matrix 的 `IndexSpec.search_params`，补齐 IVF、SCANN、HNSW variants、FAISS、DISKANN、sparse。
8. checkpoint identity 增加兼容性关键参数：`faiss_index_name`、`inverted_index_algo`、`mh_lsh_band`、`sq_type`、`refine`、`refine_type`、JSON path/cast。
9. `expected_resolved_index_type` 存在时，拒绝把 `AUTOINDEX` 当作通过，必须看到指定 resolved type。

**验证：**

```bash
PYTHONPATH=. uv run pytest -q \
  milvus_client/tests/test_validate_index_compatibility.py \
  milvus_client/tests/test_workload.py
```

### 任务 4：增加统一 feature semantics validator

**文件：**

- 创建：`milvus_client/common/feature_validators.py`
- 创建：`milvus_client/requests/validate_schema_features.py`
- 创建：`milvus_client/tests/test_feature_validators.py`
- 创建：`milvus_client/tests/test_validate_schema_features.py`

**注册项：**

- `nullable_vector_semantics`
- `struct_array_scalar_round_trip`
- `struct_array_element_search`
- `struct_array_scalar_index_queries`
- `geometry_filter`
- `text_lob_round_trip`
- `text_match_phrase_match`
- `minhash_search`
- `entity_ttl`
- `index_engine_version`

**步骤：**

1. 读取 matrix 中每个 spec 的 `validators`。
2. `count/pk_sample/search_smoke` 标记为 external validators，由现有 workflow tasks 负责。
3. 对未知 validator fail-closed，避免 manifest 声称覆盖但没有实现。
4. feature validator 输出每个 schema/validator 的独立结果和 metrics。
5. StructArray scalar round-trip 比较固定 PK 的数组长度、每个 offset 的 `FLOAT/VARCHAR/INT64/BOOL`。
6. 普通 element search 必须断言 top hit 的 `id` 和 `offset`；`MAX_SIM_*` 必须断言 row-level top hit 的 `id`，且不伪造 element offset 契约。
7. TEXT validator 比较 payload metadata/hash，不把大文本写进结果 JSON。
8. Entity TTL 使用 checkpoint 范围外的临时 PK，避免污染 baseline count/checksum。

### 任务 5：补齐 2.6 rollback-safe matrices

**文件：**

- 修改：`milvus_client/manifests/schema_matrix_2_6.yaml`
- 修改：`milvus_client/manifests/capability_catalog.yaml`
- 修改：`milvus_client/manifests/feature_inventory.yaml`
- 修改：`milvus_client/tests/test_schema_manifest.py`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

**步骤：**

1. `NullableVector.server_version_min` 改为 `2.6.18`。
2. 将 StructArray capability 拆为 `StructArray` minimum `2.6.18` 和 `StructArrayExtended` minimum `3.0.0`。
3. 保留 `ElementHybridSearch` minimum `3.0.0`。
4. Geometry capability 至少允许 `2.6.18`，并以真实 SDK probe 为最终判定。
5. 加入本计划定义的四个 rollback-safe schemas。
6. contract test 明确断言六种 nullable vector、StructArray、Geometry 和显式 legacy indexes 均在 2.6 matrix。

### 任务 6：补齐 3.0 StructArray scalar index、FAISS、MinHash、TTL

**文件：**

- 修改：`milvus_client/manifests/schema_matrix_3_0.yaml`
- 修改：`milvus_client/tests/test_schema_manifest.py`
- 修改：`milvus_client/tests/test_data_generation.py`
- 修改：`milvus_client/tests/test_validate_index_compatibility.py`

**验收硬条件：**

- matrix 中存在显式 `FLOAT + STL_SORT` 和 `FLOAT + INVERTED` Struct sub-field。
- matrix 中存在显式 `VARCHAR + INVERTED` 和 `VARCHAR + BITMAP` Struct sub-field。
- after-upgrade、after-rollback 的 scalar indexed query metrics 均大于等于 6。
- FAISS checkpoint 保留 factory string。
- MinHash search 验证近重复关系，不只验证有结果。
- TTL collection properties 和过期行为都被验证。

### 任务 7：增加 TEXT/LOB Storage V3 matrix

**文件：**

- 创建：`milvus_client/manifests/schema_matrix_3_0_storage_v3.yaml`
- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改：`milvus_client/tests/test_schema_manifest.py`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改：`milvus_client/tests/test_generate_workflow_report.py`

**步骤：**

1. 将 standalone/cluster Loon-Vortex scenarios 的 forward matrix 指向 Storage V3 matrix。
2. 开启 forward workload 和 rollback forward validation。
3. base 仍可使用普通 3.0 matrix；TEXT collection 只在 Loon/Vortex 已确认 rollout 后创建。
4. report 将 TEXT payload/hash、BM25、text_match 结果列为 required。

### 任务 8：支持 index engine version 10/4

**文件：**

- 修改：`milvus_client/common/deploy.py`
- 修改：`milvus_client/requests/render_milvus_cr.py`
- 修改：`milvus_client/requests/render_milvus_helm_values.py`
- 修改：`milvus_client/common/gates.py`
- 修改：`milvus_client/requests/render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_render_milvus_cr.py`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_argo_template.py`
- 修改：三条 `argo/*upgrade-rollback.yaml`

**新增 phase 参数：**

```text
base-target-vec-index-version
target-target-vec-index-version
rollback-target-vec-index-version
base-target-scalar-index-version
target-target-scalar-index-version
rollback-target-scalar-index-version
```

默认全部为 `-1`，index-version scenarios 三阶段分别配置 target `10/4`。

render 后配置应为：

```yaml
spec:
  config:
    dataCoord:
      targetVecIndexVersion: 10
      targetScalarIndexVersion: 4
```

新增 runtime probe，从运行中 Pod 的合并配置读取 scalar/vector target index version；只检查 CR metadata 不算通过。Milvus 会根据 current/min/max 版本执行 resolve/clamp，公开 SDK index metadata 不暴露最终 build version，因此自动 gate 不宣称精确 `10/4` build；真实执行报告额外保存 DataNode build 日志中的 `currentIndexVersion`/`currentScalarIndexVersion` 证据。

### 任务 9：新增 index-version promoted scenarios

**文件：**

- 创建：`milvus_client/manifests/schema_matrix_3_0_index_v10_v4.yaml`
- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_argo_template.py`

**场景：**

- `standalone-3-0-index-v10-v4-upgrade-rollback`
- `cluster-3-0-index-v10-v4-upgrade-rollback`

两条场景都使用 `3.0.0 -> latest 3.0 -> 3.0.0`，base/target/rollback 保持 target index version `10/4`，执行对应算法并开启 rollback index metadata/search validation。

### 任务 10：将 feature semantics 接入三条 WorkflowTemplate

**文件：**

- 修改：`argo/standalone-2-6-upgrade-rollback.yaml`
- 修改：`argo/standalone-3-0-upgrade-rollback.yaml`
- 修改：`argo/cluster-upgrade-rollback.yaml`
- 修改：`milvus_client/tests/test_argo_template.py`
- 修改：`milvus_client/requests/generate_workflow_report.py`
- 修改：`milvus_client/tests/test_generate_workflow_report.py`

**DAG task：**

- `validate-schema-features-base`
- `validate-schema-features-after-upgrade`
- `validate-schema-features-after-rollback`
- `validate-forward-schema-features-after-upgrade`
- `validate-forward-schema-features-after-rollback`

依赖顺序：serviceability -> data integrity -> index compatibility -> feature semantics。forward rollback task 继续受 `rollback-forward-validation-enabled` 控制。

report 对 feature validator result fail-closed；缺失 required result 不允许 gate 成功。

### 任务 11：文档和提交拆分

**文件：**

- 修改：`docs/upgrade-rollback-gates/README.md`
- 修改：`milvus_client/docs/upgrade-rollback.md`
- 修改：`milvus_client/docs/upgrade-rollback-gates/README.md`

**建议 PR 拆分：**

1. `feat: add struct array schema matrix support`
2. `test: cover 2.6 nullable vectors and struct rollback`
3. `test: add 3.0 struct scalar index coverage`
4. `test: add faiss minhash and entity ttl gates`
5. `test: add text lob storage v3 coverage`
6. `feat: add index engine version gate coverage`

## 全量验证

```bash
cd milvus-bricks

PYTHONPATH=. uv run pytest -q \
  milvus_client/tests/test_schema_manifest.py \
  milvus_client/tests/test_data_generation.py \
  milvus_client/tests/test_create_schema_matrix.py \
  milvus_client/tests/test_validate_index_compatibility.py \
  milvus_client/tests/test_feature_validators.py \
  milvus_client/tests/test_validate_schema_features.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_render_milvus_cr.py \
  milvus_client/tests/test_argo_template.py \
  milvus_client/tests/test_generate_workflow_report.py

PYTHONPATH=. uv run pytest -q

uvx ruff check milvus_client
uvx ruff format --check milvus_client

argo lint --offline argo/standalone-2-6-upgrade-rollback.yaml
argo lint --offline argo/standalone-3-0-upgrade-rollback.yaml
argo lint --offline argo/cluster-upgrade-rollback.yaml

git diff --check
```

## 完成标准

- standalone 和 cluster 的 `2.6 -> 3.0 -> 2.6` 都验证六种 nullable vector、2.6 StructArray element search、Geometry 和 legacy indexes。
- 3.0 StructArray scalar index 至少覆盖 `FLOAT`、`VARCHAR`，并同时覆盖 numeric/bool 常用类型。
- nested scalar index 在 upgrade 和 rollback 后都执行真实 `MATCH_ANY` filter，而不是只 describe metadata。
- Loon/Vortex gate 真实写入并读取超过 64 KB 的 TEXT LOB。
- FAISS、MinHash、SINDI/Block-Max、JSON scalar index version 4 都有独立、可复现的配置和场景。
- 所有新增 feature result 被 workflow report 列为 required，缺失或失败不能产生假绿。
