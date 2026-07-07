# Milvus 3.0 Feature Inventory for Test Bricks

**目标：** 整合 Milvus 3.0 官方 release notes、roadmap 和 Feishu Base 中的 PR 级功能，形成后续 test bricks 设计和实现的功能目录。

**用途：** 本文不是完整测试计划，也不是一次性实现清单。它用于在实现前明确功能分组、覆盖优先级、兼容性边界，以及这些功能对 request 参数、schema matrix、workload、validator 和 Argo 编排的影响。

**来源：**
- Milvus release notes: `https://milvus.io/docs/release_notes.md`
- Milvus roadmap: `https://milvus.io/docs/roadmap.md`
- Feishu Base: `https://zilliverse.feishu.cn/base/Gy4VbeG8daTv88s5OtocmUuSnyh?table=tblOIf9XnB5mz29r&view=vew5tEP7E3`
- 本地导出快照: `work/milvus_3_features.tsv`, 108 条记录

---

## 结论摘要

Milvus 3.0 功能不能只按“数据类型覆盖”理解。它横跨 schema evolution、StructArray/EmbeddingList、query/search 语义、text/analyzer、Storage V3、External Collection、Snapshot、Import 2PC、CDC、安全和运维能力。这会直接影响 brick 协议：新 brick 需要有 capability probe、版本门控、schema feature set、兼容模式、生命周期阶段和 skip 语义。

当前实现计划里的 P0 bricks 仍然成立，但参数和 manifest 应提前为 3.0 做扩展：

- `--feature-set`: 指定 `compat_2_6`, `milvus_3_core`, `struct_array`, `storage_v3`, `external_collection` 等功能集合。
- `--compat-mode`: 指定 `rollback_safe`, `upgrade_only`, `forward_only`。
- `--capability-probe`: 默认开启，运行前检测 server/SDK/API 支持能力。
- `--skip-unsupported`: 在低版本或 API 不可用时输出 `status=skipped`，而不是误判失败。
- `--lifecycle-phase`: 指定 `before_upgrade`, `after_upgrade`, `before_rollback`, `after_rollback`, `steady_state`。
- `--schema-matrix`: 不应只有版本维度，还要表达 feature tags、compatibility 和 required capabilities。

## Feishu Base 快照统计

导出视图中共有 108 条记录：

| 维度 | 数量 |
|---|---:|
| 用户可见 | 82 |
| 非用户可见 | 23 |
| 部分/暂不算 | 3 |
| 文档已覆盖 | 29 |
| 文档需更新 | 23 |
| 文档需新建-明显缺口 | 19 |
| 文档需新建/更新 | 3 |
| 文档无需-内部/插件/测试 | 23 |
| 已 track | 67 |
| 未 track 缺口 | 17 |
| 部分 track | 6 |
| Target 标注疑似有误 | 2 |

这些状态不等同于 test brick 优先级，但能提示哪些功能仍缺少清晰验收口径。对 bricks 来说，`用户可见` 且 `需新建/需更新/未track` 的功能要优先沉淀成 capability manifest。

## 功能域分组

### 1. StructArray / EmbeddingList

Feishu 相关记录约 23 条，是 3.0 覆盖里最大的一组。核心能力包括：

- StructArray 子字段名作用域放宽。
- StructArray 支持更多向量子字段类型：`FLOAT16_VECTOR`, `BFLOAT16_VECTOR`, `INT8_VECTOR`, `BINARY_VECTOR`。
- StructArray 内向量字段支持 `DISKANN`。
- embedding-list search + element-level filter。
- StructArray 标量子字段索引：`STL_SORT`, `bitmap`。
- 嵌套数组算子：`ARRAY_CONTAINS`, `ARRAY_CONTAINS_ALL`, `ARRAY_CONTAINS_ANY`, `ARRAY_LENGTH`。
- element-level query/search。
- element-level group_by，包括按 Struct 子字段和按主键 group_by。
- element-level range search / iterator search。
- element-level hybrid search。
- StructArray null 支持。
- StructArray 动态加字段。
- StructArray 更多向量类型 csv/json import。

**Brick 影响：**

- Schema manifest 需要支持 nested/struct field 描述，不能只表达 flat fields。
- Search brick 需要支持 `search_scope=entity|element`。
- Filter generator 需要支持 element-level filter、nested array operators 和 `$[...]` 访问。
- Validator 不能只按 entity count 验证，还要验证 element count、group_by 去重、element hit path。
- Upgrade/rollback 场景中，StructArray 应默认归为 `forward_only`，除非明确验证该 schema 可由 2.6 读写。

**建议 bricks：**

- `struct_array_schema_matrix`
- `struct_array_element_search`
- `struct_array_element_query`
- `struct_array_group_by`
- `struct_array_hybrid_search`
- `struct_array_import_matrix`

### 2. Query / Search Semantics

核心能力包括：

- Query 聚合：`count`, `min`, `max`, `sum`, `avg`。
- Query/Search 服务端 `ORDER BY`。
- Search by primary key。
- QueryIterator SDK 对齐。
- 多字段组合 group_by，当前部分 PR 仍在 segcore 层，需等 SDK/API 接入后再提升优先级。

**Brick 影响：**

- Query request 不能只表达 `expr/output_fields/limit`，还要表达 `aggregates`, `order_by`, `offset`, `pagination`, `group_by`。
- Validator 需要有 deterministic ground truth：聚合结果、排序稳定性、分页一致性。
- Search by PK 不需要 query vector，request schema 要允许 `search_mode=vector|pk|hybrid`。

**建议 bricks：**

- `query_aggregation_matrix`
- `query_order_by_matrix`
- `search_order_by_matrix`
- `search_by_pk`
- `query_iterator_consistency`

### 3. Text / Analyzer / Expression

核心能力包括：

- BM25 text highlighter。
- highlighter fragment 配置。
- query term highlighter。
- semantic highlighter。
- highlighter score。
- `run_analyzer`。
- ngram tokenizer + token chars。
- jieba/custom dictionary file resource。
- synonym filter。
- pinyin filter。
- Arabic/Thai tokenizer enhancement。
- phrase match slop 计算 API。
- regex filter operators: `=~`, `!~`。
- raw string literal: `r"..."`。
- bitwise operators: `&`, `|`, `^`。

**Brick 影响：**

- Data generator 需要生成多语言 text corpus，不只是 `Faker` 英文文本。
- Request 参数需要支持 analyzer/function resources，例如 dictionary file、synonym file。
- Filter expression generator 需要版本化，避免 2.6 环境误用 3.0 表达式。
- Validator 需要比较 highlight spans、score 字段、regex/bitwise ground truth。

**建议 bricks：**

- `text_analyzer_matrix`
- `text_highlight_matrix`
- `expression_operator_matrix`
- `phrase_slop_probe`
- `run_analyzer_request`

### 4. Schema / Index / DML Evolution

核心能力包括：

- TruncateCollection API。
- AlterCollectionSchema drop field。
- nullable vector。
- bulk import nullable vector。
- entity-level TTL。
- JSON Path Index 支持 `Sort`, `Bitmap`, `Hybrid`。
- `ARRAY_APPEND` / `ARRAY_REMOVE` partial update。
- TEXT LOB storage。

**Brick 影响：**

- Schema matrix 必须表达 mutable operations，不只是 create-time schema。
- Validator 需要知道字段生命周期：added, dropped, nullable, defaulted, expired。
- DML workload 需要支持 partial update operation family。
- Upgrade/rollback 中，drop field、TEXT LOB、Storage V3 相关能力要归到 `forward_only` 或独立 compat rule。

**建议 bricks：**

- `schema_evolution_matrix`
- `json_path_index_matrix`
- `array_partial_update`
- `truncate_collection`
- `entity_ttl_validation`
- `nullable_vector_matrix`

### 5. Storage / External / Import

官方 release notes 和 Feishu Base 都强调这一组是 3.0 的核心变化：

- External Collection：创建、禁止写操作、手动刷新、数据映射、加载查询。
- Collection Snapshot。
- Milvus snapshot 作为 External Table 数据源。
- Import 2PC：`CommitImport`, `AbortImport`, `auto_commit`。
- Storage V3 / Loon FFI。
- V3 segment manifest statistics。
- scalar index V3 serialization format。
- Force Merge compaction mode。

**Brick 影响：**

- Brick 不应假设所有 collection 都可写。External Collection 是 read-only 或受限写。
- Request 参数需要 `--source-kind`, `--source-uri`, `--refresh-policy`, `--snapshot-id`。
- Import 2PC 需要 transaction-like lifecycle：prepare/import/commit/abort/visibility validation。
- Storage V3 不能默认混入 rollback-safe schema；它需要单独 feature gate 和集群配置 gate。

**建议 bricks：**

- `external_collection_read_matrix`
- `collection_snapshot_matrix`
- `import_2pc_visibility`
- `force_merge_compaction`
- `storage_v3_probe`

### 6. Function / Model Provider

核心能力包括：

- Qwen rerank provider。
- Zilliz model provider。
- Yandex Cloud embedding provider。
- `add_function`, `alter_function`, `drop_function`。
- embedding function `batch_factor`。
- Function field backfill 系列。
- Arrow-based function chain pipeline，当前更多是内部实现。

**Brick 影响：**

- Function bricks 需要外部服务凭证和网络依赖，不能作为默认 P0。
- Request 协议需要支持 secret/env injection，而不是把 provider key 写进 CLI。
- Validator 需要区分模型输出的不确定性和 API 功能正确性。

**建议 bricks：**

- `function_lifecycle`
- `embedding_function_batch_factor`
- `rerank_provider_smoke`
- `function_backfill_probe`

### 7. Ops / Security / CDC

核心能力包括：

- CDC force promote。
- 切换后 salvage。
- `GetReplicateConfiguration`。
- Go SDK client telemetry。
- KMS key revoked 后拒绝读写。
- AWS KMS config hot update。
- Huawei OBS IAM/ST​​S credentials。
- RBAC user description。
- `/livez` liveness endpoint。

**Brick 影响：**

- 这些多为 environment/scenario bricks，不适合塞进普通 data workload。
- 需要支持 `--requires-cluster-admin`, `--requires-k8s`, `--requires-secret` 这类 capability 声明。
- Upgrade/rollback scenario 可以复用其中的 livez/telemetry/KMS probe 作为观察项。

**建议 bricks：**

- `livez_probe`
- `kms_revocation_guard`
- `cdc_failover_probe`
- `replicate_configuration_probe`
- `rbac_user_description`

## 覆盖优先级

### P0: 影响通用协议和升级回滚的基础能力

这些应先设计，哪怕不立即覆盖所有 3.0 功能：

- `capability_probe`
- `feature_inventory.yaml`
- `schema_matrix` with feature tags
- `deterministic_data_writer`
- `mixed_rw_pressure`
- `data_integrity_validator`
- `upgrade_rollback_compatibility`

P0 的目标是让后续所有 feature brick 都能挂到同一协议上。

### P1: 用户可见且影响 schema/query/search 的 3.0 核心能力

- StructArray / EmbList。
- Query Aggregation。
- Order By。
- Search by PK。
- Nullable Vector。
- JSON Path Index Sort/Bitmap/Hybrid。
- ARRAY partial update。
- Entity-level TTL。
- Text highlighter / analyzer / expression operators。
- External Collection。
- Snapshot。
- Import 2PC。

### P2: 需要特殊环境、凭证或集群配置的能力

- Storage V3 / Loon FFI。
- Function / model provider。
- KMS / CMEK。
- CDC failover / salvage。
- Huawei OBS IAM/ST​​S。
- Client telemetry。

P2 不代表不重要，而是需要单独环境和 secret 设计，不能阻塞 P0/P1。

## 对 Request 参数的设计影响

### 必须新增的公共参数

```text
--feature-set <name>
--compat-mode rollback_safe|upgrade_only|forward_only
--capability-probe true|false
--skip-unsupported true|false
--lifecycle-phase before_upgrade|after_upgrade|before_rollback|after_rollback|steady_state
--schema-matrix <path>
--workload-profile <path>
```

### 建议新增的 feature-specific 参数

```text
--search-mode vector|pk|hybrid|element
--filter-profile <name>
--index-profile <name>
--function-profile <name>
--source-kind collection|snapshot|external|object_storage
--source-uri <uri>
--refresh-policy manual|interval|none
--import-commit-mode auto|manual|abort
--requires-secret <name>
```

### Result JSON 必须扩展的字段

```json
{
  "feature_set": "struct_array",
  "compat_mode": "forward_only",
  "lifecycle_phase": "after_upgrade",
  "capabilities": {
    "server_version": "3.0.0-beta",
    "sdk_version": "2.6.x",
    "supported": ["StructArray", "OrderBy", "NullableVector"],
    "unsupported": ["StorageV3"]
  },
  "skip_reason": null
}
```

这样 Argo 可以区分 `failed` 和 `skipped`，也能把同一个 brick 跑在 2.6/3.0 两类集群上。

## Schema Matrix 设计影响

现有计划中的 `schema_matrix_2_6.yaml` 和 `schema_matrix_3_0.yaml` 需要升级为 feature-tagged 格式：

```yaml
version: "3.0"
schemas:
  - name: nullable_vector
    feature_tags: ["nullable_vector", "rollback_sensitive"]
    compat_mode: "forward_only"
    required_capabilities: ["NullableVector"]
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 128, nullable: true}
    validators:
      - count
      - null_vector_semantics

  - name: query_order_by
    feature_tags: ["order_by", "query"]
    compat_mode: "rollback_safe"
    required_capabilities: ["OrderBy"]
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: score, dtype: DOUBLE}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 128}
    validators:
      - order_by_ground_truth
```

注意：`rollback_safe` 不等于 “能覆盖所有 3.0 feature”。它只表示该 schema 和数据在回滚路径中应保持可读可写。任何 Storage V3、External Collection、3.0-only schema evolution 都应默认不是 rollback-safe。

## Workload 设计影响

Mixed RW 不能只随机 insert/query/search。它需要按 feature tags 动态选择操作：

| Feature | 写操作 | 读操作 | 验证 |
|---|---|---|---|
| Basic vector | insert/upsert/delete | search/query/iterator | count/checksum/search smoke |
| Nullable vector | insert null/non-null/upsert | search/query null filter | null semantics |
| StructArray | insert nested rows | element search/query/group_by | element hit path/group count |
| JSON path index | insert JSON variants | filter/order_by | JSON path ground truth |
| Text analyzer | insert multilingual text | match/phrase/regex/highlight | tokenizer/highlight expected |
| External collection | none or refresh only | query/search | read-only and mapping |
| Import 2PC | import/commit/abort | visibility query | commit visibility |

## 升级回滚设计影响

升级回滚 scenario 必须维护两个集合池：

1. `compat_pool`
   - 由 2.6 创建。
   - 升级到 3.0 后继续读写。
   - 回滚到 2.6 后仍必须验证通过。
   - 只允许 `compat_mode=rollback_safe` schema。

2. `forward_pool`
   - 升级到 3.0 后创建。
   - 覆盖 3.0-only 功能。
   - 回滚前需要停止或标记 archived。
   - 回滚后不作为 must-pass。

推荐 lifecycle：

```text
before_upgrade:
  create compat_pool
  seed deterministic data
  start mixed_rw_pressure on compat_pool
  start validator loop on compat_pool

after_upgrade:
  continue mixed_rw_pressure on compat_pool
  create forward_pool
  run 3.0 feature bricks
  validate compat_pool + forward_pool

before_rollback:
  stop forward_pool write/load-sensitive workloads
  snapshot forward_pool results
  keep compat_pool mixed_rw running

after_rollback:
  validate compat_pool only
  report forward_pool as skipped_due_to_rollback
```

## 建议新增 Manifest

### `feature_inventory.yaml`

```yaml
features:
  - id: struct_array_element_hybrid_search
    domain: struct_array
    source:
      - feishu:#49799
      - feishu:#50243
    user_visible: true
    priority: P1
    compat_mode: forward_only
    required_capabilities:
      - StructArray
      - ElementHybridSearch
    bricks:
      - struct_array_hybrid_search
    validators:
      - element_hit_path
      - hybrid_result_shape

  - id: query_aggregation
    domain: query_search
    source:
      - feishu:#44394
      - release_notes:3.0-beta
    user_visible: true
    priority: P1
    compat_mode: rollback_safe
    required_capabilities:
      - QueryAggregation
    bricks:
      - query_aggregation_matrix
    validators:
      - aggregate_ground_truth
```

### `capability_catalog.yaml`

```yaml
capabilities:
  - id: OrderBy
    detect:
      server_version_min: "3.0.0"
      sdk_probe: "client.query(..., order_by=...)"
    unsupported_behavior: skip

  - id: StorageV3
    detect:
      config_probe: "common.storage.useLoonFFI"
    unsupported_behavior: skip
    requires_cluster_admin: true
```

## 后续实现前置任务

在正式实现 P0 request bricks 前，先完成以下任务：

1. 创建 `feature_inventory.yaml`，把本文功能域转成机器可读 manifest。
2. 创建 `capability_catalog.yaml`，定义 server/SDK/config 探测方式。
3. 更新 `brick_catalog.yaml` schema，加入 `feature_tags`, `required_capabilities`, `compat_mode`, `lifecycle_phases`。
4. 更新公共参数 parser，加入 feature/capability/compat/lifecycle 参数。
5. 更新结果 JSON schema，加入 capabilities 和 skip_reason。
6. 再开始实现 `create_schema_matrix`, `seed_data`, `mixed_rw_pressure`, `validate_data_integrity`。

## 暂不首批覆盖但需要保留设计空间

- Function/model provider：需要外部 provider credentials。
- KMS/CMEK：需要云资源和密钥生命周期控制。
- CDC failover/salvage：需要主备复制环境。
- Storage V3：需要集群配置和可能的不可逆数据格式边界。
- External Collection：需要外部数据源或 snapshot fixture。
- Import 2PC：需要 import task lifecycle 管理。

这些能力不进入第一批 P0 实现，但必须从协议层支持 skip、capability、secret、source 和 lifecycle 表达。
