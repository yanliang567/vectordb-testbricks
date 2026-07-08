# Schema Evolution Workload 实现计划

**目标：** 为 3.0 upgrade 阶段补充 schema evolution workload，覆盖 2.6 存量 collection 和 3.0 新建 collection 的 schema mutation、nullable update、upsert、read validation。

**架构：** 新增独立 request brick `schema_evolution_workload`，由 Argo 在 upgrade/post-upgrade 阶段显式调用。brick 读取 schema matrix 后对已有 collection 执行 capability-aware mutation，并输出结构化 result；workflow report 通过现有 result artifact 纳入最终报告。

**技术栈：** Python, PyMilvus `MilvusClient`, Argo WorkflowTemplate, pytest, YAML manifests。

---

### 任务 1: 新增 schema evolution 数据模型和最小测试

**文件：**
- 创建: `milvus-bricks/milvus_client/requests/schema_evolution_workload.py`
- 创建: `milvus-bricks/milvus_client/tests/test_schema_evolution_workload.py`

**步骤 1: 写失败测试**

覆盖：
- 对 2.6 存量 spec 生成 add-field/upsert/query/read validation 操作。
- 对带 analyzer text 字段的 collection 生成 add-function/drop-function 操作。
- drop field API 不存在时记录 skipped，而不是失败。

**步骤 2: 运行测试验证失败**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_schema_evolution_workload.py -v`

预期:
FAIL，因为 request module 尚未实现。

**步骤 3: 实现最小 brick**

实现：
- `add_args(parser)` 支持 `--schema-matrix`、`--rows-per-collection`、`--batch-size`、`--start-id`、`--target-existing-collections`。
- `run_schema_evolution(client, specs, collection_prefix, ...)` 返回 metrics。
- 对每个 collection 执行：
  - `add_collection_field(... nullable=True, default_value=...)`
  - 对显式 PK collection 执行 evolution range `upsert`
  - 对 nullable vector collection 执行 null-vector/non-null-vector update
  - `query`、`search`、`count` 校验
  - BM25 add function/drop function，如果 collection 有 analyzer VARCHAR 和 sparse output 字段
  - drop field 若 SDK 暂无 API，记录 skipped

**步骤 4: 运行测试验证通过**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_schema_evolution_workload.py -v`

预期:
PASS。

### 任务 2: 注册 brick 和 feature inventory

**文件：**
- 修改: `milvus-bricks/milvus_client/manifests/brick_catalog.yaml`
- 修改: `milvus-bricks/milvus_client/manifests/feature_inventory.yaml`
- 测试: `milvus-bricks/milvus_client/tests/test_brick_catalog.py`

**步骤 1: 添加 catalog 项**

新增 brick:
`schema_evolution_workload`

属性：
- category: `schema`
- milvus_versions: `["3.0"]`
- compat_mode: `forward_only`
- lifecycle_phases: `["after_upgrade", "steady_state"]`
- destructive: `optional`

**步骤 2: 挂到 3.0 feature**

将 `schema_evolution_workload` 加到：
- `nullable_vector`
- `geometry`
- `timestamptz`
- `entity_ttl`

**步骤 3: 验证 catalog**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_brick_catalog.py -v`

预期:
PASS。

### 任务 3: 扩展 Argo workflow

**文件：**
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `milvus-bricks/argo/standalone-3-0-upgrade-rollback.yaml`
- 测试: `milvus-bricks/milvus_client/tests/test_argo_template.py`

**步骤 1: 添加参数**

新增：
- `schema-evolution-existing-enabled`
- `schema-evolution-forward-enabled`

默认：
- 2.6 workflow: `schema-evolution-existing-enabled=false`,
  `schema-evolution-forward-enabled=false`
- 3.0 workflow: `schema-evolution-existing-enabled=true`,
  `schema-evolution-forward-enabled=false`

**步骤 2: 插入 DAG task**

2.6 workflow:
- 在 `validate-after-upgrade` 后，对 2.6 baseline collections 执行 existing schema evolution。
- 在 `validate-forward-after-upgrade` 后，对 3.0 forward collections 执行 forward schema evolution。
- rollback 前完成，rollback 后只校验 2.6 baseline；3.0 forward collection 不作为 2.6 rollback gate。

3.0 workflow:
- 在 `validate-after-upgrade` 后执行 existing schema evolution。
- 在 rollback 后继续 `validate-after-rollback` 和 rollback-forward validation。

**步骤 3: 更新测试**

断言 workflow template 包含：
- 新参数
- `schema_evolution_workload` 命令
- existing 和 forward 两条 schema matrix 路径

### 任务 4: 更新报告和验证

**文件：**
- 修改: `milvus-bricks/milvus_client/docs/reports/2026-07-08-milvus-config-matrix-implementation-report.md`

**步骤 1: 更新 coverage**

记录新增 coverage：
- 2.6 存量 collection schema evolution
- 3.0 新 collection schema evolution
- nullable vector null/non-null update
- add function/drop function
- add field/drop field skipped boundary
- upsert/search/query/count

**步骤 2: 运行验证**

运行:
`PYTHONPATH=. pytest milvus_client/tests -v`

运行:
`argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/upgrade-rollback-compatibility.yaml`

运行:
`git diff --check origin/main...HEAD`

预期:
全部通过。
