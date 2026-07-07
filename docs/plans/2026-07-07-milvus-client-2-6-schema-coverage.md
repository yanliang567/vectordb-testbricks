# Milvus Client 2.6 Schema Coverage 实现计划

**目标：** 用 3 个 rollback-safe collection 覆盖 Milvus 2.6 主要标量/向量类型、schema 形态、向量索引、标量索引，并保持 upgrade/rollback 闭环可验证。

**架构：** 扩展 `milvus_client` manifest schema，使 collection 级别可以表达 dynamic field、partition key、explicit partitions、BM25 function 和 auto_id。seed 阶段生成确定性数据并记录 checkpoint；auto_id collection 捕获服务端返回的真实主键用于后续校验。

**技术栈：** PyMilvus `MilvusClient`、YAML schema matrix、pytest、Argo WorkflowTemplate。

---

### 任务 1: 扩展 schema manifest 模型

**文件：**
- 修改: `milvus-bricks/milvus_client/common/schema.py`
- 测试: `milvus-bricks/milvus_client/tests/test_schema_manifest.py`

**步骤：**
1. 为 `FieldSpec` 增加 `is_partition_key`、`enable_analyzer`、`analyzer_params`。
2. 增加 `FunctionSpec`，支持 `BM25` 函数输入/输出字段。
3. 为 `SchemaSpec` 增加 `description`、`enable_dynamic_field`、`num_partitions`、`partitions`、`functions`。
4. builder 将这些配置映射到 PyMilvus schema/create_collection/index params。

### 任务 2: 扩展数据生成与 checkpoint

**文件：**
- 修改: `milvus-bricks/milvus_client/common/data.py`
- 修改: `milvus-bricks/milvus_client/requests/seed_data.py`
- 修改: `milvus-bricks/milvus_client/requests/validate_data_integrity.py`
- 修改: `milvus-bricks/milvus_client/common/validators.py`
- 测试: `milvus-bricks/milvus_client/tests/test_data_generation.py`
- 测试: `milvus-bricks/milvus_client/tests/test_seed_data.py`
- 测试: `milvus-bricks/milvus_client/tests/test_validate_data_integrity.py`

**步骤：**
1. dynamic field collection 插入额外动态字段。
2. BM25 function output 字段不在 insert row 中生成。
3. explicit partitions collection 按 PK 分片写入。
4. auto_id collection 捕获 insert IDs，checkpoint 记录真实 PK samples 和校验范围。
5. validator 支持 checkpoint 中的真实 PK samples。

### 任务 3: 扩展 pressure workload

**文件：**
- 修改: `milvus-bricks/milvus_client/common/workload.py`
- 测试: `milvus-bricks/milvus_client/tests/test_workload.py`

**步骤：**
1. search 覆盖每个向量列，BM25 function output 用文本 query。
2. auto_id collection 跳过 upsert/delete，避免破坏兼容性基线。
3. 保持 query/query_iterator 使用实际 primary field。

### 任务 4: 扩展 2.6 schema matrix

**文件：**
- 修改: `milvus-bricks/milvus_client/manifests/schema_matrix_2_6.yaml`
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`

**步骤：**
1. `scalar_dynamic_partition_key` 覆盖全部 2.6 标量/数组类型、dynamic field、partition key、标量索引。
2. `vector_autoid_bm25` 覆盖 auto_id、多向量列、BM25、HNSW/IVF_RABITQ/DISKANN/AUTOINDEX/BIN_IVF/SPARSE 索引。
3. `explicit_partitions_nullable` 覆盖显式多 partition、VARCHAR PK、nullable scalar。

### 任务 5: 固化 Argo 实跑修复

**文件：**
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback-rbac.yaml`
- 测试: `milvus-bricks/milvus_client/tests/test_argo_template.py`

**步骤：**
1. WorkflowTemplate 中 Python 命令统一使用 `python3 -m`。
2. RBAC 增加 `workflowtaskresults` create/patch。
3. 运行 pytest、Argo lint、dry-run 和 `git diff --check`。
