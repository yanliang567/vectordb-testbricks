# Phase DML/DQL Validation 实现计划

**目标：** 删除已确认不支持的 `2.6.18 -> 3.0 latest -> rollback 2.6.18` path，并在升级后、回滚后覆盖新 collection、老 collection DML，以及新老 collection DQL。

**架构：** 新增独立 request brick `validate_phase_dml_dql`，每个阶段创建阶段专属新 collection，并对 baseline checkpoint 中的老 collection 执行 insert/upsert/delete 后做 query/search。WorkflowTemplate 通过集中参数控制该验证，final report 将 after-upgrade 和 after-rollback 结果设为 required validation。

**技术栈：** Python request brick、MilvusClient、Argo WorkflowTemplate、YAML gate manifest、pytest。

---

### 任务 1: 删除不支持的 2.6.18 rollback path

**文件：**
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改: `milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改: `milvus_client/docs/upgrade-rollback.md`

**步骤 1: 修改 manifest**

删除以下两个 diagnostic scenario：

- `standalone-2-6-18-to-3-0-latest-rollback-2-6-18`
- `cluster-2-6-18-to-3-0-latest-rollback-2-6-18`

保留支持路径：

- `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest`
- `cluster-2-6-18-to-3-0-latest-rollback-2-6-latest`

**步骤 2: 更新测试**

运行：

```bash
PYTHONPATH=. pytest milvus_client/tests/test_upgrade_rollback_gates_manifest.py -q
```

预期：manifest 不再要求两个 diagnostic scenario。

---

### 任务 2: 新增 phase DML/DQL request brick

**文件：**
- 创建: `milvus_client/requests/validate_phase_dml_dql.py`
- 创建: `milvus_client/tests/test_validate_phase_dml_dql.py`

**步骤 1: 写测试**

覆盖：

- 对 existing collection 执行 insert、upsert、delete，并查询样本字段验证 upsert 确实更新。
- 对 new collection create/index/load/insert。
- 对 existing 和 new collection 都执行 count/query/search。
- auto-id collection 会 insert/delete returned ids，并显式 skip upsert。

**步骤 2: 实现 brick**

参数：

- `--schema-matrix`
- `--checkpoint-file`
- `--phase`
- `--existing-collection-prefix`
- `--new-collection-prefix`
- `--new-collection-rows`
- `--existing-dml-rows`
- `--existing-delete-rows`
- `--batch-size`
- `--existing-start-id`
- `--new-start-id`
- `--drop-new-collections-if-exist`

默认数据规模：

- phase 新 collection：每个 schema 3000 rows。
- 老 collection DML：每个 schema insert 1000 rows，upsert 同一批 PK，delete 100 rows。

---

### 任务 3: 接入三个 WorkflowTemplate

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 修改: `milvus_client/tests/test_argo_template.py`

**步骤 1: 新增 workflow 参数**

- `phase-dml-dql-validation-enabled`: `"true"`
- `phase-new-collection-rows`: `"3000"`
- `phase-existing-dml-rows`: `"1000"`
- `phase-existing-delete-rows`: `"100"`

**步骤 2: 新增 DAG task**

升级后：

- `validate-phase-dml-dql-after-upgrade`
- 依赖 `validate-index-compatibility-after-upgrade`
- `strict-pressure-after-upgrade` 改依赖该 task

回滚后：

- `validate-phase-dml-dql-after-rollback`
- 依赖 `validate-index-compatibility-after-rollback`
- `validate-after-rollback` 改依赖该 task

**步骤 3: 更新 report 参数透传**

将 phase DML/DQL 参数写入 env/flow summary 和 `generate_workflow_report` CLI。

---

### 任务 4: renderer/report/docs

**文件：**
- 修改: `milvus_client/common/gates.py`
- 修改: `milvus_client/requests/generate_workflow_report.py`
- 修改: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改: `milvus_client/tests/test_generate_workflow_report.py`
- 修改: `milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改: `milvus_client/docs/upgrade-rollback.md`

**步骤 1: manifest renderer**

defaults 增加：

- `phase_dml_dql_validation_enabled: true`
- `phase_new_collection_rows: 3000`
- `phase_existing_dml_rows: 1000`
- `phase_existing_delete_rows: 100`

renderer 输出对应 Argo 参数。

**步骤 2: final report**

当 rollback enabled 且 phase DML/DQL validation enabled 时，required validation 增加：

- `validate_phase_dml_dql_after_upgrade`
- `validate_phase_dml_dql_after_rollback`

报告 config matrix 记录 phase 数据规模。

---

### 任务 5: 验证和提交

运行：

```bash
PYTHONPATH=. pytest milvus_client/tests -q
uvx ruff check milvus_client/common/gates.py milvus_client/requests/generate_workflow_report.py milvus_client/requests/validate_index_compatibility.py milvus_client/requests/validate_phase_dml_dql.py milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py milvus_client/tests/test_render_upgrade_rollback_params.py milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_validate_index_compatibility.py milvus_client/tests/test_validate_phase_dml_dql.py
uvx ruff format --check milvus_client/common/gates.py milvus_client/requests/generate_workflow_report.py milvus_client/requests/validate_index_compatibility.py milvus_client/requests/validate_phase_dml_dql.py milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py milvus_client/tests/test_render_upgrade_rollback_params.py milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_validate_index_compatibility.py milvus_client/tests/test_validate_phase_dml_dql.py
source ~/.zshrc >/dev/null 2>&1 || true
argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/cluster-upgrade-rollback.yaml
git diff --check
```

预期：测试、scoped lint/format、Argo lint、diff check 全部通过。
