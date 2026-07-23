# Phase New Collection Rows 3000 实现计划

**目标：** 将升级/回滚 phase 新建 collection 的默认插入量从 1000 提升到 3000，并保持 manifest、WorkflowTemplate、文档、测试和 PR 描述一致。

**架构：** 继续使用集中 manifest default 驱动 renderer 参数，三个 Argo WorkflowTemplate 默认参数同步为 3000。报告和 README 只记录参数与确定性数据量口径，不改变 old/carried collection 的 DML 语义。

**技术栈：** YAML manifest、Argo WorkflowTemplate、Python pytest、GitHub PR。

---

### 任务 1: 更新集中默认值

**文件：**
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`

**步骤：**
1. 将 `phase_new_collection_rows` / `phase-new-collection-rows` 默认值改为 `3000`。
2. 保持 `phase_existing_dml_rows=1000`、`phase_existing_delete_rows=100` 不变。

### 任务 2: 更新测试和文档

**文件：**
- 修改: `milvus_client/tests/test_argo_template.py`
- 修改: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改: `milvus_client/tests/test_generate_workflow_report.py`
- 修改: `milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改: `milvus_client/docs/upgrade-rollback.md`

**步骤：**
1. 测试断言改为 `3000`。
2. 数据量表按新建 collection `3000` rows/schema 重新计算。
3. 明确 old/carried collection 的 upsert/delete 当前作用于本 phase 新插入的 PK range。

### 任务 3: 验证和提交

运行：

```bash
PYTHONPATH=. pytest milvus_client/tests -q
uvx ruff check <changed python files>
uvx ruff format --check <changed python files>
argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/cluster-upgrade-rollback.yaml
git diff --check
```

预期：全部通过后提交并推送到 PR #9 分支。
