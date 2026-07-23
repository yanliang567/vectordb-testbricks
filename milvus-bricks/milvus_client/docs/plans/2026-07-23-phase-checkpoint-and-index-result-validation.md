# Phase checkpoint and index result validation 实现计划

**目标：** 防止升级后新增/变更数据在回滚过程中丢失但 gate 仍通过，并让索引兼容验证校验命中结果而不是只校验请求不报错。

**架构：** `validate_phase_dml_dql` 在 after-upgrade 阶段写入 phase checkpoint，记录 baseline 50000000 区间 DML 结果和 after-upgrade 新 collection 60000000 区间数据；after-rollback 阶段在执行 70000000/80000000 新写入前先验证该 checkpoint。`validate_index_compatibility` 对标量 query 和向量 search 的返回结果执行预期 PK / 距离断言。

**技术栈：** Python request bricks、pytest fake Milvus client、Argo WorkflowTemplate YAML、Milvus Python SDK 结果格式兼容解析。

---

### 任务 1: Phase checkpoint 失败测试

**文件：**
- 修改: `milvus_client/tests/test_validate_phase_dml_dql.py`

**步骤 1: 写 after-upgrade checkpoint 测试**

添加测试：执行 `phase=after-upgrade` 后必须生成 `phase_dml_dql_after_upgrade.json`，并记录 existing collection、new collection、PK 范围、delete 数量、upsert 样本和新 collection 行数。

**步骤 2: 写 rollback 前置验证负向测试**

添加测试：先运行 after-upgrade 写 checkpoint，再删除 fake client 中 50000000/60000000 区间数据，执行 `phase=after-rollback --validate-phase-checkpoint true`，预期失败且不继续执行新的 rollback DML 写入。

**步骤 3: 运行定向测试验证失败**

运行:
```bash
PYTHONPATH=. pytest milvus_client/tests/test_validate_phase_dml_dql.py -q
```

预期: 新增测试在实现前失败。

### 任务 2: Phase checkpoint 实现

**文件：**
- 修改: `milvus_client/requests/validate_phase_dml_dql.py`
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`

**步骤 1: 添加参数**

添加 `--phase-checkpoint-file` 和 `--validate-phase-checkpoint`。默认 checkpoint 路径为 `${checkpoint_dir}/phase_dml_dql_after_upgrade.json`。

**步骤 2: 记录 after-upgrade phase checkpoint**

在 after-upgrade 成功执行 DML/DQL 后写 checkpoint，包含：
- existing collection: schema、collection、primary field、start_id、rows、deleted count、remaining count、deleted sample、remaining sample、upsert validation field、upsert samples。
- new collection: schema、collection、primary field、start_id、rows、sample PK。

**步骤 3: rollback 先验证 checkpoint**

在 after-rollback 阶段且 `--validate-phase-checkpoint true` 时，先读取 checkpoint，验证 existing/new collection 的 count、deleted PK、remaining PK、upsert sample 值和 search。通过后再执行 rollback 阶段新增 DML。

**步骤 4: 更新三个 WorkflowTemplate**

after-upgrade task 传入 checkpoint 输出路径；after-rollback task 传入同一路径并开启 `--validate-phase-checkpoint true`。

### 任务 3: Index query/search 结果正确性

**文件：**
- 修改: `milvus_client/requests/validate_index_compatibility.py`
- 修改: `milvus_client/tests/test_validate_index_compatibility.py`

**步骤 1: 写 scalar query 负向测试**

fake client 返回空结果或错误 PK 时，预期 `INDEX_SCALAR_QUERY_FAILED`。

**步骤 2: 写 vector search 负向测试**

fake client 返回空结果或错误 PK 时，预期 `INDEX_SEARCH_FAILED`。

**步骤 3: 实现结果解析和断言**

新增 helper 兼容 SDK 返回格式，标量 query 必须包含目标 PK；向量 search 必须包含目标 PK，且可读取 distance 时按 metric 做自搜索距离 sanity check。

### 任务 4: 文档、验证、提交

**文件：**
- 修改: `milvus_client/docs/upgrade-rollback.md`
- 修改: `milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改: PR #9 description

**步骤 1: 运行验证**

```bash
PYTHONPATH=. pytest milvus_client/tests -q
uvx ruff check <changed-python-files>
uvx ruff format --check <changed-python-files>
source ~/.zshrc >/dev/null 2>&1 || true
argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/cluster-upgrade-rollback.yaml
git diff --check
```

**步骤 2: 提交并推送**

```bash
git add <changed-files>
git commit -s -m "test: validate rollback phase data continuity"
source ~/.zshrc >/dev/null 2>&1 || true
git push origin fix/rollback-index-validation
```

**步骤 3: 更新 PR 描述并检查 GitHub CI**

使用 `gh pr edit 9 --body-file ...` 同步覆盖说明和验证结果，随后检查 PR checks。
