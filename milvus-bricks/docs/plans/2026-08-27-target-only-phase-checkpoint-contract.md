# Target-Only Phase Checkpoint 合同实现计划

**目标：** 防止 target-only index engine 场景把目标阶段新建索引 artifact 错误纳入 baseline 回滚兼容门禁，同时保留其他 DML-DQL 覆盖。

**架构：** 复用现有 `index-engine-contract-mode`。WorkflowTemplate 在 target-only 回滚前删除 phase new collections；rollback validator 根据合同验证 new group 必须不存在或必须 round-trip，existing group 始终严格验证。

**技术栈：** Python 3、pytest、PyYAML、Argo WorkflowTemplate、MilvusClient。

---

### 任务 1：锁定 validator 合同

**文件：**
- 修改: `milvus_client/tests/test_validate_phase_dml_dql.py`
- 修改: `milvus_client/requests/validate_phase_dml_dql.py`

**步骤 1：写失败测试**

- target-only checkpoint new collection 不存在时，existing group 正常 reload/query，new group 只记录 absent evidence。
- target-only checkpoint new collection 仍存在时，以专用 failure type 失败。
- target-only main path 不对 carried prefix 执行 DML，但继续 baseline existing 和 rollback new collection DML。
- round-trip 默认行为继续 reload existing/new 两组。

**步骤 2：运行测试验证失败**

```bash
python3 -m pytest milvus_client/tests/test_validate_phase_dml_dql.py -q -k 'target_only or phase_checkpoint_reloads'
```

预期：新 CLI 参数、metrics 和 target-only 分支尚不存在，测试失败。

**步骤 3：写最小实现**

- 增加 `--phase-checkpoint-new-collections-contract` choices。
- checkpoint contract 结构校验保持不变。
- target-only new group 使用 `has_collection` 验证 absent，不调用 release/load/query。
- main carried-DML 分支仅在合同非 target-only 时运行。

**步骤 4：运行 validator 测试**

```bash
python3 -m pytest milvus_client/tests/test_validate_phase_dml_dql.py -q
```

预期：全部通过。

### 任务 2：接入三套 WorkflowTemplate

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 修改: `milvus_client/tests/test_argo_template.py`

**步骤 1：写失败测试**

对每套模板断言：

- `drop-phase-new-collections` 使用 base `schema-matrix` 和 `${collection-prefix}_after_upgrade`。
- task 仅在 rollback + target_only 时运行。
- `drop-forward-schema` 依赖该 task。
- rollback phase validator 传入 `--phase-checkpoint-new-collections-contract {{workflow.parameters.index-engine-contract-mode}}`。

**步骤 2：运行测试验证失败**

```bash
python3 -m pytest milvus_client/tests/test_argo_template.py -q -k 'phase_dml_dql or index_engine'
```

**步骤 3：写最小 YAML 实现并验证**

将相同 DAG 变更应用到三套模板，保持 topology-specific deploy 逻辑不变。

### 任务 3：文档和路径影响

**文件：**
- 修改: `milvus_client/docs/upgrade-rollback.md`
- 修改: `docs/upgrade-rollback-gates/README.md`
- 修改: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

记录：target-only 的 forward 与 phase-new artifact 都不会进入 baseline rollback；round-trip 继续验证二者。四个 v10/v4、v11/v4 standalone/cluster 场景由现有 manifest mode 自动获得行为，无需新增 per-scenario 字段。

### 任务 4：完整验证并提交 PR

```bash
python3 -m pytest milvus_client/tests -q
python3 -m ruff check <CI 白名单文件>
python3 -m ruff format --check <CI 白名单文件>
argo lint --offline argo
git diff --check
```

预期：测试、lint、format、三套 WorkflowTemplate lint 全部通过；单个 DCO commit 基于 PR #42 merge 后的 `main`，PR 关联 `milvus-io/milvus#52767` 和 canary `milvus-standalone-3-0-upgrade-rollback-pbzt6`。
