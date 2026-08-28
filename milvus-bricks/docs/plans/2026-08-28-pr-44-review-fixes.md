# PR #44 Review 修复实现计划

**目标：** 修正 PR #44 对 #52893 release contract 的过度声明、限制 Argo retry 到幂等基础设施模板，并恢复 Ruff format CI。

**架构：** 2.6 round-trip 使用完整原始 matrix 并明确建模为 `known_limitation`，由严格失败场景持续跟踪 #52893，但不参与 release gate。Cluster WorkflowTemplate 不再使用全局 retry；仅对可安全重入的 Helm deploy/patch 和 readiness wait 模板配置 `OnError` retry。

**技术栈：** YAML manifest、Argo Workflows v4、Python/pytest、Ruff、GitHub Actions。

---

### 任务 1：恢复 #52893 的完整失败合同

**文件：**
- 修改: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改: `milvus_client/tests/fixtures/upgrade_rollback_execution_paths_v1.yaml`
- 删除: `milvus_client/manifests/schema_matrix_2_6_round_trip.yaml`
- 修改: `milvus_client/common/schema.py`
- 修改: `milvus_client/tests/test_schema_manifest.py`

**步骤：**

1. 先把 manifest 测试改为要求 standalone/cluster 通用 2.6 round-trip 使用完整 `schema_matrix_2_6.yaml`、分类为 `known_limitation`、状态为 `unsupported`、渲染 eligibility 为 `false`。
2. 运行定向测试并确认旧 manifest 失败。
3. 修改 manifest，移除派生 matrix alias；同步 execution fixture。
4. 删除不再使用的派生 matrix 文件、schema loader 继承实现和对应测试，避免保留无消费者的合同机制。
5. 更新 README、计划和执行报告，不再将 9-schema 成功结果描述为 release-gate 证据。

### 任务 2：将 Argo retry 限制到幂等模板

**文件：**
- 修改: `milvus_client/tests/test_argo_template.py`
- 修改: `argo/cluster-upgrade-rollback.yaml`

**步骤：**

1. 先修改模板测试，要求不存在 `spec.templateDefaults`，要求 `run-brick`、`optional-run-brick`、pressure 等业务模板没有 retry。
2. 要求 `deploy-milvus`、`wait-milvus-ready`、`patch-milvus-image`、`patch-milvus-config` 各自配置相同的 `OnError` retry/backoff。
3. 运行定向测试确认全局 retry 版本失败。
4. 删除全局 retry 和 main DAG override，只在上述四个模板增加 retry。
5. 运行 Argo 模板测试和 `argo lint`。

### 任务 3：修复并复核 CI

**文件：**
- 格式化: `milvus_client/tests/test_argo_template.py`
- 更新: `docs/upgrade-rollback-gates/2026-08-27-index-contract-e2e-matrix-report.md`

**步骤：**

1. 运行 CI 使用的 `ruff format`，修复稳定复现的格式失败。
2. 运行完整 pytest、CI Ruff check/format、全目录离线 Argo lint 和 `git diff --check`。
3. 验证两条 2.6 场景渲染为完整 matrix 且 `release-gate-eligible=false`，验证所有业务模板没有 retry。
4. 提交并推送 PR #44，更新 PR 描述和 review 结论，等待 GitHub CI。
