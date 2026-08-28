# Index Contract E2E Matrix Validation 实现计划

**目标：** 串行完成剩余 index-engine contract E2E 矩阵，必要时修复并重验，最终生成可审计的 Markdown 执行报告。

**架构：** 以 manifest v2 为参数唯一来源，所有正式 workflow 使用 PR #43 merge commit 和同一目标 Milvus 镜像。每个场景在提交前验证 WorkflowTemplate、渲染参数、不可变 baseline/rollback 镜像和 QA 空闲状态；场景结束后验证关键节点、合同指标、pressure gate 与资源清理，再进入下一步。

**技术栈：** Python 3、pytest、Milvus Operator、Helm、Argo Workflows v4、kubectl、Milvus Bricks manifest renderer。

---

### 任务 1：Standalone v11/v4 target-only

**文件：**
- 读取: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 应用: `argo/standalone-3-0-upgrade-rollback.yaml`
- 场景: `standalone-3-0-index-v11-v4-upgrade-rollback`

**步骤：**

1. 运行 manifest renderer，覆盖 `repo-revision=c7c7b6901bef458906d7fdcd24bfb1e0e8f3f05f` 和不可变 target image。最初使用 `harbor.milvus.io/milvusdb/milvus:3.0-20260826-e47a679a`；该 tag 被 Harbor 清理后，cluster 重验改用最接近且不含 Woodpecker #52904 变更的 `harbor.milvus.io/milvusdb/milvus:3.0-20260827-f78de400@sha256:40aeeec2c833ccb695690637d62379aeb152f9c656cc3f8f9bd58bf1b45c35e3`。
2. 断言合同为 `target_only`、vector/scalar version 为 `11/4`、baseline/rollback 为 digest-pinned v3.0.0、log level 为 debug。
3. 确认 QA 无同模板 active workflow，应用并读取集群 WorkflowTemplate，确认 phase-new drop task 和 rollback contract 参数存在。
4. 提交并监控至终态。
5. 验证 forward index、11/11 phase-new drop、rollback absent/present、baseline reload、carried skip、rollback-new、pressure、final gate 和 cleanup。

### 任务 2：Cluster v10/v4 target-only

**文件：**
- 应用: `argo/cluster-upgrade-rollback.yaml`
- 场景: `cluster-3-0-index-v10-v4-upgrade-rollback`

**步骤：**

1. 用 renderer 生成 cluster v10/v4 参数并验证 immutable images、`target_only`、`10/4`、Pulsar 1CU profile 和 debug log。首次执行发现 v3.0.0 Woodpecker client v0.1.33 无法读取 3.0 branch v0.1.37+ 留下的 reader temp metadata，因此 index 专项路径切换到 Pulsar，将消息存储兼容性留给独立 Woodpecker gate。
2. 确认 QA 无同模板 active workflow，应用并读取 cluster WorkflowTemplate。
3. 提交并监控至终态。
4. 用与任务 1 相同的合同指标验收，并额外验证 Helm/cluster rollout 和 cleanup。

### 任务 3：Cluster v11/v4 target-only

**文件：**
- 场景: `cluster-3-0-index-v11-v4-upgrade-rollback`

**步骤：**

1. 复用已验证的 cluster 模板，重新渲染并逐项验证 v11/v4 参数。
2. 在任务 2 清理完成且无 active workflow 后提交。
3. 验证完整 target-only 合同、pressure 与 cleanup，补齐 capability × topology 矩阵。

### 任务 4：Standalone 2.6 `none` 合同控制场景

**文件：**
- 应用: `argo/standalone-2-6-upgrade-rollback.yaml`
- 场景: `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest`

**步骤：**

1. 从已验证记录或镜像查询解析 concrete 2.6 rollback image，禁止正式提交 placeholder/mutable image。
2. 渲染参数并验证 contract 为 `none`、phase-new drop task 条件结果为 Skipped、`drop-forward-schema` 下游仍可继续。
3. 提交并监控至终态。
4. 验证 phase checkpoint existing/new 均 reload/query、carried DML 未跳过、rollback-new 正常、pressure/final gate/cleanup 通过。

### 任务 5：失败诊断与条件修复

**文件：**
- 按实际根因确定，禁止预先扩大修改范围。

**步骤：**

1. 任一场景失败时读取 workflow node、brick JSON、pod/Milvus debug logs 和资源状态。
2. 使用 `systematic-debugging` 复现并定位根因；测试问题先写失败回归测试。
3. 实现最小修复，运行定向测试、全量 pytest、CI Ruff 白名单、Argo lint 和 `git diff --check`。
4. 用修复 commit/template 重跑当前失败步骤，只有通过后才进入下一个任务。

### 任务 6：生成执行报告并完成验证

**文件：**
- 创建: `docs/upgrade-rollback-gates/2026-08-27-index-contract-e2e-matrix-report.md`

**步骤：**

1. 记录 PR #43、代码 revision、镜像、模板 generation、每个 workflow、耗时、关键指标、pressure 分类、cleanup 和任何中间异常。
2. 汇总代码/模板改动；若无额外修复，明确报告只有文档新增。
3. 运行 Markdown/diff 检查；若有代码改动，重新运行完整 CI 等价验证。
4. 最后重新查询所有 workflow 终态和 QA 残留资源，以新鲜证据完成报告。
