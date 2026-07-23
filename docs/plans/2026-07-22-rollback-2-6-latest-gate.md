# Rollback 2.6 Latest Gate 实现计划

**目标：** 将 `2.6.18 -> 3.0 latest -> 2.6` 正式 gate 的 rollback target 从 `v2.6.18` 调整为包含 #50792 修复的 `2.6 latest`。

**架构：** 保留 `v2.6.18` 作为 baseline 数据构造版本；升级阶段仍使用 `3.0 latest` 且禁用 LoonFFI/StorageV3/Vortex；回滚阶段使用 `2.6 latest` 作为正向 gate。`rollback 2.6.18` 只作为已知不兼容/诊断路径记录，不作为发布 gate。

**技术栈：** Argo WorkflowTemplate、Milvus Operator CR、Python manifest renderer、pytest、4am Harbor image。

---

### 任务 1: 调整 gate manifest 语义

**文件：**
- 修改: `milvus-bricks/milvus_client/manifests/upgrade_rollback_gates.yaml`

**步骤 1:** 将 standalone/cluster 的 `rollback-2-6-18` 场景从正式 gate 语义调整为 known unsupported/diagnostic，说明 #50694 / #50792 背景。

**步骤 2:** 保持 standalone/cluster 的 `rollback-2-6-latest` 为正式 gate。

**步骤 3:** 确保 `milvus-2-6-latest` 仍通过 image alias 集中维护，便于后续只替换一个 alias 或提交参数。

### 任务 2: 更新 README 和执行示例

**文件：**
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback-gates/README.md`
- 视情况修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`
- 视情况修改: `milvus-bricks/milvus_client/README.md`

**步骤 1:** 将正式 gate 表格和示例切到 `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest`。

**步骤 2:** 增加说明：`rollback to v2.6.18` 会命中 #50694，只有包含 #50792 的 2.6 构建可作为正向 rollback target。

**步骤 3:** 更新手工 `argo submit` 示例的 rollback image/version 为 2.6 latest placeholder 说明。

### 任务 3: 更新 Argo 默认参数

**文件：**
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`
- 视情况修改: `milvus-bricks/argo/cluster-upgrade-rollback.yaml`

**步骤 1:** 将 2.6 rollback template 默认 rollback image 从 `v2.6.18` 改为 `2.6-latest-placeholder` 或文档推荐通过提交参数覆盖。

**步骤 2:** 保持运行时 storage safety guard 不变。

### 任务 4: 运行本地验证

**命令：**
```bash
cd milvus-bricks
python3 -m pytest milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_upgrade_rollback_params.py -q
ruff check milvus_client
argo template lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/cluster-upgrade-rollback.yaml -n qa -o simple
```

**预期：** pytest、ruff、Argo lint 均通过。

### 任务 5: 更新 live WorkflowTemplate 并跑 smoke

**步骤 1:** 查询 4am Harbor 最新 2.6 和 3.0 deployable image。

**步骤 2:** 更新 live Argo WorkflowTemplate。

**步骤 3:** 提交 standalone smoke：

```text
2.6.18 baseline -> 3.0 latest -> 2.6 latest
```

**步骤 4:** 观察到 terminal 状态，收集 pass/fail 证据；如果 standalone 通过，再提交 cluster smoke。
