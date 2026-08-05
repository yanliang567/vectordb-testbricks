# Cluster Upgrade/Rollback Gate Parity 实现计划

**目标：** 为现有 standalone target-only feature gate 和 JSON Shredding gate 增加 cluster 模式对应场景，补齐功能场景在两种部署模式下的覆盖。

**架构：** 继续复用 `milvus-cluster-upgrade-rollback` WorkflowTemplate。新增场景只通过中心化 gate manifest 选择 Pulsar 1CU 或 Woodpecker 1CU profile，并复用已存在的 forward schema、post-config rollout、数据、索引、serviceability、phase DML/DQL 和报告门禁。

**技术栈：** Python、pytest、PyYAML、Argo WorkflowTemplate、Helm、Milvus 2.6/3.0。

---

### 任务 1：新增 cluster target-only feature gate

**文件：**

- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 测试：`milvus_client/tests/test_render_upgrade_rollback_params.py`

**场景：**

`cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest`

**步骤：**

1. 使用 Pulsar 1CU profile 和 `schema_matrix_2_6.yaml` 创建 rollback-safe baseline 数据。
2. 升级到 3.0 后使用 `schema_matrix_3_0.yaml` 创建 forward-only collections。
3. target 阶段要求 forward 数据、索引、search/query 和 schema evolution 通过。
4. 回滚到 2.6 后仅要求 baseline contract；`rollback_forward_validation_enabled=false`。
5. 使用短 `submit_generate_name`，避免 Helm/Pulsar release 资源名过长。

### 任务 2：新增 cluster JSON Shredding gate

**文件：**

- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 测试：`milvus_client/tests/test_render_upgrade_rollback_params.py`

**场景：**

`cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`

**步骤：**

1. 使用 Woodpecker 1CU profile，base 和 target 初始关闭 JSON Shredding。
2. 升级完成后通过 Helm post-config rollout 开启 JSON Shredding。
3. 配置 assertion 通过后创建并写入 JSON-heavy forward collections。
4. 回滚到 3.0 baseline 时保持 JSON Shredding 开启。
5. 回滚后要求 forward JSON 数据、动态字段、JSON path index 和查询继续通过。

### 任务 3：更新场景文档

**文件：**

- 修改：`docs/upgrade-rollback-gates/README.md`
- 修改：`milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改：`docs/plans/2026-08-04-upgrade-rollback-gate-hardening.md`

**步骤：**

1. 场景总数更新为 12 条，其中 11 条 promoted gate、1 条 negative。
2. 场景表增加 cluster target-only 和 cluster JSON Shredding。
3. 增加两条场景的参数渲染示例。
4. 记录剩余工作转为真实环境执行验证和独立 availability SLO 设计。

### 任务 4：验证和提交

运行：

```bash
PYTHONPATH=. pytest -q milvus_client/tests
argo lint --offline milvus-bricks/argo
uvx --from ruff==0.15.22 ruff check <changed-python-files>
uvx --from ruff==0.15.22 ruff format --check <changed-python-files>
git diff --check
```

验收：

- 12 条场景全部 resolve，并向对应 WorkflowTemplate 渲染 43 个参数。
- cluster target-only 场景在 2.6 rollback 后不要求 3.0-only forward collections。
- cluster JSON Shredding 场景在 post-config rollout 后写入，并在 rollback 后继续验证 forward 数据和索引。
- 不触发真实 4am workflow。
