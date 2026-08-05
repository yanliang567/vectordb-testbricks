# Woodpecker 2CU HA Upgrade/Rollback Gate 实现计划

**目标：** 新增一条 3.0 Woodpecker 2CU 多副本集群滚动升级/回滚 gate，并防止场景被单副本 deploy profile 覆盖后产生假绿。

**架构：** 继续复用 `milvus-cluster-upgrade-rollback` WorkflowTemplate。场景通过 `cluster-woodpecker-2cu.yaml` 声明多副本拓扑，`common/gates.py` 在 render 和 workflow runtime 两个入口校验最小副本要求；现有 pressure、serviceability、数据、索引和 phase DML/DQL 门禁保持不变。

**技术栈：** Python、pytest、PyYAML、Argo WorkflowTemplate、Helm、Milvus 3.0、Woodpecker。

---

### 任务 1：固化 2CU HA 拓扑契约

**文件：**

- 修改：`milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml`
- 修改：`milvus_client/common/gates.py`
- 测试：`milvus_client/tests/test_deploy_profiles.py`
- 测试：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 测试：`milvus_client/tests/test_render_milvus_cr.py`

**步骤：**

1. 先写失败测试，要求 2CU profile 的 `proxy`、`queryNode`、`dataNode`、`streamingNode` 均为至少 2 副本。
2. 为 gate scenario 增加 `topology_requirements.min_replicas`，并校验字段为正整数。
3. `resolve_gate_scenario()` 加载 deploy profile 后验证最小副本；使用 1CU override 时必须失败。
4. Helm values 和 topology summary 测试必须确认 2CU 副本数被实际渲染。

### 任务 2：新增 Woodpecker 2CU HA gate scenario

**文件：**

- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 测试：`milvus_client/tests/test_render_upgrade_rollback_params.py`

**场景：**

`cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline`

**步骤：**

1. 使用 3.0 baseline image、3.0 schema matrix 和 Woodpecker 2CU profile。
2. 升级到 3.0 latest 后回滚到相同 3.0 baseline。
3. 保持 LoonFFI、Vortex 和 JSON Shredding 关闭，隔离多副本滚动升级变量。
4. 启用 strict data/serviceability/pressure、index compatibility、phase DML/DQL 和 schema evolution gate。
5. 使用短 `submit_generate_name`，避免 Argo 生成名超过 Kubernetes 限制。

### 任务 3：增加 workflow runtime 拓扑防线

**文件：**

- 修改：`argo/cluster-upgrade-rollback.yaml`
- 测试：`milvus_client/tests/test_argo_template.py`

**步骤：**

1. `deploy-milvus` clone repo 后，根据 `scenario-id` 和实际 `deploy-profile` 调用中心化 scenario resolver。
2. 已注册的 HA scenario 若被直接提交为 1CU profile，必须在 Helm deploy 前失败。
3. 未注册的手工 scenario 保持现有行为，不强制依赖 gate manifest。
4. 不新增 WorkflowTemplate，不增加零请求失败 SLO。

### 任务 4：更新执行文档

**文件：**

- 修改：`docs/upgrade-rollback-gates/README.md`
- 修改：`milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改：`docs/plans/2026-08-04-upgrade-rollback-gate-hardening.md`

**步骤：**

1. 场景总数更新为 10 条，其中 9 条 promoted gate、1 条 negative。
2. 记录 2CU gate 的多副本组件和不包含 availability SLO 的边界。
3. 增加 2CU gate 的参数渲染示例。
4. 将总计划任务 5 标记为已具体化实现。

### 任务 5：验证和提交

运行：

```bash
PYTHONPATH=. pytest -q milvus_client/tests
argo lint --offline milvus-bricks/argo
uvx --from ruff==0.15.22 ruff check <changed-python-files>
uvx --from ruff==0.15.22 ruff format --check <changed-python-files>
git diff --check
```

验收：

- 10 条场景全部 resolve，并向各自 WorkflowTemplate 渲染完整参数。
- 2CU scenario 使用 1CU profile override 时失败。
- Cluster Helm values 和 topology summary 均包含要求的多副本。
- 不触发真实 4am workflow。
