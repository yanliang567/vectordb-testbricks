# Upgrade/Rollback Gates And Cluster Mode 实现计划

**目标：** 将 Milvus upgrade/rollback gate 场景固化，并新增 cluster mode 覆盖；Milvus deployment spec 改为代码化 profile，不再只在 Workflow 脚本里内联。

**架构：** WorkflowTemplate 负责编排，`deploy_profile` 描述 Milvus CR 拓扑，`render_milvus_cr` 负责渲染最终 CR。Standalone 和 cluster 复用同一套 schema/seed/validation/pressure/serviceability/report 逻辑。

**技术栈：** Argo WorkflowTemplate、Milvus Operator CRD、YAML deploy profile、Python renderer、pytest。

---

### 任务 1: 固化 gate 场景

**文件：**

- 创建: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

**实现要点：**

- `2.6.18 -> 3.0 latest -> rollback 2.6.18` 是正式支持路径。
- `2.6.18 -> 3.0 latest -> rollback latest 2.6` 是正式支持路径。
- 上述两个路径升级到 3.0 后必须禁用 storage v3 和 vortex，否则 rollback 到 2.6 存在 panic 风险。
- `3.0 baseline -> 3.0 latest -> rollback 3.0 baseline` 是 3.0 branch gate。
- LoonFFI/vortex rollback 到 2.6 保留为 negative coverage。

**验证：**

```bash
pytest milvus_client/tests/test_upgrade_rollback_gates_manifest.py -v
```

### 任务 2: 代码化 Milvus deploy profile

**文件：**

- 创建: `milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml`
- 创建: `milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml`
- 创建: `milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml`
- 测试: `milvus_client/tests/test_deploy_profiles.py`

**实现要点：**

- standalone profile 保持当前 RocksMQ + in-cluster etcd/minio。
- cluster profile 使用 Woodpecker，显式声明 mixCoord/proxy/queryNode/dataNode/streamingNode。
- profile 中 requests/limits、dependencies deletionPolicy/pvcDeletion 必须显式。

**验证：**

```bash
pytest milvus_client/tests/test_deploy_profiles.py -v
```

### 任务 3: 实现 Milvus CR renderer

**文件：**

- 创建: `milvus_client/common/deploy.py`
- 创建: `milvus_client/requests/render_milvus_cr.py`
- 测试: `milvus_client/tests/test_render_milvus_cr.py`

**实现要点：**

- 输入 deploy profile、image/version、config 开关、workflow owner metadata。
- 输出 Milvus CR YAML。
- 输出 `deploy_topology.json`，供最终 report 使用。

**验证：**

```bash
pytest milvus_client/tests/test_render_milvus_cr.py -v
```

### 任务 4: 改造 standalone workflow 使用 deploy profile

**文件：**

- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**实现要点：**

- 新增 `scenario-id` 和 `deploy-profile` 参数。
- `deploy-milvus` 调用 `milvus_client.requests.render_milvus_cr`。
- `generate-final-report` 传入 scenario/profile/topology。

**验证：**

```bash
pytest milvus_client/tests/test_argo_template.py -v
argo template lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml -n qa
```

### 任务 5: 新增 cluster workflow

**文件：**

- 创建: `argo/cluster-upgrade-rollback.yaml`
- 创建: `argo/cluster-upgrade-rollback-rbac.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**实现要点：**

- 默认 `deploy-profile=milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml`。
- 复用 standalone DAG 的验证、压力、serviceability、report、cleanup 逻辑。
- 报告记录 cluster topology。

**验证：**

```bash
pytest milvus_client/tests/test_argo_template.py -v
argo template lint argo/cluster-upgrade-rollback.yaml -n qa
```

### 任务 6: 集中化 gate 场景参数渲染

**文件：**

- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 创建: `milvus_client/common/gates.py`
- 创建: `milvus_client/requests/render_upgrade_rollback_params.py`
- 创建: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改: `milvus_client/docs/upgrade-rollback.md`

**实现要点：**

- 在 manifest 中新增 `image_aliases`，场景只引用 alias，避免每个 scenario 重复写 image/version。
- 每个 scenario 显式声明 `workflow_template`，submit 时可以从 `scenario-id` 直接生成 Argo 参数。
- 新增 cluster gate 场景，复用同一套 scenario manifest。
- `2.6 -> 3.0 -> 2.6` gate 校验必须拒绝 storage v3/vortex；LoonFFI 和 vortex 参数保持显式分离，避免未来场景静默丢配置。
- CLI 输出 JSON/YAML 或 `argo submit` 参数片段，后续改 3.1/4.0/version 主要改 manifest 中 alias/scenario 和必要的 schema/deploy profile。

**验证：**

```bash
pytest milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_upgrade_rollback_params.py -v
python -m milvus_client.requests.render_upgrade_rollback_params --scenario-id standalone-2-6-18-to-3-0-latest-rollback-2-6-18 --format json
```
