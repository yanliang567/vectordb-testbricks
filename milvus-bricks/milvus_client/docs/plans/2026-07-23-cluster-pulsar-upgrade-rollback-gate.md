# Cluster Pulsar Upgrade Rollback Gate 实现计划

**目标：** 将 2.6.18 -> 3.0 latest -> 2.6 rollback 的 cluster gate 从 Woodpecker profile 切换到 Pulsar profile，避免覆盖 2.6 不支持的 external Woodpecker client 拓扑迁移。

**架构：** 新增一个 Helm 管理的 `cluster-pulsar-1cu` deploy profile，显式启用 Pulsar、禁用 Woodpecker，并继续复用代码化 Argo WorkflowTemplate。只切换 2.6 -> 3.0 -> 2.6 cluster 场景；3.0 -> 3.0 cluster 场景保留 Woodpecker 覆盖。

**技术栈：** Python renderer/validator、YAML manifest、Argo WorkflowTemplate、Milvus Helm Chart。

---

### 任务 1: 新增 Pulsar cluster deploy profile

**文件：**
- 创建: `milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml`
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`

**步骤 1: 写 profile**

创建 `cluster-pulsar-1cu.yaml`：

```yaml
name: cluster-pulsar-1cu
mode: cluster
deployer: helm
helm:
  repo_name: zilliztech
  repo_url: https://zilliztech.github.io/milvus-helm/
  chart: zilliztech/milvus
  chart_version: 5.0.24
helm_values:
  cluster:
    enabled: true
  streaming:
    enabled: false
  woodpecker:
    enabled: false
  pulsar:
    enabled: false
  pulsarv3:
    enabled: true
    persistence: false
    volumes:
      persistence: false
  kafka:
    enabled: false
components:
  mixCoord: ...
  proxy: ...
  queryNode: ...
  dataNode: ...
dependencies:
  msgStreamType: pulsar
  etcd:
    inCluster:
      deletionPolicy: Delete
      pvcDeletion: true
  storage:
    inCluster:
      deletionPolicy: Delete
      pvcDeletion: true
```

**步骤 2: 更新 manifest 引用**

在 `deploy_profiles` 增加：

```yaml
cluster_pulsar_1cu: milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml
```

将 2.6 -> 3.0 -> 2.6 的 cluster scenarios 的 `deploy_profile_ref` 改为 `cluster_pulsar_1cu`。

**步骤 3: 运行 profile 单测**

运行: `PYTHONPATH=. pytest milvus_client/tests/test_deploy_profiles.py -q`
预期: PASS。

### 任务 2: 更新 renderer/test 断言

**文件：**
- 修改: `milvus_client/tests/test_deploy_profiles.py`
- 修改: `milvus_client/tests/test_render_milvus_cr.py`
- 修改: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改: `milvus_client/tests/test_argo_template.py`

**步骤 1: 增加 Pulsar profile 断言**

验证：
- `dependencies.msgStreamType == "pulsar"`
- `helm_values.pulsar.enabled is False`
- `helm_values.pulsarv3.enabled is True`
- `helm_values.woodpecker.enabled is False`
- 2.6 cluster gate 渲染出的 `deploy-profile` 指向 Pulsar profile。

**步骤 2: 保留 Woodpecker 断言**

确保 3.0 -> 3.0 cluster 场景仍使用 `cluster-woodpecker-1cu.yaml`，避免丢失 Woodpecker 正向覆盖。

**步骤 3: 运行相关单测**

运行: `PYTHONPATH=. pytest milvus_client/tests/test_deploy_profiles.py milvus_client/tests/test_render_upgrade_rollback_params.py milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_milvus_cr.py milvus_client/tests/test_argo_template.py -q`
预期: PASS。

### 任务 3: 更新 WorkflowTemplate 默认值和文档

**文件：**
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 修改: `milvus_client/docs/upgrade-rollback.md`
- 可能修改: `milvus_client/docs/upgrade-rollback-gates/README.md`

**步骤 1: 修改 Cluster WorkflowTemplate 默认 deploy profile**

将默认 `deploy-profile` 从 `cluster-woodpecker-1cu.yaml` 改成 `cluster-pulsar-1cu.yaml`，使直接提交模板默认落在 2.6 兼容的 Pulsar topology。

**步骤 2: 更新文档**

说明：
- 2.6 -> 3.0 -> 2.6 cluster gate 使用 Pulsar，因为 2.6 不支持 external Woodpecker client。
- 3.0 -> 3.0 cluster gate 继续覆盖 external Woodpecker。

**步骤 3: 运行 YAML/lint 验证**

运行:

```bash
argo lint argo/cluster-upgrade-rollback.yaml
ruff check .
ruff format --check .
git diff --check
```

预期: 全部通过。

### 任务 4: 提交 PR

**文件：**
- 所有上述改动

**步骤 1: 查看 diff**

运行: `git diff --stat && git diff --check`
预期: diff 只包含 Pulsar profile、scenario 引用、测试和文档。

**步骤 2: 提交**

运行:

```bash
git add milvus_client/manifests milvus_client/tests argo milvus_client/docs
git commit -m "test: use pulsar for 2.6 cluster rollback gates"
```

**步骤 3: 创建 PR**

运行: `gh pr create --fill`
预期: PR 创建成功，描述包含验证结果和 standalone issue 调查结论摘要。
