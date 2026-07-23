# Cluster Helm Woodpecker Upgrade/Rollback 实现计划

**目标：** 将 cluster Woodpecker upgrade/rollback gate 从 Milvus Operator CR 部署切换到 Helm chart 部署，并提交包含 `2.6 latest rollback` gate 调整的新 PR。

**架构：** standalone gate 继续使用现有 Operator CR 路径；cluster gate 使用 code-managed deploy profile 渲染 Helm values，再通过 `helm upgrade --install` 部署和变更镜像/配置。Workflow 内的 wait/snapshot/cleanup 需要按 Helm release 资源和 service 状态工作，不再依赖 Milvus CR status。

**技术栈：** Argo WorkflowTemplate、Milvus Helm chart、Python YAML renderer、pytest、kubectl/helm。

---

### 任务 1: 保留已验证的 2.6 latest rollback gate 调整

**文件：**
- 修改: `milvus-bricks/milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`
- 修改: `milvus-bricks/milvus_client/tests/test_*.py`

**步骤 1: 确认正式 gate 只把 `2.6.18 -> 3.0 latest -> 2.6 latest` 作为 positive gate**

运行:
```bash
PYTHONPATH=. python3 -m pytest milvus_client/tests/test_upgrade_rollback_gates_manifest.py -q
```

预期: 相关 manifest 测试通过，`rollback-2-6-18` 为 diagnostic/unsupported_known_issue。

### 任务 2: 增加 Helm deploy profile 渲染能力

**文件：**
- 修改: `milvus-bricks/milvus_client/common/deploy.py`
- 修改: `milvus-bricks/milvus_client/requests/render_milvus_cr.py`
- 测试: `milvus-bricks/milvus_client/tests/test_render_milvus_cr.py`
- 测试: `milvus-bricks/milvus_client/tests/test_deploy_profiles.py`

**步骤 1: 扩展 deploy profile schema**

允许 profile 增加:
```yaml
deployer: helm
helm:
  chart: milvus/milvus
  repo_name: milvus
  repo_url: https://zilliztech.github.io/milvus-helm/
```

**步骤 2: 渲染 Helm values**

增加 renderer，将 image/version/config 和 workflow labels 注入 values:
```yaml
image:
  all:
    repository: harbor.milvus.io/milvusdb/milvus
    tag: <tag>
extraConfigFiles:
  user.yaml: |+
    common:
      storage:
        jsonShreddingEnabled: false
        useLoonFFI: false
```

Vortex 关闭时不写 `dataNode.storage.format`；开启时写 `vortex`。

**步骤 3: 写/更新测试**

验证 Helm profile:
- 能渲染 `cluster.enabled=true`
- `woodpecker.enabled=true`
- image repository/tag 分离正确
- StorageV3/Vortex 关闭时没有 Vortex format
- topology summary 标明 `deployer=helm`

### 任务 3: 把 cluster workflow 切到 Helm deployer

**文件：**
- 修改: `milvus-bricks/argo/cluster-upgrade-rollback.yaml`
- 测试: `milvus-bricks/milvus_client/tests/test_argo_template.py`

**步骤 1: deploy-base 使用 `helm repo add` + `helm upgrade --install`**

命令形态:
```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update milvus
helm upgrade --install "$release" milvus/milvus -n "$ns" -f /tmp/milvus-values.yaml --wait --timeout 15m
```

**步骤 2: wait-ready 改为 Helm/service/pod 组合验证**

对于 cluster Helm:
- `helm status <release>`
- service `${release}-milvus:19530` 存在
- selector `app.kubernetes.io/instance=<release>` 下关键 Milvus pods Ready
- deployment/statefulset/container image 包含 expected image tag

**步骤 3: patch image/config 改为复用 renderer + Helm upgrade**

升级/回滚阶段渲染新 values，然后:
```bash
helm upgrade "$release" milvus/milvus -n "$ns" -f /tmp/milvus-values.yaml --wait --timeout 15m
```

**步骤 4: snapshot-config 改为保存 Helm values 与 workload image**

保存:
- `helm get values <release> -a -o yaml`
- `kubectl get deploy,sts -l app.kubernetes.io/instance=<release> -o json`
- summary JSON 里保留 phase/image/version/config。

**步骤 5: cleanup 删除 Helm release 和残留资源**

onExit:
```bash
helm uninstall "$release" -n "$ns" --wait --timeout 5m
kubectl delete ... -l app.kubernetes.io/instance="$release"
```

### 任务 4: 更新文档和参数说明

**文件：**
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback-gates/README.md`
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`

说明:
- standalone 使用 Operator CR
- cluster Woodpecker 使用 Helm chart
- `2.6.18 rollback` 是 diagnostic，不是 formal gate
- StorageV3/Vortex 在 `2.6 -> 3.0 -> 2.6` gate 中必须保持关闭

### 任务 5: 验证并提交 PR

**步骤 1: 本地验证**

运行:
```bash
PYTHONPATH=. python3 -m pytest milvus_client/tests -q
argo template lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/cluster-upgrade-rollback.yaml -n qa -o simple
git diff --check
```

预期:
- pytest 全通过
- Argo lint 通过
- diff check 无输出

**步骤 2: 4am 验证**

更新 live templates，提交:
- standalone smoke: `2.6.18 -> 3.0 latest -> 2.6 latest`
- cluster smoke: Helm Woodpecker `2.6.18 -> 3.0 latest -> 2.6 latest`

**步骤 3: 创建 PR**

使用 DCO 签名 commit:
```bash
git add -A
git commit -s -m "test: update Milvus upgrade rollback gates"
git push -u origin feature/upgrade-rollback-gates-cluster
gh pr create --fill
```
