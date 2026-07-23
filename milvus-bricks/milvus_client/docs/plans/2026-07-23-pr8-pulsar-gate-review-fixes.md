# PR8 Pulsar Gate Review Fixes 实现计划

**目标：** 修复 PR #8 review 指出的 Pulsar cluster gate 阻断问题，保证 2.6 rollback cluster 场景可以通过 Helm/RBAC 部署并产出可诊断报告。

**架构：** 保留 2.6 cluster rollback gate 使用 Pulsar profile，保留 3.0 cluster gate 使用 Woodpecker profile。Pulsar profile 只作为显式场景参数使用；WorkflowTemplate 默认值保持已注册的 3.0 + Woodpecker gate。RBAC 最小扩展到 Helm Chart 5.0.24 需要创建的 namespace 资源，并用测试锁住 topology/report/snapshot 行为。

**技术栈：** Argo WorkflowTemplate YAML、Kubernetes RBAC、Milvus Helm Chart 5.0.24、pytest、Ruff、Argo lint。

---

### 任务 1: 修复 Pulsar 1CU profile 的有效运行配置

**文件：**
- 修改: `milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml`
- 测试: `milvus_client/tests/test_deploy_profiles.py`
- 测试: `milvus_client/tests/test_render_milvus_cr.py`

**步骤 1: 写失败测试**

在 Pulsar profile 测试中断言：
- BookKeeper `managedLedgerDefaultEnsembleSize/writeQuorum/ackQuorum` 都是 `"1"`。
- Broker `PULSAR_MEM` 不再使用 Chart 默认 4g/8g，而是与 4Gi limit 匹配。
- Proxy `PULSAR_MEM` 与 2Gi limit 匹配。

**步骤 2: 实现 profile override**

在 `pulsarv3.broker.configData` 中设置：
- `managedLedgerDefaultEnsembleSize: "1"`
- `managedLedgerDefaultWriteQuorum: "1"`
- `managedLedgerDefaultAckQuorum: "1"`
- `PULSAR_MEM: -Xms512m -Xmx512m -XX:MaxDirectMemorySize=1024m`

在 `pulsarv3.proxy.configData` 中设置：
- `PULSAR_MEM: -Xms256m -Xmx512m -XX:MaxDirectMemorySize=512m`

**步骤 3: 运行 profile/render 单测**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_deploy_profiles.py milvus_client/tests/test_render_milvus_cr.py -q`

预期: 通过。

### 任务 2: 修复 Workflow RBAC

**文件：**
- 修改: `argo/cluster-upgrade-rollback-rbac.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤 1: 写 RBAC 断言**

断言 qa-milvus manager Role 包含：
- `batch/jobs` create/patch/update/delete。
- `rbac.authorization.k8s.io` 的 `roles`、`rolebindings` create/patch/update/delete。
- core `pods` 包含 create/delete，以允许 Helm 创建 Pulsar subchart Role 后不触发 privilege escalation。

**步骤 2: 实现 RBAC**

在 manager Role 中补齐 Helm install/upgrade/uninstall 所需 namespace-scoped 权限，避免 cluster-admin。

**步骤 3: 运行 RBAC 单测**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py -q`

预期: 通过。

### 任务 3: 修复 cluster WorkflowTemplate 默认参数元组

**文件：**
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤 1: 写默认参数断言**

断言 cluster WorkflowTemplate 默认 profile 仍是 `cluster-woodpecker-1cu.yaml`，并与 3.0 branch 默认镜像、schema、collection prefix、rollback-forward validation 形成已注册 3.0 gate 元组。

**步骤 2: 实现默认值回退**

把 `deploy-profile` 默认值从 Pulsar profile 改回 Woodpecker profile。

**步骤 3: 运行模板单测**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py -q`

预期: 通过。

### 任务 4: 补齐 Pulsar topology 与 K8s snapshot

**文件：**
- 修改: `milvus_client/common/deploy.py`
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 测试: `milvus_client/tests/test_render_milvus_cr.py`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤 1: 写 topology 断言**

对 Pulsar profile 的 Helm summary 断言 `dependencies.pulsarv3` 存在，并包含 `enabled=true`、broker/bookkeeper/zookeeper replica 信息。

**步骤 2: 写 snapshot 断言**

断言 collect artifacts 与 cleanup snapshot 同时采集：
- `app.kubernetes.io/instance={{workflow.name}}`
- `release={{workflow.name}}`

**步骤 3: 实现 summary/snapshot**

在 `helm_deploy_topology_summary()` 中加入 `pulsarv3`。在 `collect-artifacts` 和 cleanup snapshot 中新增 `release` label 资源输出文件。

**步骤 4: 运行相关单测**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_render_milvus_cr.py milvus_client/tests/test_argo_template.py -q`

预期: 通过。

### 任务 5: 全量验证并提交

**文件：**
- 修改本 PR 涉及文件。

**步骤 1: 运行全量 Python 测试**

运行:
`PYTHONPATH=. pytest milvus_client/tests -q`

预期: 全部通过，允许没有 Helm 的环境跳过真实 Helm 测试。

**步骤 2: 运行 lint/format/check**

运行:
- `uvx ruff check milvus_client/tests/test_deploy_profiles.py milvus_client/tests/test_render_milvus_cr.py milvus_client/tests/test_argo_template.py`
- `uvx ruff format --check milvus_client/tests/test_deploy_profiles.py milvus_client/tests/test_render_milvus_cr.py milvus_client/tests/test_argo_template.py`
- `argo lint argo/cluster-upgrade-rollback.yaml`
- `git diff --check`

预期: 全部 exit 0。

**步骤 3: 提交并推送 PR branch**

运行:
- `git status --short`
- `git add ...`
- `git commit -m "test: fix pulsar cluster rollback gate deployment"`
- `git push`

预期: PR #8 更新成功。
