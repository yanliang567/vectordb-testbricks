# PR11 配置生效硬校验实现计划

**目标：** 处理 PR #11 review：在 target 和 rollback 数据校验前硬校验 LoonFFI/storage v3 与 Vortex 配置已按预期落地，并修正 README 场景文档链接。

**架构：** 在三个 Upgrade/Rollback WorkflowTemplate 中新增 `assert-milvus-storage-config` template。standalone 从 Milvus CR `spec.config` 读取配置；cluster 从 Helm release values 的 `extraConfigFiles.user.yaml` 读取配置。DAG 在 `snapshot-after-upgrade-config` 和 `snapshot-after-rollback-config` 后插入断言 step，后续 precheck/DML/DQL/index validation 依赖该断言。测试通过解析 Argo YAML 确认 step、依赖和配置键存在。

**技术栈：** Argo WorkflowTemplate YAML, kubectl, helm, Python stdlib + PyYAML, pytest。

---

### 任务 1: 新增 WorkflowTemplate 配置断言

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤 1: 写模板结构断言测试**

在 `test_argo_template.py` 增加断言：

- 三个模板都包含 `assert-milvus-storage-config`。
- after-upgrade assertion 依赖 `snapshot-after-upgrade-config`。
- `precheck-after-upgrade` 依赖 `assert-after-upgrade-storage-config`。
- after-rollback assertion 依赖 `snapshot-after-rollback-config`。
- `precheck-after-rollback` 依赖 `assert-after-rollback-storage-config`。
- 命令内容包含 `common.storage.useLoonFFI` 和 `dataNode.storage.format`。

**步骤 2: 实现 standalone 断言模板**

模板输入：

```yaml
- phase
- expected-loon-ffi-enabled
- expected-vortex-enabled
- deployment-mode
```

standalone 读取：

```bash
kubectl -n <milvus-namespace> get mi <workflow.name> -o json
```

断言：

```text
spec.config.common.storage.useLoonFFI == expected
spec.config.dataNode.storage.format == "vortex" when expected vortex is true
spec.config.dataNode.storage.format is absent/null/empty when expected vortex is false
```

**步骤 3: 实现 cluster 断言模板**

cluster 读取：

```bash
helm get values <workflow.name> -n <milvus-namespace> -a -o yaml
```

解析 `extraConfigFiles.user.yaml` 后执行同样断言。

### 任务 2: 插入 DAG 依赖

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`

**步骤 1: after-upgrade 插入**

在 `snapshot-after-upgrade-config` 后新增：

```yaml
- name: assert-after-upgrade-storage-config
  dependencies: [snapshot-after-upgrade-config, pressure-daemon]
```

把 `observe-after-upgrade` 和 `precheck-after-upgrade` 中需要确保配置生效的依赖改到 assertion 后。

**步骤 2: after-rollback 插入**

在 `snapshot-after-rollback-config` 后新增：

```yaml
- name: assert-after-rollback-storage-config
  dependencies: [snapshot-after-rollback-config, pressure-daemon]
```

把 `precheck-after-rollback` 改为依赖 assertion。

### 任务 3: 修正文档链接

**文件：**
- 修改: `milvus_client/README.md`

**步骤 1: 修正相对路径**

把 `docs/upgrade-rollback-gates/README.md` 改为 `../docs/upgrade-rollback-gates/README.md`，避免解析到 `milvus_client/docs/...`。

### 任务 4: 验证和更新 PR

**步骤 1: 运行测试**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests/test_argo_template.py milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_upgrade_rollback_params.py
```

**步骤 2: 运行全量校验**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests
uvx ruff check milvus_client/tests/test_argo_template.py
uvx ruff format --check milvus_client/tests/test_argo_template.py
argo lint argo/standalone-2-6-upgrade-rollback.yaml
argo lint argo/standalone-3-0-upgrade-rollback.yaml
argo lint argo/cluster-upgrade-rollback.yaml
git diff --check
```

**步骤 3: amend commit 并 force-push PR branch**

```bash
git add argo/*.yaml milvus_client/tests/test_argo_template.py milvus_client/README.md docs/plans/2026-07-24-pr11-config-assertions.md
git commit --amend --no-edit
git push --force-with-lease
```
