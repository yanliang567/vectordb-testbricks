# Index Gate Target-Only 与 Debug 日志实现计划

**目标：** 避免 v3.0.0 的 #52767 阻断 index gate，并让所有升级回滚场景默认以 debug 日志运行。

**架构：** 将 v10/v11 index matrix 从 base matrix 移到 target-only forward workload，rollback 前删除；通过统一的 `milvus-log-level` workflow 参数把默认 debug 配置传给 Operator CR 与 Helm values renderer，并进入报告快照。

**技术栈：** Python 3、PyYAML、pytest、Argo WorkflowTemplate、Milvus Operator CR、Milvus Helm chart。

---

### 任务 1: 锁定 index gate 的 target-only 合同

**文件：**
- 修改: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`

**步骤 1: 写失败测试**

对 standalone/cluster 的 v10/v11 四个场景断言：

```python
assert scenario["schema_matrix"].endswith("schema_matrix_2_6.yaml")
assert scenario["forward_schema_matrix"].endswith(expected_index_matrix)
assert scenario["forward_workload_enabled"] is True
assert scenario["drop_forward_before_rollback_enabled"] is True
assert scenario["rollback_forward_validation_enabled"] is False
assert scenario["base"].get("target_vec_index_version", -1) == -1
assert scenario["target"]["target_vec_index_version"] == expected_version
assert scenario["rollback"].get("target_vec_index_version", -1) == -1
```

**步骤 2: 运行测试验证失败**

运行：

```bash
pytest -q milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_upgrade_rollback_params.py
```

预期：旧场景仍在 base 使用 index matrix，断言失败。

**步骤 3: 写最小实现**

修改四个场景为 `rollback_safe_base` + 对应 `forward_schema_matrix_ref`，设置 forward/drop/rollback validation 标志，并仅在 target phase 保留 index version。

**步骤 4: 运行测试验证通过**

重复步骤 2，预期全部通过。

### 任务 2: 给 deploy renderer 增加日志级别

**文件：**
- 修改: `milvus_client/common/deploy.py`
- 修改: `milvus_client/requests/render_milvus_cr.py`
- 修改: `milvus_client/requests/render_milvus_helm_values.py`
- 修改: `milvus_client/tests/test_render_milvus_cr.py`
- 修改: `milvus_client/tests/test_render_milvus_helm_values.py`

**步骤 1: 写失败测试**

```python
assert rendered["spec"]["config"]["log"]["level"] == "debug"
assert yaml.safe_load(values["extraConfigFiles"]["user.yaml"])["log"]["level"] == "debug"
```

并增加 `info` 显式覆盖用例。

**步骤 2: 运行测试验证失败**

运行：

```bash
pytest -q milvus_client/tests/test_render_milvus_cr.py milvus_client/tests/test_render_milvus_helm_values.py
```

预期：渲染结果中不存在 `log.level`。

**步骤 3: 写最小实现**

为公共 renderer 和两个 CLI 增加默认 `log_level="debug"` / `--log-level`，合并到 Milvus runtime config，并在 topology summary 中保留该配置。

**步骤 4: 运行测试验证通过**

重复步骤 2，预期全部通过。

### 任务 3: 在全部升级回滚 WorkflowTemplate 传递参数

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 修改: `milvus_client/common/gates.py`
- 修改: `milvus_client/tests/test_argo_template.py`
- 修改: `milvus_client/tests/test_render_upgrade_rollback_params.py`

**步骤 1: 写失败测试**

验证每个模板声明 `milvus-log-level: debug`，部署 renderer 调用传入 `--log-level`，cluster Helm upgrade 继续传入该值，环境快照包含 `milvus_log_level`；场景 renderer 默认生成 `milvus-log-level=debug`。

**步骤 2: 运行测试验证失败**

运行：

```bash
pytest -q milvus_client/tests/test_argo_template.py milvus_client/tests/test_render_upgrade_rollback_params.py
```

预期：模板和参数渲染缺少新字段。

**步骤 3: 写最小实现**

在 manifest defaults、gate 参数 renderer、三份模板的 arguments/deploy/patch/report 中统一加入 `milvus-log-level`。

**步骤 4: 运行测试验证通过**

重复步骤 2，预期全部通过。

### 任务 4: 完整验证与检查

**文件：**
- 修改: `milvus_client/docs/upgrade-rollback.md`
- 修改: `docs/upgrade-rollback-gates/README.md`

**步骤 1: 更新文档**

说明 index gate 为 target-only、回滚前删除，并记录默认 `milvus-log-level=debug` 与覆盖方法。

**步骤 2: 运行相关完整测试**

```bash
pytest -q milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_render_milvus_cr.py \
  milvus_client/tests/test_render_milvus_helm_values.py \
  milvus_client/tests/test_argo_template.py
```

预期：0 failed。

**步骤 3: 校验 YAML 与 Argo**

```bash
python3 -c 'import pathlib,yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path("argo").glob("*upgrade-rollback*.yaml")]'
argo lint argo/standalone-2-6-upgrade-rollback.yaml
argo lint argo/standalone-3-0-upgrade-rollback.yaml
argo lint argo/cluster-upgrade-rollback.yaml
```

预期：YAML 解析成功；Argo lint 在当前 CLI 支持 WorkflowTemplate lint 时通过。

**步骤 4: 检查差异**

```bash
git diff --check
git status --short
git diff --stat
```

预期：无 whitespace error，仅包含本任务文件。
