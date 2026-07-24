# Pressure Maintenance Window Summary 实现计划

**目标：** 将 schema evolution 维护窗口内的 `SchemaMismatchRetryableException` 作为受限已知 transient 排除，并在 pressure summary/report 中记录 rollout/maintenance window 耗时。

**架构：** 复用 `milvus_client.common.pressure_maintenance.classify_pressure_result()` 作为唯一分类入口，新增受限 classifier helper：只有 failure 时间与 `schema-evolution-*` window 重叠、且错误类型/文本明确为 schema mismatch retryable 时才排除。三份 Argo WorkflowTemplate 的 `check-pressure-results` 继续调用共享 helper，仅增强 window summary 输出 `duration_sec`。

**技术栈：** Python, pytest, Argo WorkflowTemplate YAML。

---

### 任务 1: 补 pressure classifier 失败/通过测试

**文件：**
- 修改: `milvus_client/tests/test_argo_template.py`

**步骤 1: 添加 schema evolution 内可排除测试**

新增测试构造：

- `maintenance_windows[0].label = "schema-evolution-existing"`
- failure:
  - `error_type = "SchemaMismatchRetryableException"`
  - `operation = "upsert"`
  - failure interval 在 window 内
  - `connectivity_transient = false`
- 预期 `classify_pressure_result()` 返回 `excluded`
- 预期 entry 包含 `maintenance_window.label == "schema-evolution-existing"`

**步骤 2: 添加负向测试**

新增测试覆盖：

- 同样错误但 window label 是 `rollback-rollout`，必须 `failed`
- schema evolution window 内但 `error_type = "AssertionError"`，必须 `failed`

**步骤 3: 添加 duration summary 模板测试**

在现有模板测试中断言三份 WorkflowTemplate 的 `check-pressure-results` 脚本输出 maintenance window 时包含：

- `duration_sec`

### 任务 2: 实现受限 schema mismatch 排除

**文件：**
- 修改: `milvus_client/common/pressure_maintenance.py`

**步骤 1: 新增 helper**

添加 `is_schema_evolution_schema_mismatch(failure, window)`：

- window label 必须以 `schema-evolution-` 开头
- `error_type` 必须等于 `SchemaMismatchRetryableException`，或 error text 包含该类型名
- error text 必须包含 `schema mismatch`
- 不依赖 `connectivity_transient`

**步骤 2: 接入 classify**

在 `classify_pressure_result()` 的 per-failure 循环里：

- 如果 window 重叠且 `is_connectivity_failure(failure)`，保持现有排除逻辑；
- 如果 window 重叠且 `is_schema_evolution_schema_mismatch(failure, window)`，也排除；
- 其他 correctness failure 继续 strict failed。

### 任务 3: 记录 rollout/maintenance window duration

**文件：**
- 修改:
  - `argo/standalone-2-6-upgrade-rollback.yaml`
  - `argo/standalone-3-0-upgrade-rollback.yaml`
  - `argo/cluster-upgrade-rollback.yaml`
  - `milvus_client/requests/generate_workflow_report.py`

**步骤 1: 模板 summary 增加 duration**

在 `check-pressure-results` 的 `_maintenance_windows()` 中，生成 window 时增加：

```python
"duration_sec": max(0.0, (end - start).total_seconds())
```

在 summary 的 `maintenance_windows` 输出里保留该字段。

**步骤 2: final markdown 展示 duration**

在 `build_markdown()` 的 pressure section 增加 maintenance window 列表：

```text
- maintenance window `<label>`: duration_sec=`...`
```

### 任务 4: 验证和提交 PR

**步骤 1: 跑相关测试**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py
```

**步骤 2: 跑全量测试**

```bash
uv run pytest -q
```

**步骤 3: 跑 lint**

```bash
uvx ruff check milvus_client/common/pressure_maintenance.py milvus_client/requests/generate_workflow_report.py milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py
uvx ruff format --check milvus_client/common/pressure_maintenance.py milvus_client/requests/generate_workflow_report.py milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py
argo lint argo/standalone-2-6-upgrade-rollback.yaml
argo lint argo/standalone-3-0-upgrade-rollback.yaml
argo lint argo/cluster-upgrade-rollback.yaml
git diff --check
```

**步骤 4: 创建 PR**

```bash
git add ...
git commit -m "fix: classify schema evolution pressure windows"
git push -u origin feat/pressure-maintenance-window-summary
gh pr create --title "fix: classify schema evolution pressure windows" --body-file /tmp/pr12-body.md
```
