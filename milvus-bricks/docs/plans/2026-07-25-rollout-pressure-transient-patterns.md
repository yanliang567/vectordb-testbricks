# Rollout Pressure Transient Patterns 实现计划

**目标：** 将 upgrade/rollback rollout 维护窗口内明确的 Milvus 服务切换错误归类为 transient pressure failure。

**架构：** 在 `pressure_maintenance.py` 增加 rollout-only classifier，匹配有限的 Milvus 服务切换错误文本。继续保留 metrics-only、部分缺失 failure detail、schema-evolution 非目标错误和 correctness failure 的 strict 行为。

**技术栈：** Python, pytest, Argo Workflow pressure summary classifier。

---

### 任务 1: 增加 rollout-only failure 分类

**文件：**
- 修改: `milvus_client/common/pressure_maintenance.py`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤 1: 写失败测试**

添加用例覆盖：
- `channel distribution is not serviceable` + `channel not available` 在 `rollback-rollout` 中被 excluded。
- `find no available mixcoord` 在 `upgrade-rollout` 中被 excluded。
- 同样错误在 `schema-evolution-existing` 中保持 failed。
- 普通 correctness failure 继续保持 failed。

**步骤 2: 运行测试验证失败**

运行:

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests/test_argo_template.py
```

预期：新增 rollout transient 用例失败。

**步骤 3: 实现最小分类**

在 `pressure_maintenance.py` 中添加：
- rollout window label 判定。
- 有限 error pattern 判定。
- 在 `classify_pressure_result()` 中将匹配 failure 加入 `excluded_failures`。

**步骤 4: 验证**

运行:

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py
uv run pytest -q
uvx ruff check milvus_client/common/pressure_maintenance.py milvus_client/tests/test_argo_template.py
uvx ruff format --check milvus_client/common/pressure_maintenance.py milvus_client/tests/test_argo_template.py
git diff --check
```

**步骤 5: 提交 PR**

```bash
git add milvus_client/common/pressure_maintenance.py milvus_client/tests/test_argo_template.py docs/plans/2026-07-25-rollout-pressure-transient-patterns.md
git commit -m "fix: classify rollout service switching pressure failures"
git push -u origin feat/rollout-pressure-transient-patterns
```
