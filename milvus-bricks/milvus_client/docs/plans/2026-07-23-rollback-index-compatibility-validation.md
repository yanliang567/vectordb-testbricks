# Rollback Index Compatibility Validation 实现计划

**目标：** 在升级/回滚 gate 中显式覆盖“3.0 阶段重建 sealed index 后回滚到 2.6 是否仍可 load/search/query”的兼容性边界。

**架构：** 新增一个独立 request brick `validate_index_compatibility`。升级后对 baseline checkpoint 集合执行 flush/load、记录实际 index metadata、search/query，并写出 index compatibility checkpoint；回滚后读取同一 checkpoint，不再重建 index，重新枚举并比较实际 index metadata，再执行 load/search/query 验证。WorkflowTemplate 通过 `index-compatibility-validation-enabled` 参数控制，默认在 standalone-2.6 和 cluster rollback template 中开启，并在最终报告中作为 required validation。`--rebuild-index=true` 保留为手工诊断能力，不作为 promoted gate 默认路径。

**技术栈：** Python/pymilvus request brick、Argo WorkflowTemplate DAG、pytest、Ruff、Argo lint。

---

### 任务 1: 新增 index compatibility request

**文件：**
- 创建: `milvus_client/requests/validate_index_compatibility.py`
- 测试: `milvus_client/tests/test_validate_index_compatibility.py`

**步骤 1: 写失败测试**

覆盖：
- after-upgrade 模式会对 checkpoint collection 执行 `flush -> load_collection -> list_indexes/describe_index -> search/query`，并写出实际 index metadata checkpoint。
- after-rollback 模式读取 index checkpoint，不 drop/create index，重新 `list_indexes/describe_index` 并比较同一批 index metadata，再 load/search/query。
- 搜索失败会写结构化 `INDEX_SEARCH_FAILED`。

**步骤 2: 实现 request**

参数：
- `--schema-matrix`
- `--checkpoint-file`
- `--index-checkpoint-file`
- `--phase`
- `--rebuild-index`
- `--timeout-sec`

输出：
- 标准 `BrickResult`
- metrics: checked collections/indexes/searches/rebuilt counts
- checkpoint: index compatibility checkpoint path

**步骤 3: 运行 request 单测**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_validate_index_compatibility.py -q`

预期: 通过。

### 任务 2: 接入 standalone-2.6 WorkflowTemplate

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤 1: 写 DAG 断言**

断言：
- 参数包含 `index-compatibility-validation-enabled`，默认 true。
- `validate-index-compatibility-after-upgrade` 在 `validate-after-upgrade` 后、`strict-pressure-after-upgrade` 前。
- `validate-index-compatibility-after-rollback` 在 `wait-rollback-serviceability` 后、`validate-after-rollback` 前。

**步骤 2: 修改 DAG**

用 `optional-run-brick` 调用：
- after-upgrade: `validate_index_compatibility --rebuild-index false --index-checkpoint-file /tmp/milvus-bricks/checkpoints/index_compatibility.json`
- after-rollback: `validate_index_compatibility --rebuild-index false --index-checkpoint-file /tmp/milvus-bricks/checkpoints/index_compatibility.json`

**步骤 3: 运行模板测试**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py -q`

预期: 通过。

### 任务 3: 接入 cluster WorkflowTemplate

**文件：**
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤 1: 写 DAG 断言**

同 standalone-2.6，确认 cluster template 也插入同名任务。

**步骤 2: 修改 DAG**

保持 cluster service URI 路径由现有 `optional-run-brick` 处理。

**步骤 3: 运行模板测试**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py -q`

预期: 通过。

### 任务 4: 报告和 scenario renderer 参数

**文件：**
- 修改: `milvus_client/requests/generate_workflow_report.py`
- 修改: `milvus_client/common/gates.py`
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试: `milvus_client/tests/test_generate_workflow_report.py`
- 测试: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 测试: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

**步骤 1: 写报告 required validation 测试**

当 `index_compatibility_validation_enabled=true` 且 rollback enabled 时，报告必须要求：
- `validate_index_compatibility_after_upgrade`
- `validate_index_compatibility_after_rollback`

**步骤 2: 实现参数透传**

新增 manifest bool 字段 `index_compatibility_validation_enabled`，renderer 输出 `index-compatibility-validation-enabled`。

**步骤 3: 运行相关测试**

运行:
`PYTHONPATH=. pytest milvus_client/tests/test_generate_workflow_report.py milvus_client/tests/test_render_upgrade_rollback_params.py milvus_client/tests/test_upgrade_rollback_gates_manifest.py -q`

预期: 通过。

### 任务 5: 全量验证、提交和 PR

**文件：**
- 本 PR 所有变更。

**步骤 1: 全量测试**

运行:
`PYTHONPATH=. pytest milvus_client/tests -q`

预期: 通过。

**步骤 2: lint**

运行:
- `uvx ruff check milvus_client`
- `uvx ruff format --check milvus_client`
- `argo lint argo/standalone-2-6-upgrade-rollback.yaml`
- `argo lint argo/cluster-upgrade-rollback.yaml`
- `git diff --check`

预期: 通过。

**步骤 3: 提交并创建 PR**

运行:
- `git add ...`
- `git commit -m "test: add rollback index compatibility validation"`
- `git push -u origin fix/rollback-index-validation`
- `gh pr create ...`

预期: 新 PR 创建成功。
