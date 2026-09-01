# PR #46 可见性 deadline 与发布证据 Review 修复计划

**目标：** 让 phase DML/DQL 的 120 秒可见性 deadline 真正约束每一次 PyMilvus query RPC，并用当前 PR head 的 same-image standalone rollout 补齐发布证据。

**架构：** `_wait_for_validation` 为每个 attempt 提供一个“剩余 RPC 预算”函数，validator 在每个 count/PK/upsert/deleted-PK query 前重新取值并传给 PyMilvus `timeout`。这样不会把总上界放大为“RPC 数量 × 120 秒”。报告保留原始业务矩阵的 revision 边界，另行记录之后的 `25582ec`、`cd3b126`、`ca321c3` 和本次修复，以及当前 head 的代表性 live E2E。

**技术栈：** Python 3、PyMilvus、pytest、Ruff、Argo Workflows、4am Kubernetes。

---

### 任务 1：固化无界 RPC 回归测试

**文件：**
- 修改：`milvus_client/tests/test_validate_phase_dml_dql.py`
- 修改：`milvus_client/tests/test_validators.py`

**步骤 1：** 增加 existing collection 用例，检查 count、PK sample、upsert sample 和 deleted-PK query 都收到正的 `timeout`。

**步骤 2：** 增加 new collection 用例，检查 count 和 PK sample query 都收到正的 `timeout`。

**步骤 3：** 增加 validator helper 用例，检查 `query_count` / `validate_pk_samples` 将 timeout 透传到 `client.query`。

**步骤 4：** 运行定向 pytest，预期新用例在实现前失败，证明可复现 review 指出的缺口。

### 任务 2：实现剩余 RPC 预算

**文件：**
- 修改：`milvus_client/common/validators.py`
- 修改：`milvus_client/requests/validate_phase_dml_dql.py`

**步骤 1：** 为 `query_count`、`validate_collection_count` 和 `validate_pk_samples` 增加可选 timeout 参数，保持旧调用者兼容。

**步骤 2：** 让 `_wait_for_validation` 创建 wall-clock deadline，将可调用的剩余预算传给 callback，并将 sleep 限制在剩余时间内。

**步骤 3：** 为 `_query_rows_by_pk_values`、`_validate_pk_values_present_strict`、`_validate_deleted_pk_values` 和 `_validate_upserted_values` 增加剩余预算传递。

**步骤 4：** 在 existing/new 两个 visibility callback 的每个 RPC 前重新获取 timeout。

**步骤 5：** 运行定向 pytest，预期新老用例全部通过。

### 任务 3：静态验证并更新发布报告

**文件：**
- 修改：`milvus_client/docs/reports/2026-08-29-3-0-1-upgrade-rollback-validation.md`

**步骤 1：** 运行完整 `milvus_client/tests`、CI 范围 Ruff check/format、`argo lint --offline argo` 和 shell syntax 检查。

**步骤 2：** 更新最终 post-E2E revision 链，准确记录 `25582ec`、`cd3b126`、`ca321c3` 和本次 visibility RPC deadline 修复的实现/验证边界。

**步骤 3：** 更新 pytest 数量和命令，不宣称未执行的业务 E2E。

### 任务 4：当前 head 的 same-image standalone live E2E

**文件：**
- 修改：`milvus_client/docs/reports/2026-08-29-3-0-1-upgrade-rollback-validation.md`

**步骤 1：** 确认 4am API/Argo 可用、目标镜像不变，并从当前 head 创建隔离 WorkflowTemplate，不覆盖共享模板。

**步骤 2：** 提交一条 standalone 3.0 same-image config rollout，使用不可变 candidate image、`milvus-log-level=debug`、`keep-milvus=false` 和当前 head revision。

**步骤 3：** 监控到终态，检查 image/config Pod recycle、phase existing/new visibility、rollback、final gate 和 `onExit` 清理。

**步骤 4：** 下载 `orchestrator_report.json`、`flow_summary.json` 和必要日志，将 workflow ID、revision、节点结果和清理结果写入报告。

**步骤 5：** 删除隔离 WorkflowTemplate；根据 `keep-milvus=false` 和精确标签复核无残留测试资源。

### 任务 5：完成 PR 更新

**文件：**
- 提交本计划所列代码、测试、报告和证据。

**步骤 1：** 重新运行完整验证和 `git diff --check`，确认无用户无关文件进入提交。

**步骤 2：** 提交并推送当前 PR 分支，更新 PR #46 描述与 review 回复。

**步骤 3：** 检查 GitHub CI 和 PR merge state；只在新 head 验证通过后报告可重新 review。
