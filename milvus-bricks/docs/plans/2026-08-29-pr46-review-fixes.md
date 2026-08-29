# PR #46 Review 修复实现计划

**目标：** 修复 PR #46 review 发现的 standalone Pod recycle 权限与超时边界、新集合加载可见性假失败，以及验证报告中的审计信息偏差。

**架构：** standalone WorkflowTemplate 使用异步 Pod 删除和统一 600 秒 wall-clock deadline，通过非阻塞 Pod Ready 状态查询判断新 UID 收敛；namespace Role 仅补充 Pod 删除权限。phase DML/DQL 将新集合的首次 count/PK 可见性检查纳入既有有界重试，保留后续 search、scalar index 和严格 reload 断言。

**技术栈：** Argo Workflows、Kubernetes RBAC/kubectl、POSIX shell、Python/PyMilvus、pytest、Ruff、PyYAML。

---

### 任务 1：锁定 RBAC 和 rollout deadline 回归

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback-rbac.yaml`
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 测试: `milvus_client/tests/test_argo_template.py`

**步骤：**
1. 修改测试，要求 standalone Role 对 core `pods` 具有最小 `delete` 权限，且不授予 `pods/log`、events 删除权限。
2. 修改测试，要求两个 standalone 模板的 image/config recycle 都使用 `--wait=false`、600 秒 wall-clock deadline、5 秒 Kubernetes request timeout和非阻塞 Ready 查询，并禁止嵌套 `kubectl wait`。
3. 运行定向测试，确认当前代码失败。
4. 为 standalone Role 增加独立 `pods/delete` 规则。
5. 在四个 recycle 路径实现统一 deadline/Ready 查询。
6. 重跑定向测试和 `argo lint --offline argo`。

### 任务 2：修复新集合 load 超时后的立即假失败

**文件：**
- 修改: `milvus_client/requests/validate_phase_dml_dql.py`
- 测试: `milvus_client/tests/test_validate_phase_dml_dql.py`

**步骤：**
1. 增加 fake client：首次 best-effort load 超时，随后 count/PK 可见性从失败收敛为成功。
2. 运行定向测试，确认当前新集合一次性验证失败。
3. 为 `_run_new_collection_dml_dql` 增加 visibility timeout/interval 参数，把 count/PK 检查包装进 `_wait_for_validation`，并把最终 report/metrics 合并回主报告。
4. 从 CLI 参数向新集合路径传递现有 visibility 配置。
5. 重跑 phase DML/DQL 全文件测试。

### 任务 3：校准计划和最终报告

**文件：**
- 修改: `docs/plans/2026-08-29-3-0-1-upgrade-rollback-validation.md`
- 修改: `milvus_client/docs/reports/2026-08-29-3-0-1-upgrade-rollback-validation.md`

**步骤：**
1. 将 standalone JSON Shredding tracker 的 issue 归属从 #52341 更正为 #52768。
2. 将 `keep-milvus=true` 声明改为包含 `v4rzd` 默认 false 的实际例外。
3. 区分最终业务 E2E revision `8400590` 与 post-E2E review 修复 revision，列出后续修复及其验证范围。
4. 明确 post-E2E 修复未重跑完整 20 条矩阵，避免把静态验证表述为业务 E2E 证据。

### 任务 4：完成前验证并更新 PR

**步骤：**
1. 运行 `PYTHONPATH=. python3 -m pytest milvus_client/tests -q`。
2. 运行 CI 范围 Ruff check 和 format check。
3. 运行 `argo lint --offline argo` 和 `git diff --check`。
4. 提交功能修复，记录 commit SHA；再用该 SHA 和最新测试数量更新报告并提交文档修复。
5. 推送 PR #46，等待 GitHub CI 通过并记录最终 head。
