# Milvus Config Matrix Upgrade/Rollback 实现计划

**目标：** 扩展 MilvusClient upgrade/rollback bricks 和 4am Argo Workflow，覆盖 jsonShredding、LoonFFI、2.6->2.6、2.6->3.0、3.0->3.0 的配置矩阵升级回滚场景，并输出可复现实测报告。

**架构：** 保留 `milvus-bricks/2.6/` 不变，继续在 `milvus-bricks/milvus_client/` 内扩展 request bricks、schema manifests、workflow templates 和报告生成。Argo Workflow 负责部署 Milvus standalone、配置 patch、升级/回滚、压力 daemon、数据完整性 gate 和 artifact 收集；Python bricks 负责 schema/data/request/result 的可组合执行协议。

**技术栈：** Python 3, pymilvus, pytest, PyYAML, Argo WorkflowTemplate, kubectl, Milvus Operator CR `milvus.io/v1beta1`, 4am namespace `qa` / `qa-milvus`。

---

## 场景判断

1. `v2.6.18 + jsonShredding -> 2.6-latest + jsonShredding -> rollback v2.6.18`：可做。全程只跑 2.6 compatible schema/workload，作为第一优先级。
2. `v2.6.18 + jsonShredding -> master-latest + jsonShredding -> rollback v2.6.18`：可做。仍然只跑 2.6 compatible schema/workload，验证 master 不破坏旧数据和回滚兼容性。
3. `v2.6.18 jsonShredding disabled -> master-latest LoonFFI enabled -> update jsonShredding enabled`：可做，但回滚目标是 2.6 时，3.0 新 schema/data 不能作为回滚硬门禁。phase 1 跑 2.6 workload；phase 2/3 可以增加 forward-only 3.0 workload，回滚前后只把 2.6 数据完整性设为 hard gate。
4. `3.0-20260701-d19d8484-47f6c14 -> master-latest -> rollback baseline 3.0`：可做。全程跑 3.0 compatible schema/workload，作为 3.0 baseline 回滚门禁。
5. `3.0 baseline + jsonShredding -> master-latest + jsonShredding + LoonFFI -> rollback baseline 3.0`：可做但有风险。若 LoonFFI 写入数据格式没有明确 rollback guarantee，默认将 baseline 数据兼容性作为 hard gate，LoonFFI 新数据作为独立 warning/risk artifact；确认保证后再提升为 hard gate。

## Phase 1: 配置矩阵建模和 2.6 workflow 参数化

**文件：**
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `milvus-bricks/milvus_client/requests/generate_workflow_report.py`
- 修改: `milvus-bricks/milvus_client/tests/test_argo_template.py`
- 修改: `milvus-bricks/milvus_client/tests/test_generate_workflow_report.py`
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`

**步骤 1: 为 2.6 workflow 添加配置参数**

添加参数：
- `base-json-shredding-enabled`
- `target-json-shredding-enabled`
- `rollback-json-shredding-enabled`
- `target-loon-ffi-enabled`
- `post-upgrade-json-shredding-enabled`
- `post-upgrade-config-toggle-enabled`
- `forward-schema-matrix`
- `forward-workload-enabled`
- `rollback-forward-validation-enabled`

**步骤 2: 在 deploy/patch 中写入 `spec.config`**

在 `deploy-milvus` 创建 base CR 时写入：

```yaml
config:
  common:
    storage:
      jsonShreddingEnabled: {{workflow.parameters.base-json-shredding-enabled}}
```

在升级 patch 里按参数写入 target config。`common.storage.useLoonFFI` 仅在 target/master/3.0 阶段启用，不写入 2.6 base 或 2.6 rollback。

**步骤 3: 增加配置 snapshot**

新增 `snapshot-milvus-config` template，在 base deploy、upgrade、post-upgrade toggle、rollback 后采集 Milvus CR YAML/JSON 和关键配置到 `/tmp/milvus-bricks/k8s/config-*.json`。

**步骤 4: 报告记录配置矩阵**

`generate_workflow_report.py` 新增参数并输出到 `parameters.config_matrix` 和 Markdown。

**步骤 5: 测试**

运行：

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py -v
argo lint argo/standalone-2-6-upgrade-rollback.yaml
```

预期：pytest 和 argo lint 均通过。

## Phase 2: 2.6->3.0 target 兼容升级/回滚支持

**文件：**
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`
- 修改: `milvus-bricks/milvus_client/tests/test_argo_template.py`

**步骤 1: 放宽 target 参数语义**

保持 workflow 名称兼容已有入口，但文档说明它支持：
- 2.6 base -> 2.6 target -> 2.6 rollback
- 2.6 base -> 3.0/master target -> 2.6 rollback

**步骤 2: 默认关闭 3.0 schema**

当 rollback target 是 2.6 时，默认 `forward-workload-enabled=false`，保证只把 2.6 compatible 数据设为 hard gate。

**步骤 3: 测试**

运行：

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py -v
argo lint argo/standalone-2-6-upgrade-rollback.yaml
```

预期：通过。

## Phase 3: Post-upgrade config toggle 和 forward-only workload

**文件：**
- 修改: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `milvus-bricks/milvus_client/requests/generate_workflow_report.py`
- 修改: `milvus-bricks/milvus_client/tests/test_argo_template.py`

**步骤 1: 增加 post-upgrade toggle**

在 `validate-after-upgrade` 后支持可选 `patch-post-upgrade-config`，将 jsonShredding 切到目标值并等待 ready。

**步骤 2: 增加 forward-only 3.0 workload 分支**

当 `forward-workload-enabled=true` 时：
- 创建 `forward-schema-matrix`
- seed forward data
- validate forward data after upgrade
- 回滚前记录 forward result

当 rollback 到 2.6 时，`rollback-forward-validation-enabled=false`，回滚后不验证 3.0 数据为 hard gate。

**步骤 3: 测试**

运行：

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py milvus_client/tests/test_generate_workflow_report.py -v
argo lint argo/standalone-2-6-upgrade-rollback.yaml
```

预期：通过。

## Phase 4: 3.0 baseline workflow

**文件：**
- 创建: `milvus-bricks/argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `milvus-bricks/milvus_client/tests/test_argo_template.py`
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`

**步骤 1: 复制 2.6 workflow 并改为 3.0 baseline**

默认参数：
- `base-milvus-image`: `harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14`
- `base-version`: `3.0.0`
- `schema-matrix`: `milvus_client/manifests/schema_matrix_3_0.yaml`
- `rollback-forward-validation-enabled=true`

**步骤 2: 保持 RBAC 复用**

继续使用 `milvus-upgrade-rollback-runner`，不新增 cluster scope 权限。

**步骤 3: 测试**

运行：

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py -v
argo lint argo/standalone-3-0-upgrade-rollback.yaml
```

预期：通过。

## Phase 5: 4am Argo 三轮实测和报告

**文件：**
- 创建/更新: `milvus-bricks/milvus_client/docs/reports/2026-07-08-milvus-config-matrix-implementation-report.md`

**步骤 1: 解析具体 image tag**

使用 4am Harbor 的 concrete tag，不在正式验证中使用 mutable latest。

**步骤 2: 三轮 workflow 验证**

建议顺序：
1. 2.6 jsonShredding -> 2.6-latest jsonShredding -> rollback 2.6。
2. 2.6 jsonShredding -> master jsonShredding -> rollback 2.6。
3. 3.0 baseline -> master -> rollback 3.0。

每轮记录：
- workflow name
- image tags
- data scale
- schema matrix
- request modules
- pressure duration/concurrency
- final status
- failed/warning artifacts
- blocker 和优化点

**步骤 3: Phase review 和优化**

每轮实测后：
- 如果 workflow/template/brick bug 导致失败，修复后重跑同一轮。
- 如果是 Milvus 产品兼容性失败，记录为 blocker，不掩盖为 workflow 成功。
- 如果是资源、超时、日志、artifact 可观测性不足，作为优化点修复。

**步骤 4: 最终报告**

报告必须包含：
- 已实现能力
- 5 个场景当前支持状态
- 每轮测试结果
- 覆盖的数据类型、schema 特性、索引类型和 request
- blocking issues
- follow-up 优化项

## 完成标准

- 不修改 `milvus-bricks/2.6/`。
- 所有新增/修改 WorkflowTemplate 通过 `argo lint`。
- `milvus_client/tests` 全量通过。
- `git diff --check origin/main...HEAD` 通过。
- 至少完成三轮 4am Argo workflow 验证或明确记录阻塞原因。
- 报告文档落库，最终结论不依赖口头描述。
