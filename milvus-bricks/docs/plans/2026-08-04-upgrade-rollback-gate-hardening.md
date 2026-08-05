# Upgrade/Rollback Gate Hardening 实现计划

**目标：** 强化 Milvus 升级/回滚 gate 的场景契约、Standalone 服务恢复门禁和 JSON Shredding 正向兼容性覆盖，同时继续复用现有 3 条参数化 WorkflowTemplate。

**架构：** `upgrade_rollback_gates.yaml` 继续作为场景定义层，`common/gates.py` 负责解析和提交参数，3 条 Argo WorkflowTemplate 作为执行引擎。本轮不拆分为一场景一 Workflow，也不把 target 版本重建索引后回滚作为核心验证。

**技术栈：** Python、pytest、PyYAML、Argo WorkflowTemplate、Milvus Operator/Helm、PyMilvus。

---

## 实施范围

本轮完成：

1. 全场景与 Workflow 参数契约校验，并支持正式运行时覆盖 placeholder 镜像。
2. Standalone 升级后的数据 serviceability 重试门禁。
3. JSON Shredding 正向 upgrade/rollback gate，包括配置切换后写入和回滚读取。
4. 2.6 -> 3.0 target-only feature upgrade gate。
5. Woodpecker 2CU HA 滚动升级/回滚 gate。

## 非目标

- 不新增 6 份场景专属 Workflow YAML。
- 不在正向 gate 中将 3.0-only 数据回滚到 2.6。
- 不启用 `validate_index_compatibility --rebuild-index`。
- 不在本轮增加多副本可用性 SLO。

---

### 任务 1：场景与 Workflow 契约校验

**文件：**

- 修改：`milvus_client/common/gates.py`
- 修改：`milvus_client/requests/render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_argo_template.py`

**步骤：**

1. 为 `resolve_gate_scenario()` 增加 base/target/rollback image 和 version override。
2. CLI 增加对应 override 参数，正式 render 仍拒绝未覆盖的 placeholder。
3. 遍历 manifest 中全部场景，验证每条都能 resolve 和 render。
4. 读取场景对应 WorkflowTemplate，验证 renderer 输出参数全部被 Workflow 声明。
5. 读取 deploy profile，验证 `scenario.mode`、profile mode 和 Workflow 类型一致。
6. 为普通 standalone 3.0 gate 增加独立渲染断言。

**验收：**

- 7 条场景全部通过统一 contract test。
- 参数拼写、profile mode、Workflow ref 漂移会在 pytest 阶段失败。
- 不传具体 latest 镜像时正式 render 失败；提供 override 后成功。

---

### 任务 2：Standalone Upgrade Serviceability 门禁

**文件：**

- 修改：`argo/standalone-2-6-upgrade-rollback.yaml`
- 修改：`argo/standalone-3-0-upgrade-rollback.yaml`
- 修改：`milvus_client/tests/test_argo_template.py`

**步骤：**

1. 在两个 Standalone DAG 中增加 `wait-upgrade-serviceability`。
2. 复用 `milvus_client.requests.wait_data_serviceability`。
3. 使用 baseline seed checkpoint 和当前 schema matrix。
4. task 放在 `precheck-after-upgrade` 与 `validate-after-upgrade` 之间。
5. 保持只重试明确 channel/shard leader serviceability 错误，correctness failure 立即失败。

**验收：**

- Standalone 和 Cluster 都在升级后执行 serviceability 门禁。
- task 参数包含 timeout/interval 和 baseline checkpoint。
- DAG 依赖保证数据验证不会早于 serviceability 恢复。

---

### 任务 3：JSON Shredding 正向 Gate

**文件：**

- 创建：`milvus_client/manifests/schema_matrix_json_shredding.yaml`
- 修改：`milvus_client/common/data.py`
- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改：`argo/standalone-3-0-upgrade-rollback.yaml`
- 修改：`argo/cluster-upgrade-rollback.yaml`
- 修改：`argo/standalone-2-6-upgrade-rollback.yaml`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_argo_template.py`
- 修改：`milvus_client/tests/test_data_generation.py`
- 修改：`milvus_client/tests/test_schema_manifest.py`

**场景：**

`standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`

**配置：**

- base：JSON Shredding 关闭。
- target 初始：JSON Shredding 关闭。
- post-upgrade config：JSON Shredding 开启。
- forward workload：开启，使用 JSON-heavy schema matrix。
- rollback：JSON Shredding 保持开启。
- rollback forward validation：开启。

**步骤：**

1. 增加 JSON-heavy schema，覆盖 nested JSON、nullable JSON、ARRAY 和 dynamic field。
2. 在 post-upgrade config rollout 后创建、写入并校验 forward collection。
3. 回滚前等待 forward collection serviceability。
4. 回滚后通过 forward checkpoint 校验 count、PK 和 scalar checksum。
5. 扩展运行时配置 assertion，检查 declared/runtime `jsonShreddingEnabled`。
6. base、after-upgrade、post-config、after-rollback 四个阶段分别断言期望配置。

**验收：**

- 确认 JSON 数据写入发生在配置切换完成后。
- 回滚后 forward JSON collection 数据完整且可查询。
- 配置声明值与 Milvus runtime config 一致。
- 场景保持 strict data、serviceability 和 pressure gate 策略。

---

### 任务 4：2.6 -> 3.0 Target-Only Feature Gate

**文件：**

- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改：`milvus_client/common/gates.py`
- 修改：`argo/standalone-2-6-upgrade-rollback.yaml`
- 修改：`argo/standalone-3-0-upgrade-rollback.yaml`
- 修改：`argo/cluster-upgrade-rollback.yaml`
- 修改：`milvus_client/requests/generate_workflow_report.py`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_argo_template.py`
- 修改：`milvus_client/tests/test_generate_workflow_report.py`

**场景：**

`standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest`

**步骤：**

1. 使用 `schema_matrix_2_6.yaml` 创建 rollback-safe baseline 数据。
2. 升级到 3.0 后使用 `schema_matrix_3_0.yaml` 创建 forward collections。
3. 对 forward collections 执行数据完整性、索引 metadata、load/search/query 和 schema evolution 验证。
4. 回滚到 2.6 后只要求 baseline 数据、索引和 phase DML/DQL 继续工作。
5. 明确禁止把 3.0-only forward collections 设置为 rollback required validation。
6. 最终报告将 forward index upgrade validation 纳入 required results；只有 rollback forward validation 开启时才要求 rollback forward index result。

**验收：**

- 3.0-only schema/index 能力在 target 阶段被实际创建、写入和查询。
- target-only forward index checkpoint 独立存放，不覆盖 baseline checkpoint。
- 2.6 rollback 不因预期不可兼容的 3.0-only collections 判失败。
- baseline 2.6 数据在升级和回滚后继续通过严格门禁。

---

### 任务 5：Woodpecker 2CU HA 滚动升级/回滚 Gate

**文件：**

- 修改：`milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml`
- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改：`milvus_client/common/gates.py`
- 修改：`argo/cluster-upgrade-rollback.yaml`
- 修改：`milvus_client/tests/test_deploy_profiles.py`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py`
- 修改：`milvus_client/tests/test_render_milvus_cr.py`
- 修改：`milvus_client/tests/test_argo_template.py`

**场景：**

`cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline`

**步骤：**

1. 使用 Woodpecker 2CU profile，Proxy、QueryNode、DataNode 和 StreamingNode 均保持至少 2 副本。
2. 复用 cluster Helm rolling upgrade/rollback DAG，不新增 WorkflowTemplate。
3. 在场景 resolve 和 Helm deploy 前校验实际 deploy profile 满足最小副本契约。
4. 保持存储特性关闭，复用 strict pressure、serviceability、数据、索引、phase DML/DQL 和 schema evolution 门禁。
5. 不增加零请求失败或多副本可用性 SLO。

**验收：**

- 2CU topology 被实际渲染进 Helm values 和最终 topology summary。
- 使用 1CU profile override 时，在提交渲染或 Helm deploy 前失败。
- 升级和回滚后继续通过既有严格正确性门禁。

---

## 验证命令

```bash
PYTHONPATH=. uv run pytest -q \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_argo_template.py

PYTHONPATH=. uv run pytest -q

uvx ruff check \
  milvus_client/common/data.py \
  milvus_client/common/gates.py \
  milvus_client/requests/render_upgrade_rollback_params.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_argo_template.py \
  milvus_client/tests/test_data_generation.py \
  milvus_client/tests/test_schema_manifest.py

uvx ruff format --check \
  milvus_client/common/gates.py \
  milvus_client/requests/render_upgrade_rollback_params.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_argo_template.py

git diff --check
```

## 提交拆分

1. `test: validate upgrade rollback scenario contracts`
2. `feat: support concrete gate image overrides`
3. `fix: wait for standalone upgrade serviceability`
4. `feat: add json shredding upgrade rollback gate`
