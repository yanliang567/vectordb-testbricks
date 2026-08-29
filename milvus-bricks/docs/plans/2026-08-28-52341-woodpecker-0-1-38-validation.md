# Milvus #52341 Woodpecker 0.1.38 验证实现计划

**目标：** 用显式锁定的 Woodpecker server 0.1.38 和包含 Woodpecker client 0.1.38 的最新 Milvus 3.0 分支镜像，验证 `v3.0.0 -> 最新 3.0 -> v3.0.0` 在持续 DML 下不再发生 channel tSafe 永久停滞。

**架构：** 复用现有 cluster JSON Shredding full-DML known-limitation 场景，新增版本化 1CU Woodpecker deploy profile，并为该场景提供一个从 `schema_matrix_2_6.yaml` 精确选取的回归矩阵。回归矩阵保留原问题中出现过的普通 collection 和多 channel 压力，只排除会独立触发 #52768 的两个 nested scalar AutoIndex schema。验证执行期间保持非 release-gate，并在失败时保留 Milvus 环境；验证通过后将该受限九 schema 合同提升为 release gate。

**技术栈：** Python、PyYAML、pytest、Ruff、Argo Workflows、Helm、Milvus cluster、Woodpecker 0.1.38。

**验证后状态：** 三次完整 workflow 均通过，#52341 已关闭；manifest 已提升为 `gate / supported_with_config_constraints / release-gate-eligible=true`，#52768 继续由独立限制场景跟踪。

---

### 任务 1：支持可维护的 schema 子集矩阵

**文件：**

- 修改：`milvus_client/common/schema.py`
- 修改：`milvus_client/tests/test_schema_manifest.py`
- 创建：`milvus_client/manifests/schema_matrix_2_6_woodpecker_reader_recovery.yaml`

**步骤 1：写失败测试**

- 验证 matrix 可以用 `source_matrix` 和 `include_schemas` 从同目录基础 matrix 精确选取 schema。
- 验证缺失 schema、重复 schema、同时声明 `schemas` 与 `source_matrix` 时拒绝加载。

**步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest milvus_client/tests/test_schema_manifest.py -q
```

预期：新增的组合矩阵测试失败。

**步骤 3：实现最小加载逻辑**

- 仅支持一层本地相对路径 `source_matrix`。
- 保持源矩阵 schema 原始顺序。
- `include_schemas` 必须是非空、无重复的字符串列表，且每个名称必须存在。
- 现有直接声明 `schemas` 的矩阵行为不变。

**步骤 4：新增 #52341 专用矩阵**

从 `schema_matrix_2_6.yaml` 保留除以下两项外的 9 个 schema：

- `struct_array_varchar_autoindex_rollback_safe`
- `struct_array_numeric_autoindex_rollback_safe`

这两个排除项只用于隔离 #52768，不改变任何 release-gate 场景。

### 任务 2：锁定 Woodpecker server 0.1.38

**文件：**

- 创建：`milvus_client/manifests/deploy_profiles/cluster-woodpecker-v0-1-38-1cu.yaml`
- 修改：`milvus_client/tests/test_deploy_profiles.py`

**步骤 1：写失败测试**

- 验证新 profile 使用 `harbor.milvus.io/milvusdb/woodpecker:v0.1.38@sha256:bdea08758377fea309c18087334c63d20e26ba0940a4d63369bf7794f5f2060e`，避免 release tag 被覆盖后改变 gate 输入。
- 验证 topology、Helm chart 和依赖配置与现有 Woodpecker 1CU profile 一致。

**步骤 2：创建版本化 profile**

- 复制 1CU topology。
- 仅将 profile 名称和 Woodpecker tag 改为版本化值。
- 保持 `pullPolicy: Always` 和 amd64 node selector。

### 任务 3：收敛 #52341 manifest 合同

**文件：**

- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py`
- 更新：`milvus_client/tests/fixtures/upgrade_rollback_execution_paths_v1.yaml`

**步骤 1：写失败测试**

- 验证执行前，场景保持 `known_limitation / unsupported / release-gate-eligible=false`；三次验证通过并关闭 #52341 后，提升为 `gate / supported_with_config_constraints / release-gate-eligible=true`。
- 场景必须使用 WP 0.1.38 版本化 profile。
- 场景必须使用 #52341 专用 9-schema 矩阵。
- 描述必须分别准确引用 #52341 和被隔离的 #52768。

**步骤 2：修改 manifest 并重新生成 fixture**

- 注册新 deploy profile 和 schema matrix。
- 更新现有 cluster JSON Shredding full-DML 场景，不新增重复 scenario ID。
- 用代码管理的 renderer 重新生成 execution path fixture。

### 任务 4：静态验证与提交准备

运行：

```bash
python3 -m pytest milvus_client/tests -q
ruff check milvus_client
ruff format --check milvus_client
argo lint argo/cluster-upgrade-rollback.yaml
```

预期：全部退出码为 0。随后提交并推送验证分支，以便 Argo workflow 能通过 `repo-revision` 获取新 profile 和矩阵。

### 任务 5：QA Argo 端到端验证

1. 将 `origin/main` 中 PR #44 合入后的 `argo/cluster-upgrade-rollback.yaml` 应用到 `qa`，确认全局 `templateDefaults.retryStrategy` 已移除，仅四个幂等基础设施 template 有重试。
2. 用以下镜像渲染 #52341 场景：
   - base/rollback：`harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
   - target：`harbor.milvus.io/milvusdb/milvus:3.0-20260828-8bd63b88@sha256:61479222e9229d88df0decbba388a07a4890e212573c7a739f7e9a5a7212affb`
   - Woodpecker server：`harbor.milvus.io/milvusdb/woodpecker:v0.1.38@sha256:bdea08758377fea309c18087334c63d20e26ba0940a4d63369bf7794f5f2060e`
3. 主路径重复运行三轮；每轮设置 `keep-milvus=true`，确认结果后再按 workflow ownership 精确清理通过的环境，失败环境保留。
4. 检查所有 collection rollback serviceability、count/query/search、持续 DML、tSafe，以及 `no record extract`、`reader temp info not found`、`update reader info failed` 日志。
5. 如果主路径失败，保留现场并运行全阶段最新 3.0/WP client 0.1.38 的同版本控制组，区分旧 client 兼容问题与修复未生效。

### 任务 6：验证报告

**文件：**

- 创建：`milvus_client/docs/reports/2026-08-28-52341-woodpecker-0-1-38-validation.md`

记录每轮 workflow URL、精确镜像/digest、repo commit、有效 Helm values、Pod UID、压力统计、rollback serviceability、关键日志计数、结论和保留环境信息。
