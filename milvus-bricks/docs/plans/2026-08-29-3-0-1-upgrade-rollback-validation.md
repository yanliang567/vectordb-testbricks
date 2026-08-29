# Milvus 3.0.1 升级/回滚发布验证执行计划

**目标：** 使用 2026-08-29 最新的 Milvus 3.0 分支多架构镜像完成 3.0.1 发布前升级/回滚验证，在 standalone 最多 3 并发、cluster 最多 2 并发的约束下跑完正式 gate 和非门禁跟踪场景，修复测试基础设施阻塞并输出可审计的测试报告。

**架构：** 以 `upgrade_rollback_gates.yaml` 为唯一场景合同源，运行时只覆盖 concrete image、semantic version 和 merge commit，不修改场景验证语义。正式 gate 分波次并行提交；known limitation/negative 场景与 release gate 结果分开统计。测试代码或 Argo 模板问题采用“先复现和补测试、最小修复、局部重跑、相邻路径回归”的闭环，Milvus 产品回归不通过排除用例或降级断言来掩盖。

**技术栈：** Python/PyYAML、pytest、Ruff、Argo Workflows、Kubernetes、Helm、Harbor OCI manifest digest、Milvus standalone/cluster。

---

### 任务 1：锁定不可变输入和执行基线

**文件：**
- 检查: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 检查: `milvus_client/manifests/deploy_profiles/*.yaml`
- 检查: `argo/*.yaml`
- 记录: `milvus_client/docs/reports/2026-08-29-3-0-1-upgrade-rollback-validation.md`

**步骤 1：记录测试代码版本**

运行：

```bash
git fetch origin main
git rev-parse origin/main
```

预期：所有正式 workflow 的 `repo-revision` 使用同一个完整 merge SHA。

**步骤 2：解析并固定最新分支镜像**

运行：

```bash
python3 /Users/yanliang.qiao/.codex/skills/milvus-image-tag/scripts/find_milvus_image_tags.py \
  --branches '2.6,3.0' --format json
docker manifest inspect '<3.0-tag>@<3.0-manifest-list-digest>'
docker manifest inspect '<2.6-tag>@<2.6-manifest-list-digest>'
```

本轮锁定输入：

- 3.0/3.0.1 candidate：`harbor.milvus.io/milvusdb/milvus:3.0-20260829-257a535b@sha256:d3a0d1d64368139ab59a28989392bcffdefaaa0f724596b870ed7c0b16d15c20`
- 2.6 latest rollback：`harbor.milvus.io/milvusdb/milvus:2.6-20260829-3b859656@sha256:989e085e45c44f513387f361c0c6b326a434a0828964798a459728460ebe04b6`
- 3.0.0 baseline：沿用 manifest 中已固定的 `v3.0.0@sha256:49371c30...`。

预期：三者均可通过 immutable `tag@digest` 拉取；正式运行期间不重新解析 “latest”。

**步骤 3：验证模板和 live template 一致性**

运行：

```bash
argo lint -n qa argo/*.yaml
kubectl diff -n qa -f argo/
```

预期：lint 通过；live WorkflowTemplate 已包含 PR #45 合入后的 scoped retry 和不可重试 DML wrapper。只有确认 drift 后才 apply reviewed manifest。

### 任务 2：预渲染全部 3.0.1 发布 gate

**文件：**
- 检查: `milvus_client/requests/render_upgrade_rollback_params.py`
- 测试: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 测试: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`

**步骤 1：按合同类型设置 runtime override**

- `2.6 -> 3.0.1 -> 2.6`：base 使用固定 v2.6.18；target 使用本轮 3.0 candidate 且 `target-version=3.0.1`；rollback 使用本轮 2.6 latest 且 `rollback-version=2.6.0`。
- `3.0.0 -> 3.0.1 -> 3.0.0`：base/rollback 使用固定 v3.0.0；target 使用本轮 candidate 且 `target-version=3.0.1`。
- `3.0.1 self/config round-trip`：base/target/rollback 都使用本轮 candidate，三个 semantic version 都为 `3.0.1`。
- `3.0.0 -> 3.0.1 -> 3.0.1`：base 使用固定 v3.0.0；target/rollback 使用本轮 candidate 且版本为 `3.0.1`。

**步骤 2：渲染 20 条正式 gate 并验证关键参数**

Standalone（10 条）：

1. `standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest`
2. `standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline`
3. `standalone-3-0-index-v10-v4-upgrade-rollback`
4. `standalone-3-0-index-v11-v4-upgrade-rollback`
5. `standalone-3-0-1-vortex-self-compat-upgrade-rollback`
6. `standalone-3-0-0-to-3-0-1-vortex-enable-rollback`
7. `standalone-3-0-1-json-shredding-vortex-rollback`
8. `standalone-3-0-1-loon-ffi-rollback`
9. `standalone-3-0-1-vortex-disable-rollback`
10. `standalone-3-0-1-vortex-disable-keep-loon-rollback`

Cluster（10 条）：

1. `cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest`
2. `cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline`
3. `cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`
4. `cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline`
5. `cluster-3-0-index-v10-v4-upgrade-rollback`
6. `cluster-3-0-index-v11-v4-upgrade-rollback`
7. `cluster-3-0-1-vortex-self-compat-upgrade-rollback`
8. `cluster-3-0-0-to-3-0-1-vortex-enable-rollback`
9. `cluster-3-0-1-json-shredding-vortex-rollback`
10. `cluster-3-0-1-loon-ffi-rollback`

对每条运行 renderer，预期：无 placeholder；`release-gate-eligible=true`；classification/support status 符合 manifest；Milvus log level 为 `debug`；镜像均为上述 immutable digest；2.6 rollback 路径的 LoonFFI/Vortex 在所有阶段保持关闭。

**步骤 3：运行提交前静态检查**

运行：

```bash
PYTHONPATH=. python3 -m pytest milvus_client/tests -q
argo lint --offline argo
# Ruff check/format 使用 ../.github/workflows/milvus-bricks.yml 中列出的
# 23 个受管 Python 文件，避免把仓库历史脚本的存量 lint 计入本轮 gate。
git diff --check
```

预期：全部通过后才开始正式矩阵。

### 任务 3：按并发上限执行正式 gate

**文件：**
- 使用: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 使用: `argo/standalone-3-0-upgrade-rollback.yaml`
- 使用: `argo/cluster-upgrade-rollback.yaml`

**步骤 1：建立独立的两类并发队列**

- standalone semaphore：最多 3 个 Running workflow。
- cluster semaphore：最多 2 个 Running workflow。
- 两类队列可同时运行，总并发最多 5；新任务只在同类 slot 释放后提交。
- `c30json-p6hz6` 是 PR #45 合入后的预检 run，因其 `target-version=3.0.0`，只记录为前置证据，不替代 3.0.1 正式 gate。

**步骤 2：按风险优先级分波次提交**

优先级顺序：

1. 2.6 跨版本 target-only、3.0.0 baseline round-trip、v10/v4、v11/v4。
2. 3.0.0 -> 3.0.1 Vortex enable、3.0.1 self-compat。
3. JSON Shredding、LoonFFI、Vortex disable/keep-Loon 配置 round-trip。
4. cluster Woodpecker v0.1.38 #52341 和 2CU HA。

每次提交后记录 workflow 名称、参数摘要、开始/结束时间、最终 phase 和 Argo URL。预期：20 条正式 gate 均为 `Succeeded`，且无非幂等 DML retry attempt。

**步骤 3：逐阶段验证每条 workflow**

至少验证：

- base/target/rollback 实际 Pod `imageID` 与期望 digest 一致；Woodpecker 场景也验证依赖镜像 digest。
- 每阶段 Helm/CR 配置与 Pod 合并后的 runtime config 一致，Milvus 日志级别为 debug。
- baseline seed、checkpoint、schema feature、index compatibility、phase DML/DQL、严格压力套件全部通过。
- 升级和回滚观察窗口内持续压力 daemon 无未解释错误。
- target-only 集合在不兼容 rollback 阶段按合同处理；round-trip 集合在 rollback 后继续可读写。
- JSON Shredding/LoonFFI/Vortex 的写入者与读取者边界符合 manifest 合同。
- onExit 完成 Helm/CR 和数据 PVC 清理；失败时按 `keep-milvus` 策略保留问题环境。

### 任务 4：执行非门禁跟踪和负向控制

**文件：**
- 检查: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 记录: `milvus_client/docs/reports/2026-08-29-3-0-1-upgrade-rollback-validation.md`

**步骤 1：正式 gate 完成后运行 3 条 known limitation tracker**

- `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest`（#52893）
- `cluster-2-6-18-to-3-0-latest-rollback-2-6-latest`（#52893）
- `standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`（#52341 standalone 限制）

预期：这些结果单独记录，不参与 release-gate 绿色率；保留原始失败点，不通过删 schema 或放宽断言制造绿灯。

**步骤 2：验证负向合同 guard**

场景：`standalone-3-0-loon-vortex-to-2-6-negative`。

预期：只有显式 unsafe negative coverage 才允许提交，且 2.6 reader 对 storage v3/Vortex 的不兼容按注册负向合同体现；结果不计入正式 gate。

**步骤 3：明确本轮不执行的冻结 candidate**

- `standalone-3-0-vortex-candidate-upgrade-rollback`
- `cluster-3-0-vortex-candidate-upgrade-rollback`

原因：它们固定到 Vortex 0.75 预发布历史镜像，不能覆盖为本轮 3.0 分支 candidate，也不属于 3.0.1 release gate。报告中列为 `not_applicable`，避免误算为漏测。

### 任务 5：阻塞问题修复与定向重跑

**文件：**
- 按实际根因修改 `milvus_client/requests/`、`milvus_client/common/`、`milvus_client/manifests/` 或对应测试。
- 测试: 与修改模块对应的 `milvus_client/tests/test_*.py`

**步骤 1：先分类失败**

分类为：Harbor/集群基础设施、Argo/测试框架、Milvus 产品回归、已注册 known limitation、数据/断言噪音。保存失败 node、attempt、Pod event、Milvus debug log 和 immutable image 信息。

**步骤 2：只修复授权范围内的阻塞**

- 测试代码/模板问题：先写或补充失败测试，运行确认失败，再做最小修复。
- 短暂基础设施问题：仅重试幂等 deploy/wait/patch；非幂等 DML 不自动重放。
- Milvus 产品回归：不改测试掩盖；保留环境、记录 issue 证据并把 gate 标记为真实阻塞。

**步骤 3：验证修复**

运行精确失败测试、相关测试文件、完整 pytest、Ruff、Argo lint 和 `git diff --check`。预期全部通过。

**步骤 4：重跑最小充分集合**

先重跑失败 workflow，再重跑同 template/同合同类型的一条相邻场景。两者通过后继续剩余队列；避免无条件重跑已通过的非幂等 workload。

### 任务 6：生成最终测试报告

**文件：**
- 创建: `milvus_client/docs/reports/2026-08-29-3-0-1-upgrade-rollback-validation.md`

**步骤 1：汇总可审计输入**

记录代码 SHA、3.0/2.6/baseline image tag@digest、WorkflowTemplate generation、Woodpecker digest、执行时段和并发上限。

**步骤 2：汇总结果矩阵**

每条场景记录合同类型、classification、support status、workflow、结果、关键验证点、重试次数、是否计入 release gate、清理状态。

**步骤 3：记录问题与修复**

区分首次失败、根因、修复 commit、定向重跑和最终结论；known limitation、产品回归、基础设施失败分别统计。

**步骤 4：记录优化建议**

建议至少覆盖 suite 调度器、运行前镜像 digest 可用性检查、manifest 中 3.0.1 alias 晋级、失败环境保留/TTL、阶段化可观测性和自动报告生成；只记录有本轮证据支持的建议。

**步骤 5：最终验收**

正式 gate 必须全部通过才能给出 3.0.1 升级/回滚 release-ready 结论。若存在 Milvus 产品阻塞，报告明确列出阻塞场景和证据，不以其他绿色场景抵消。
