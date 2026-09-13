# Milvus 3.0.2 升级/兼容发布验证执行计划

**目标：** 使用 3.0 分支最新多架构镜像作为 3.0.2 release candidate，以 v2.6.20 和 v3.0.1 为基线，完成 vectordb-testbricks 的升级/回滚兼容矩阵；运行中对失败给出可复核归因，修复测试阻塞，记录非阻塞优化项，并生成测试与分析报告。

**架构：** 以 `milvus_client/manifests/upgrade_rollback_gates.yaml` 为场景合同源，初始执行固定到 `origin/main` 完整 SHA，并只通过 renderer 支持的 immutable image/version overrides 刷新版本输入。正式 gate 使用两个独立队列，standalone 最多 4 并发、cluster 最多 3 并发；known limitation 和 negative control 与正式 gate 分开统计。失败按基础设施、测试框架、已知限制和 Milvus 产品问题分类，只有测试阻塞允许最小修复和重跑，产品问题保留原断言及日志/metadata/重复证据。

**技术栈：** Python/PyYAML、pytest、Ruff、Argo Workflows、Kubernetes、Helm、Harbor OCI manifest digest、Milvus standalone/cluster。

---

### 任务 1：固定不可变输入并验证测试框架基线

**文件：**
- 检查: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 检查: `milvus_client/manifests/deploy_profiles/*.yaml`
- 检查: `argo/*.yaml`
- 测试: `milvus_client/tests/`

**步骤 1：固定代码 revision**

运行：

```bash
git fetch origin main
git rev-parse origin/main
git merge-base --is-ancestor HEAD origin/main
```

预期：初始正式 workflow 使用同一个远端完整 SHA；若测试框架被修复，则后续重跑使用包含修复且已推送的完整 SHA，并在报告中明确证据边界。

**步骤 2：固定镜像 `tag@manifest-list-digest`**

运行：

```bash
python3 /Users/yanliang.qiao/.codex/skills/milvus-image-tag/scripts/find_milvus_image_tags.py --branches 3.0 --format json
curl -fsS 'https://harbor.milvus.io/api/v2.0/projects/milvusdb/repositories/milvus/artifacts/v2.6.20'
curl -fsS 'https://harbor.milvus.io/api/v2.0/projects/milvusdb/repositories/milvus/artifacts/v3.0.1'
curl -fsS 'https://harbor.milvus.io/api/v2.0/projects/milvusdb/repositories/milvus/artifacts/3.0-20260911-c4412246'
```

本轮版本语义：

- 2.6 base/rollback：`v2.6.20`，semantic version `2.6.20`。
- 3.0 base/rollback：`v3.0.1`，semantic version `3.0.1`。
- 3.0.2 target candidate：`3.0-20260911-c4412246`，semantic version `3.0.2`。

预期：三者都是可拉取 multi-arch manifest；运行期间不重新解释 `latest`。

**步骤 3：运行框架基线**

运行：

```bash
cd milvus-bricks
PYTHONPATH=. python3 -m pytest milvus_client/tests -q
argo lint --offline argo
git diff --check
```

预期：pytest、Argo lint 和 diff check 全部通过；若基线失败，先归因并修复后再提交 E2E。

### 任务 2：预渲染和审计 20 条正式 gate

**文件：**
- 使用: `milvus_client/requests/render_upgrade_rollback_params.py`
- 测试: `milvus_client/tests/test_render_upgrade_rollback_params.py`
- 测试: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 生成: `artifacts/3-0-2-upgrade-compatibility/rendered/`

**步骤 1：建立版本覆盖规则**

- 所有 2.6 合同的 base/rollback 都覆盖为 v2.6.20；target 覆盖为 3.0.2 candidate。
- 所有常规 3.0 合同的 base/rollback 都覆盖为 v3.0.1；target 覆盖为 3.0.2 candidate。
- 原 `3.0.1 self/config round-trip` 场景使用 v3.0.1 base/rollback 和 candidate target。
- 原 `3.0.0 -> 3.0.1 Vortex enable` 场景刷新为 v3.0.1 legacy base、candidate Vortex target、v3.0.1 Vortex rollback，用于覆盖配置与格式跨 patch reader/writer。

**步骤 2：渲染 standalone 10 条 gate**

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

**步骤 3：渲染 cluster 10 条 gate**

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

对每条 renderer 输出检查：无 placeholder；`release-gate-eligible=true`；classification/support status 符合 manifest；`milvus-log-level=debug`；镜像为本轮固定 digest；2.6 rollback 合同所有阶段关闭 LoonFFI/Vortex。

### 任务 3：安装隔离模板并按 4/3 并发运行正式 gate

**文件：**
- 使用: `argo/standalone-2-6-upgrade-rollback.yaml`
- 使用: `argo/standalone-3-0-upgrade-rollback.yaml`
- 使用: `argo/cluster-upgrade-rollback.yaml`
- 生成: `artifacts/3-0-2-upgrade-compatibility/runs.json`

**步骤 1：避免覆盖共享 WorkflowTemplate**

为本轮复制带唯一后缀的 WorkflowTemplate 名称，执行 `argo lint` 和 `kubectl diff` 后 apply；所有 workflow 参数记录完整 test revision 和 scenario id。

**步骤 2：建立双队列 scheduler**

- standalone semaphore：同一时间最多 4 个本轮 workflow 为 Pending/Running。
- cluster semaphore：同一时间最多 3 个本轮 workflow 为 Pending/Running。
- 每 30–60 秒读取 Argo phase；只有同类型 slot 释放后才提交下一条。
- 每次提交记录 workflow、场景、参数摘要、开始时间和 Argo URL。

**步骤 3：风险优先提交**

先跑 2.6 跨版本、3.0 core round-trip、v10/v4、v11/v4；再跑 Vortex enable/self-compat；最后跑 JSON Shredding、LoonFFI、Vortex disable/keep-Loon 和 Woodpecker 2CU HA。

**步骤 4：逐 workflow 验证证据**

验证 base/target/rollback imageID、runtime config、schema/data checkpoint、index reader/writer、phase DML/DQL、新集合、release/reload、压力窗口和 serviceability；成功环境允许 onExit 清理，失败环境默认保留并加原因记录。

### 任务 4：失败归因、测试阻塞修复与充分重跑

**文件：**
- 按根因修改: `milvus_client/requests/`、`milvus_client/common/`、`milvus_client/manifests/` 或 `argo/`
- 测试: 对应 `milvus_client/tests/test_*.py`
- 记录: `artifacts/3-0-2-upgrade-compatibility/incidents/`

**步骤 1：使用 systematic-debugging 流程取证**

保存失败 node/attempt、Pod event、容器 termination、Milvus debug 日志、实际 imageID、CR/Helm/runtime config、brick JSON、checkpoint 和最小复现；先判断失败发生在产品请求路径还是测试/控制面路径。

**步骤 2：修复测试阻塞**

为可复现的测试框架问题先添加失败回归测试，再做最小修复。基础设施无效轮次仅重跑完整 workflow；只对明确幂等的 deploy/wait/patch 使用 retry，seed/DML 等非幂等节点不原地重试。

**步骤 3：确认 Milvus 产品问题**

至少需要稳定复现或跨 standalone/cluster 一致证据，并把客户端错误、服务端日志/stack、受影响 collection/channel/segment/index、版本对比和排除测试误判的证据串联起来。不得删除 schema、放宽 gate 或吞掉错误。

**步骤 4：验证并重跑**

测试修复后运行精确回归、相关测试文件、完整 pytest、Ruff、Argo lint 和 `git diff --check`；提交并推送测试分支后，先重跑失败场景，再跑一条同 template/合同的相邻场景。

### 任务 5：运行非 gate 观察项

**文件：**
- 使用: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 记录: `artifacts/3-0-2-upgrade-compatibility/non-gates.json`

**步骤 1：known limitation trackers**

- standalone/cluster strict full `2.6 -> 3.0 -> 2.6`。
- standalone JSON Shredding rollback tracker。

这些结果独立于 release gate 统计；如果历史问题在 v2.6.20/v3.0.1 边界已修复，记录为 limitation candidate for closure，但不自动修改支持合同。

**步骤 2：negative control**

运行 `standalone-3-0-loon-vortex-to-2-6-negative`，验证只有显式 unsafe negative coverage 才可提交，且失败边界符合预期。

**步骤 3：冻结 candidate 场景**

历史 `standalone/cluster-3-0-vortex-candidate-upgrade-rollback` 固定旧预发布镜像，不纳入本轮 3.0.2 gate；在报告中列为 not applicable。

### 任务 6：生成测试与分析报告并最终验收

**文件：**
- 创建: `milvus_client/docs/reports/2026-09-13-milvus-3-0-2-upgrade-compatibility-validation.md`

**步骤 1：汇总不可变输入和并发证据**

记录代码 SHA、三类 image `tag@digest`、WorkflowTemplate 名称/generation、依赖镜像、执行时段，并从 workflow 时间线证明 standalone≤4、cluster≤3。

**步骤 2：汇总结果矩阵**

每条场景记录 classification、support status、workflow URL、phase/节点数、关键验证点、重试、是否计入 release gate、环境保留/清理状态。

**步骤 3：汇总问题分析**

分别报告 Milvus bug、测试框架修复、基础设施无效轮次、已知限制和优化建议；每个 Milvus bug 附可复核的证据链，每个测试修复附失败测试与重跑证据。

**步骤 4：最终验证**

重新运行完整 pytest、受管 Ruff check/format、Argo lint、报告链接/数字一致性检查和 `git diff --check`。只有 20 条正式 gate 全部通过且无未解释失败，才给出 upgrade/compat release-ready；否则明确列出 blocker，不能用其他绿色结果抵消。
