# Index Contract E2E Matrix 改动与执行报告

## 结论

PR #43 合入后的 index-engine 优先验证矩阵已完成：standalone/cluster 的
v10/v4、v11/v4 四条 `target_only` release gate 均通过。完整 11-schema 的
standalone `2.6.18 -> 3.0 branch -> 2.6 branch` 路径稳定复现 #52893，因此它和
cluster 对应路径在 PR #44 review 后被明确降级为 `known_limitation + unsupported +
release-gate-eligible=false`，不能以排除失败 schema 后的绿色结果证明完整支持合同。

执行过程中发现并修正了三类测试基础设施/合同边界问题：

1. cluster Helm patch 可能重建 Woodpecker Pod；模板现在用 `OnDelete` 并比较
   patch 前后的 Pod `name=uid`。`OnError` retry 只配置在 deploy、ready wait 和
   两个 Helm patch 幂等基础设施模板，绝不覆盖非幂等 DML brick。
2. v3.0.0 携带的 Woodpecker client 无法读取较新 3.0 branch client 留下的
   reader 临时元数据。v10/v4、v11/v4 是 index-engine 专项合同，cluster 路径改用
   Pulsar 1CU 隔离消息存储兼容性；Woodpecker 仍由独立 gate 覆盖。
3. 3.0 target 创建的两个 StructArray nested scalar AutoIndex 格式无法由 2.6
   reader 加载。最终 manifest 保留完整 matrix，并让严格失败路径持续跟踪
   #52893；曾用于诊断的 9-schema 子集不进入最终合同。

## 输入与版本

- 基线代码：PR #43 merge commit
  `c7c7b6901bef458906d7fdcd24bfb1e0e8f3f05f`。
- 本轮修正：`af1d259f40c9066c7c0cb5e3e89fb8449b19083c`
  (`fix: stabilize upgrade rollback gate matrix`)。
- v3.0.0 baseline/rollback：
  `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`。
- v2.6.18 baseline：
  `harbor.milvus.io/milvusdb/milvus:v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6`。
- 2.6 branch rollback：
  `harbor.milvus.io/milvusdb/milvus:2.6-20260827-4d3c88af@sha256:05ca40c9caac21f8f799e1eb8899d253e9954b5278d87ca998042578b430b7a1`。
- 最初指定的 target
  `harbor.milvus.io/milvusdb/milvus:3.0-20260826-e47a679a` 在 standalone
  验证后已被 Harbor retention 清理。后续使用最接近且包含 #52767 修正、但不含
  后续 Woodpecker #52904 变更的不可变替代镜像：
  `harbor.milvus.io/milvusdb/milvus:3.0-20260827-f78de400@sha256:40aeeec2c833ccb695690637d62379aeb152f9c656cc3f8f9bd58bf1b45c35e3`。
- 所有正式场景渲染参数均为 `milvus-log-level=debug`。

## 最终 E2E 矩阵

时间均为 UTC。`failed_all` 包含压力进程记录的预期/允许失败样本；四条正式
index-engine gate 的最终严格压力 `failed` 均为 0。

| 场景 | Workflow | 合同 / topology | 结果与时间 | 压力 `failed / failed_all / passed / total` |
| --- | --- | --- | --- | --- |
| standalone v10/v4 | `milvus-standalone-3-0-upgrade-rollback-7g224` | `target_only`, Rocksmq | Succeeded `63/63`, 2026-08-27 14:06:25–15:08:24 | `0 / 19 / 192 / 211` |
| standalone v11/v4 | `milvus-standalone-3-0-upgrade-rollback-frj7m` | `target_only`, Rocksmq | Succeeded `63/63`, 2026-08-27 16:08:05–17:13:24 | `0 / 21 / 210 / 231` |
| cluster v10/v4 | `c30-index-v10-v4-6bp49` | `target_only`, Pulsar 1CU | Succeeded `63/63`, 2026-08-28 06:08:37–07:15:09 | `0 / 19 / 215 / 234` |
| cluster v11/v4 | `c30-index-v11-v4-w7tm4` | `target_only`, Pulsar 1CU | Succeeded `63/63`, 2026-08-28 07:16:32–08:20:56 | `0 / 18 / 217 / 235` |
| standalone 2.6 strict tracker | `s2618-30-rb26-none-8bx5f` | `known_limitation`, Rocksmq, full 11-schema matrix | #52893 reproduced at rollback serviceability; stopped `45/46`, 2026-08-28 08:22:42–09:07:47 | N/A |
| standalone 2.6 diagnostic subset | `s2618-30-rb26-safe-7pplv` | diagnostic only, Rocksmq, reduced 9-schema matrix | Succeeded `58/58`, 2026-08-28 09:20:10–10:19:53 | `0 / 19 / 190 / 209` |

前两条 standalone workflow 使用原始 e47 target 与 PR #43 revision。后两条
cluster workflow 使用 digest-pinned f78 target；其运行时测试代码 revision 仍固定为
PR #43，但提交的 WorkflowSpec 包含本轮 cluster 模板保护。两次 2.6 运行使用
f78 target；9-schema 成功运行仅用于隔离根因，不作为 release-gate 证据。

## 验证点

### v10/v4 与 v11/v4 target-only

- baseline 和 rollback 固定为 v3.0.0，使用 rollback-safe 11-schema matrix；
  target phase 分别设置 vector/scalar runtime version `10/4` 和 `11/4`。
- target phase 的 existing/new/reload 各 11 个 collection，合同失败为 0。
- target-only phase-new collection 在回滚前全部删除；rollback 校验为
  `absent=11, present=0`，没有要求 v3.0.0 验证其不具备的 #52767 能力。
- baseline collections 在升级和回滚后继续执行 reload、index、search/query、
  DML/DQL；严格压力窗口和最终 gate 均通过。
- standalone 与 cluster 两种 topology 的 v10/v4、v11/v4 矩阵均已覆盖。

### 2.6 known-limitation tracker

- 严格路径使用完整 11-schema `schema_matrix_2_6.yaml`，target phase 的
  existing/new/reload 检查均通过，回滚到 2.6 后因 nested scalar STL_SORT
  `is_nested=true` 无法加载而失去 serviceability，准确复现 #52893。
- 诊断时排除两个 schema 后，9-schema 子集确实可以完成 rollback、DML/DQL 和
  pressure；这只证明其余场景未受该根因影响，不证明完整 2.6 round-trip 支持。
- 最终 manifest 恢复完整 matrix；standalone/cluster 均渲染为
  `known_limitation + unsupported + release-gate-eligible=false`。只有包含两个
  触发 schema 的严格运行转绿，才能把它重新提升为 release gate。

## 诊断过程与修正依据

### 已删除的 e47 target tag

cluster 首次提交 `c30-index-v10-v4-rcccg` 在拉取 e47 时进入
`ImagePullBackOff`。确认 tag 已被 Harbor 清理后，没有用 mutable tag 继续执行，
而是选取并锁定 f78 manifest digest。f78 比 e47 多 9 个提交，包含 #52767 修正，
且排除了更晚的 Woodpecker #52904 变更。

### Cluster Woodpecker 边界

诊断 workflow `c30-index-v10-v4-tm2tf` 与 `c30-index-v10-v4-q6hft`
均在回滚后出现 reader temp metadata 不兼容。Pod UID 对比证明问题并非 Helm
意外重启 Woodpecker：v3.0.0 的 Woodpecker client 0.1.33 无法读取 3.0 branch
client 0.1.37+ 写入的元数据。该问题属于消息存储合同，不应阻塞 index-engine
合同，因此仅将 cluster v10/v4、v11/v4 两个专项场景切换到 Pulsar；独立
Woodpecker gates 未改变。

模板层仍保留通用保护：若存在 Woodpecker StatefulSet，patch 前切为
`OnDelete`（同时清空 `rollingUpdate`），记录排序后的 Pod `name=uid`，Helm 完成后
强制 diff。单 task 对 `OnError` 最多重试两次；main DAG 明确 `limit: 0`，避免整条
升级/回滚状态机被重复执行。`bt8gr`、`8w788`、`6shv5`、`qnvrl` 等中间执行还
用于确认 tSafe reader、StatefulSet patch 结构和 Kubernetes quota evaluation
瞬时错误的边界。

### 2.6 nested scalar index 边界

首次 none-contract 控制场景 `s2618-30-rb26-none-8bx5f` 使用完整 11-schema
matrix。target 上所有检查均通过，但回滚后 2.6 日志明确报告：

```text
Assert "!is_nested" => nested scalar sort index is not supported in 2.6
```

受影响的是两个 target-built StructArray scalar AutoIndex schema。它不是简单的
index version pin 问题：v2.6.18、2.6 branch、v3.0.0 和 3.0 branch 的默认 scalar
engine version 均为 3，但 2.6 reader 不具备 nested 格式能力。最初的派生 matrix
可帮助隔离根因，却会在 `release-gate-eligible=true` 下掩盖支持合同；PR #44 review
后已删除派生 loader/matrix，恢复完整失败用例并将场景降级为 known limitation。

## 改动范围及对其他 path 的影响

- `cluster-3-0-index-v10-v4-upgrade-rollback` 和
  `cluster-3-0-index-v11-v4-upgrade-rollback`：deploy profile 从 Woodpecker 1CU
  改为 Pulsar 1CU；合同和 schema 验证不变。
- 所有 cluster upgrade/rollback workflow：获得 Woodpecker Pod 不重建保护；
  retry 只作用于 `deploy-milvus`、`wait-milvus-ready`、`patch-milvus-image` 和
  `patch-milvus-config`。`run-brick`、optional brick 和 pressure 等业务/DML 模板
  没有 retry。
- standalone 与 cluster 的通用
  `2.6.18 -> 3.0 -> 2.6 latest` path：统一恢复完整 11-schema matrix，并降级为
  #52893 known limitation，不参与 release gate。
- 2.6 target-only feature paths：继续使用完整 11-schema `schema_matrix_2_6.yaml`；
  这两个被排除的 nested scalar schema 仍在 upgrade/target-only 阶段覆盖。
- 其他 3.0、Vortex、LoonFFI、JSON Shredding、Woodpecker 专项 path 的 manifest
  合同和 matrix 未改变。
- 所有已注册升级/回滚场景仍由 manifest 统一默认 `milvus_log_level: debug`。

本轮实际 E2E 运行 standalone 2.6 严格路径并复现 #52893；cluster 2.6 已保持
相同完整合同和 non-eligible classification，但未在本轮再次执行。

## 本地回归与静态验证

在初始 `af1d259` 上的 pytest 和 E2E 结果如上；PR #44 review 发现当时只执行了
`ruff check`，遗漏 CI 的 `ruff format --check`，后者准确发现
`test_argo_template.py` 未格式化。review 修正后的最终验证为：

- 完整 pytest：`603 passed in 50.06s`。测试数从 605 减少 2，是因为派生 matrix
  loader 及其两个专用测试被删除。
- manifest/schema/render/Argo template 相关测试：`298 passed`。
- CI 固定版本 `ruff==0.15.22`：完整 23 文件 check 通过，format check 显示
  `23 files already formatted`。
- `argo lint --offline argo`：全目录通过。
- 代表性 manifest 渲染：standalone/cluster 2.6 均为
  `known_limitation + unsupported + release-gate-eligible=false + full 2.6 matrix +
  debug`。
- Retry 结构断言：只有 `deploy-milvus`、`wait-milvus-ready`、
  `patch-milvus-image`、`patch-milvus-config` 四个模板含 retry；不存在全局
  `templateDefaults` retry。
- `git diff --check`：通过。

## 清理与后续建议

最终复核四条 index-engine gate 和一条诊断 subset workflow 均为 Succeeded；
对应 Milvus CR 与实例标签资源数量为 0。完整 2.6 tracker 的失败是 #52893 的预期
证据，不计入 release gate；其保留环境也已在诊断后清理。

后续发布 3.0.2、3.0.5、3.0.8 或跨 minor 的 3.1.0 时，应在 manifest 中新增/更新
不可变 baseline、target、rollback alias，并按 capability + exact image + topology
登记 qualification。只有 baseline/rollback 已通过相同能力证据时，才把对应
v10/v4 或 v11/v4 合同从 `target_only` 改为 `round_trip`；不需要修改合同实现。
跨 minor 仍需额外复核 schema/storage format、SDK/config 和模板分支约束。
