# Index Contract E2E Matrix 改动与执行报告

## 结论

PR #43 合入后的优先验证矩阵已全部完成。standalone/cluster 的 v10/v4、
v11/v4 `target_only` 合同，以及 standalone `2.6.18 -> 3.0 branch ->
2.6 branch` 的 `none` 合同均通过最终 gate；5 条正式 workflow 的压力门禁
`failed=0`，Milvus CR 和实例标签资源残留均为 0。

执行过程中发现并修正了三类测试基础设施/合同边界问题：

1. cluster Helm patch 可能重建 Woodpecker Pod；模板现在用 `OnDelete` 并比较
   patch 前后的 Pod `name=uid`，同时只对单节点基础设施错误重试，禁止整个 DAG
   重跑。
2. v3.0.0 携带的 Woodpecker client 无法读取较新 3.0 branch client 留下的
   reader 临时元数据。v10/v4、v11/v4 是 index-engine 专项合同，cluster 路径改用
   Pulsar 1CU 隔离消息存储兼容性；Woodpecker 仍由独立 gate 覆盖。
3. 3.0 target 创建的两个 StructArray nested scalar AutoIndex 格式无法由 2.6
   reader 加载。新增派生的 `2.6_round_trip` matrix，仅从跨版本 round-trip 排除
   这两个格式；完整 2.6 matrix 和 target-only 覆盖保持不变。

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

时间均为 UTC。`failed_all` 包含压力进程记录的预期/允许失败样本；最终严格门禁
使用 `failed`，5 条路径均为 0。

| 场景 | Workflow | 合同 / topology | 结果与时间 | 压力 `failed / failed_all / passed / total` |
| --- | --- | --- | --- | --- |
| standalone v10/v4 | `milvus-standalone-3-0-upgrade-rollback-7g224` | `target_only`, Rocksmq | Succeeded `63/63`, 2026-08-27 14:06:25–15:08:24 | `0 / 19 / 192 / 211` |
| standalone v11/v4 | `milvus-standalone-3-0-upgrade-rollback-frj7m` | `target_only`, Rocksmq | Succeeded `63/63`, 2026-08-27 16:08:05–17:13:24 | `0 / 21 / 210 / 231` |
| cluster v10/v4 | `c30-index-v10-v4-6bp49` | `target_only`, Pulsar 1CU | Succeeded `63/63`, 2026-08-28 06:08:37–07:15:09 | `0 / 19 / 215 / 234` |
| cluster v11/v4 | `c30-index-v11-v4-w7tm4` | `target_only`, Pulsar 1CU | Succeeded `63/63`, 2026-08-28 07:16:32–08:20:56 | `0 / 18 / 217 / 235` |
| standalone 2.6 round-trip | `s2618-30-rb26-safe-7pplv` | `none`, Rocksmq, 9-schema derived matrix | Succeeded `58/58`, 2026-08-28 09:20:10–10:19:53 | `0 / 19 / 190 / 209` |

前两条 standalone workflow 使用原始 e47 target 与 PR #43 revision。后两条
cluster workflow 使用 digest-pinned f78 target；其运行时测试代码 revision 仍固定为
PR #43，但提交的 WorkflowSpec 包含本轮 cluster 模板保护。2.6 round-trip 使用
f78 target 和 `af1d259` revision。

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

### 2.6 none-contract round-trip

- `schema_matrix_2_6_round_trip.yaml` 从完整 2.6 matrix 派生 9 个 schema；
  target checkpoint 为 `existing=9, new=9, reload=18, failures=0`。
- `none` 合同不执行 target-only drop；phase-new collections 在回滚后仍存在。
- 回滚到 2.6 branch 后可服务性探测在 11 秒内通过；index/schema 校验通过。
- rollback phase DML/DQL checkpoint 为
  `existing=9, new=9, reload=18, failures=0`；9 个 carried collections 全部参与，
  `inserted=9000, upserted=8000, deleted=900`，未发生 target-only skip。
- phase 累计 reload 27 次、search 62 次、scalar index query 80 次，最终压力门禁通过。

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
engine version 均为 3，但 2.6 reader 不具备 nested 格式能力。因此采用显式派生
matrix，而不是降低整条测试路径覆盖或隐藏回滚失败。

loader 新增 `extends` 与 `exclude_schemas`，并对继承环、版本不一致、重复/空值、
未知 exclusion、derived matrix 同时定义 schemas 等情况 fail closed。实现前的新增
测试先稳定复现 2 个失败，随后修复至通过。

## 改动范围及对其他 path 的影响

- `cluster-3-0-index-v10-v4-upgrade-rollback` 和
  `cluster-3-0-index-v11-v4-upgrade-rollback`：deploy profile 从 Woodpecker 1CU
  改为 Pulsar 1CU；合同和 schema 验证不变。
- 所有 cluster upgrade/rollback workflow：获得 task 级基础设施错误重试和
  Woodpecker Pod 不重建保护；业务断言失败不靠整 DAG 重试掩盖。
- standalone 与 cluster 的通用
  `2.6.18 -> 3.0 -> 2.6 latest` none-contract path：统一使用 9-schema
  `2.6_round_trip` matrix，保持两种 topology 的合同定义一致。
- 2.6 target-only feature paths：继续使用完整 11-schema `schema_matrix_2_6.yaml`；
  这两个被排除的 nested scalar schema 仍在 upgrade/target-only 阶段覆盖。
- 其他 3.0、Vortex、LoonFFI、JSON Shredding、Woodpecker 专项 path 的 manifest
  合同和 matrix 未改变。
- 所有已注册升级/回滚场景仍由 manifest 统一默认 `milvus_log_level: debug`。

本轮实际 E2E 运行 standalone 2.6 none-contract 作为控制路径；cluster 2.6
对应场景已同步引用相同派生 matrix，并由 manifest/renderer 测试验证，但不把它
误记为本轮已执行的第 6 条 E2E。

## 本地回归与静态验证

在 `af1d259` 上完成以下验证：

- 完整 pytest：`605 passed in 47.53s`（报告落盘后的最终重跑）。
- manifest/schema/render 定向测试：`180 passed`；新增/相关 manifest 场景测试
  `25 passed`。
- Ruff（本轮 Python 文件）：通过。
- `argo lint`：cluster、standalone 2.6、standalone 3.0 三个模板全部通过。
- 代表性 manifest 渲染断言：cluster v10/v11 均为
  `target_only + Pulsar 1CU + debug`；standalone 2.6 为
  `none + 2.6_round_trip + debug`，通过。
- `git diff --check`：通过。

仓库全量 Ruff 仍包含与本轮无关的历史问题，因此验收使用本轮改动 Python 文件的
scoped Ruff；没有修改或掩盖历史失败。

## 清理与后续建议

最终复核 5 条成功 workflow 均为 Succeeded；对应 Milvus CR 数量为 0、
`app.kubernetes.io/instance=<workflow>` 标签资源数量为 0。最后一条 2.6 workflow
的 onExit 也为 Succeeded，workflow 标签资源为 0。

后续发布 3.0.2、3.0.5、3.0.8 或跨 minor 的 3.1.0 时，应在 manifest 中新增/更新
不可变 baseline、target、rollback alias，并按 capability + exact image + topology
登记 qualification。只有 baseline/rollback 已通过相同能力证据时，才把对应
v10/v4 或 v11/v4 合同从 `target_only` 改为 `round_trip`；不需要修改合同实现。
跨 minor 仍需额外复核 schema/storage format、SDK/config 和模板分支约束。
