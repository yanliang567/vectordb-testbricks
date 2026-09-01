# Milvus 3.0.1 升级/回滚发布验证报告

## 结论

本轮在 QA 4am 集群使用 2026-08-29 最新 3.0 分支镜像，对 manifest 中 20 条正式升级/回滚 gate、3 条 known limitation tracker 和 1 条负向合同进行验证。执行期间严格限制 standalone 最多 3 并发、cluster 最多 2 并发，所有 Milvus 阶段默认使用 `debug` 日志级别。

20 条正式 gate 的最终结果为 18 条通过、2 条失败。失败的两条均为 `2.6.18 -> 3.0.1 -> 2.6 latest` target-only 路径，并稳定命中同一个 #52893 nested scalar index 回滚问题。因此：

- 3.0.1 同版本 round-trip、3.0.0 到 3.0.1、v10/v4、v11/v4、Vortex、JSON Shredding、LoonFFI 和 Woodpecker 0.1.38 路径未发现新的 Milvus 产品回归。
- standalone 和 cluster 的 `2.6.18 -> 3.0.1 -> 2.6 latest` target-only 正式 gate 都稳定命中 nested scalar index 回滚问题 #52893。该失败保留为真实 release blocker，没有删除 schema 或放宽断言。
- 严格全矩阵 standalone tracker 在 rollback 后 11 个集合中 10 个可服务，只有 `struct_array_numeric_autoindex_rollback_safe` 因 channel not serviceable 失败，与 #52893/#52768 边界一致，不是随机测试噪音。

按当前 manifest 的 supported release contract，**不建议把该 candidate 判定为完整 upgrade/rollback release-ready**：#52893 必须修复并重跑两条 2.6 rollback gate，或者由产品层显式调整支持合同；不能用其余 18 条绿色结果抵消。如果 3.0.1 发布决策只看 3.0.x 内部 round-trip 和 target-only 新能力，则本轮对应路径全部通过，但报告不据此改写现有 release gate。

## 固定输入

- 测试时段：2026-08-29，Asia/Shanghai；最后一个 Workflow 于 19:05:39 结束
- 初始代码基线：`origin/main@df4b74257130884d480385b3ef2760dfa9a76ad0`
- 完整 20-gate 业务矩阵最终 E2E revision：`8400590bc20df4754fdf7237155d549654acb4fb`
- 最终 post-E2E 功能 revision：`a2be2665f23d4c657587d07cf61317103a8cc34b`；从 `8400590` 到该 revision 的证据边界在“测试框架问题与已实现修复”和“post-E2E 当前 head 代表性实集群验证”中逐项记录；不宣称完整 20 条业务 Workflow 已在该 revision 上重跑
- 3.0/3.0.1 candidate：`harbor.milvus.io/milvusdb/milvus:3.0-20260829-257a535b@sha256:d3a0d1d64368139ab59a28989392bcffdefaaa0f724596b870ed7c0b16d15c20`
- 2.6 latest rollback：`harbor.milvus.io/milvusdb/milvus:2.6-20260829-3b859656@sha256:989e085e45c44f513387f361c0c6b326a434a0828964798a459728460ebe04b6`
- 2.6.18 baseline：`harbor.milvus.io/milvusdb/milvus:v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6`
- 3.0.0 baseline：`harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
- Woodpecker：`harbor.milvus.io/milvusdb/woodpecker:v0.1.38@sha256:bdea08758377fea309c18087334c63d20e26ba0940a4d63369bf7794f5f2060e`
- 镜像检查：上述 Milvus/Woodpecker 引用均为 `tag@manifest-list-digest`，Harbor 可拉取且包含测试所需架构。
- 压力模块：`search_pressure query_pressure query_iterator_scan count_pressure upsert_pressure delete_pressure mixed_rw_pressure`
- 并发上限：standalone 3，cluster 2；执行记录中没有超过上限。
- 除 `milvus-standalone-3-0-upgrade-rollback-v4rzd` 未显式覆盖、使用模板默认 `keep-milvus=false` 外，其余正式运行使用 `keep-milvus=true`；成功和测试框架无效轮次取证后按 Workflow UID 精确清理。最终仍保留 4 套具有独立产品失败证据的环境；两条最早 formal #52893 gate 当前只保留 Argo 历史，由 strict tracker 环境提供等价复现现场。

## 合同与验证点

正式 gate 由 `upgrade_rollback_gates.yaml` 渲染，运行时只覆盖不可变镜像、semantic version 和完整 Git SHA。预渲染的 20 条 gate 均满足：无 placeholder、`release-gate-eligible=true`、classification/support status 与 manifest 一致、Milvus log level 为 `debug`。

每条完整路径覆盖：

- base、target、rollback 实际镜像和版本断言；
- baseline seed、checkpoint、collection/entity count 和 sample PK；
- schema feature、schema evolution、persisted index reader/writer compatibility；
- upgrade/rollback 前后 phase DML/DQL、新集合和 carried collection 合同；
- release/load 后的 vector search、scalar index query 和数据可见性；
- rollout 期间持续压力、严格压力切片和 serviceability 观察窗口；
- target-only 数据不要求旧 baseline reader 读取；round-trip 数据必须在合同声明的 rollback reader 上继续可读写；
- JSON Shredding、LoonFFI、Vortex 和 Woodpecker 配置在各阶段与 manifest 合同一致。

## 正式 gate 结果

### Standalone

| # | 场景 | Workflow | 结果 | 关键结论 |
| ---: | --- | --- | --- | --- |
| 1 | `standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest` | [xrjd2](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-2-6-upgrade-rollback-xrjd2) | Failed | rollback nested scalar index channel 超时，#52893；真实 blocker；Argo 证据保留，等价 strict 环境 `5mtr6` 保留 |
| 2 | `standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | [ngdd4](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-ngdd4) | Succeeded 63/63 | 3.0.0 -> 3.0.1 -> 3.0.0 core round-trip 通过 |
| 3 | `standalone-3-0-index-v10-v4-upgrade-rollback` | [v4rzd](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-v4rzd) | Succeeded 63/63 | v10/v4 target-only 合同通过 |
| 4 | `standalone-3-0-index-v11-v4-upgrade-rollback` | [gfqmx](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-gfqmx) | Succeeded 63/63 | v11/v4 target-only 合同通过 |
| 5 | `standalone-3-0-1-vortex-self-compat-upgrade-rollback` | [n4m2z](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-n4m2z) | Succeeded 63/63 | StorageV3/Vortex self round-trip 通过 |
| 6 | `standalone-3-0-0-to-3-0-1-vortex-enable-rollback` | [lhk8j](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-lhk8j) | Succeeded 63/63 | 3.0.0 legacy reader 到 3.0.1 Vortex writer，再回 3.0.1 通过 |
| 7 | `standalone-3-0-1-json-shredding-vortex-rollback` | [ts4fw](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-ts4fw) | Succeeded 63/68 | JSON Shredding + Vortex 配置 round-trip 通过；进度包含基础设施 retry 子节点 |
| 8 | `standalone-3-0-1-loon-ffi-rollback` | [tkkxh](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-tkkxh) | Succeeded 58/58 | LoonFFI enable/disable 配置 round-trip 通过 |
| 9 | `standalone-3-0-1-vortex-disable-rollback` | [kdbqt](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-kdbqt) | Succeeded 63/63 | 无 Loon rollback-safe 2.6 matrix，通过 |
| 10 | `standalone-3-0-1-vortex-disable-keep-loon-rollback` | [sxph8](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-sxph8) | Succeeded 63/63 | rollback 保留 Loon，StorageV3 TEXT matrix，通过 |

### Cluster

| # | 场景 | Workflow | 结果 | 关键结论 |
| ---: | --- | --- | --- | --- |
| 1 | `cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest` | [c26to-xt8wc](https://argo-workflows.zilliz.cc/workflows/qa/c26to-xt8wc) | Failed | rollback nested scalar index channel 超时，#52893；真实 blocker；Argo 证据保留，等价 strict 环境 `hmtfx` 保留 |
| 2 | `cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | [stbr9](https://argo-workflows.zilliz.cc/workflows/qa/milvus-cluster-upgrade-rollback-stbr9) | Succeeded 63/63 | Woodpecker v0.1.38 下 core round-trip 通过 |
| 3 | `cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline` | [c30json-48w6t](https://argo-workflows.zilliz.cc/workflows/qa/c30json-48w6t) | Succeeded 63/63 | #52341 约束合同在 3.0.1 candidate 上通过 |
| 4 | `cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline` | [c30-2cu-ha-s94cc](https://argo-workflows.zilliz.cc/workflows/qa/c30-2cu-ha-s94cc) | Succeeded 58/58 | 2CU HA 和 4 个 Woodpecker Pod digest 通过 |
| 5 | `cluster-3-0-index-v10-v4-upgrade-rollback` | [qsp2d](https://argo-workflows.zilliz.cc/workflows/qa/c30-index-v10-v4-qsp2d) | Succeeded 63/63 | v10/v4 target-only 合同通过 |
| 6 | `cluster-3-0-index-v11-v4-upgrade-rollback` | [88stc](https://argo-workflows.zilliz.cc/workflows/qa/c30-index-v11-v4-88stc) | Succeeded 63/63 | v11/v4 target-only 合同通过 |
| 7 | `cluster-3-0-1-vortex-self-compat-upgrade-rollback` | [cj867](https://argo-workflows.zilliz.cc/workflows/qa/c301vs-cj867) | Succeeded 63/63 | 分布式 StorageV3/Vortex self round-trip 通过 |
| 8 | `cluster-3-0-0-to-3-0-1-vortex-enable-rollback` | [t8gvb](https://argo-workflows.zilliz.cc/workflows/qa/c301ve-t8gvb) | Succeeded 63/63 | 分布式 Vortex enable 合同通过 |
| 9 | `cluster-3-0-1-json-shredding-vortex-rollback` | [qw9bg](https://argo-workflows.zilliz.cc/workflows/qa/c301jv-qw9bg) | Succeeded 63/63 | JSON Shredding + Vortex 分布式 round-trip 通过 |
| 10 | `cluster-3-0-1-loon-ffi-rollback` | [tfmf6](https://argo-workflows.zilliz.cc/workflows/qa/c301lf-tfmf6) | Succeeded 58/58 | LoonFFI enable/disable 分布式 round-trip 通过 |

## Known limitation 与负向控制

| 分类 | 场景 | Workflow | 结果 | 是否计入 gate |
| --- | --- | --- | --- | --- |
| known limitation | standalone strict full `2.6.18 -> 3.0.1 -> 2.6 latest` | [5mtr6](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-2-6-upgrade-rollback-5mtr6) | Failed as expected；10/11 可服务，nested scalar AutoIndex 失败 | 否；环境保留 |
| known limitation | cluster strict full `2.6.18 -> 3.0.1 -> 2.6 latest` | [hmtfx](https://argo-workflows.zilliz.cc/workflows/qa/c26rb-hmtfx) | Failed as expected；48 attempts / 908.711s，前 8 个 collection count=5000，随后 nested numeric AutoIndex channel 永久不可服务，#52893 | 否；环境保留 |
| known limitation | standalone `3.0.0 -> 3.0.1 JSON Shredding -> 3.0.0` | [wtvrv](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-3-0-upgrade-rollback-wtvrv) | Failed as expected；target 写入的 struct-array VARCHAR HYBRID AutoIndex collection 在 v3.0.0 reload 超时，#52768 | 否；环境保留 |
| negative | standalone `3.0 Loon/Vortex -> 2.6` unsafe rollback | [z88hx](https://argo-workflows.zilliz.cc/workflows/qa/milvus-standalone-2-6-upgrade-rollback-z88hx) | Failed as expected；11/11 baseline collection count=0、sample PK 缺失、`recovered=false` | 否；环境保留 |
| frozen candidate | standalone/cluster Vortex 0.75 historical candidate | 未执行 | Not applicable；固定历史预发布镜像，不属于 3.0.1 release gate | 否 |

## 失败分类与修复闭环

### Milvus 产品限制

- standalone/cluster 两条 2.6 target-only 正式 gate 均在 rollback reader 上命中 #52893；保留原始用例、失败断言和环境。
- strict full standalone tracker 进一步确认边界集中在 nested scalar AutoIndex collection；其他 10 个 collection 可服务。
- strict full cluster tracker 在 48 次、908.711 秒轮询中先确认前 8 个 collection 均为 5000 rows，随后固定失败于 `struct_array_numeric_autoindex_rollback_safe`；剩余 collection 因 fail-fast 未继续评估。失败 channel 为 `by-dev-rootcoord-dml_15_468712908498405718v0`，与正式 cluster target-only gate 和 standalone tracker 的 #52893 症状一致。
- 不通过排除该 schema、把 supported gate 降为绿色或对 channel not serviceable 放宽断言来掩盖结果。

### 测试框架问题与已实现修复

| Commit | 问题 | 修复与验证 |
| --- | --- | --- |
| `3be0d83` | schema evolution visibility 使用弱一致性造成假失败 | 改为强一致性验证 |
| `32c4cc8` | absent text token 可能被 analyzer 处理为已有 token | 使用 analyzer-safe probe |
| `4c567c1` | function collection 不适合通用 pressure probe | 明确跳过不可用 function probe |
| `e6073f0` | TEXT feature count 未限定 checkpoint PK | 将验证范围限定到 checkpoint 数据 |
| `63db331` | standalone 同镜像配置 rollout 不重启 Pod | 配置变更时显式 recycle standalone Milvus Pod |
| `0ca7637` | generic Woodpecker 1CU/2CU profile 使用可变 master | 固定到已验证 v0.1.38 digest |
| `c1a0170` | 无 Loon 的 Vortex-disable gate 使用 StorageV3 TEXT forward matrix | 改为 rollback-safe 2.6 matrix；manifest validation 阻止同类非法合同 |
| `eef8eb7` | read-only index compatibility 在持续写压力下重复 flush，可等待 900 秒/collection | `rebuild_index=false` 时跳过 flush，并增加 collection progress 日志 |
| `8400590` | phase DML/DQL 的 best-effort flush 在持续写压力下无 deadline | 增加 10 秒客户端 deadline；禁止 TypeError 后退回无 timeout；visibility/search/query/reload 仍为最终断言；完整回归 624 passed |
| `240fff3` | standalone Pod delete 使用 kubectl 默认超长同步等待 | 改为异步删除，由后续新 UID/Ready 收敛检查负责；post-E2E 静态验证 |
| `b946b5e` | schema evolution flush/load 和 phase best-effort load 仍可能无界等待 | 增加客户端 timeout 且禁止无 timeout fallback；post-E2E 静态验证 |
| `282ad18` | standalone RBAC 缺少 Pod delete、Ready 轮询聚合上界接近 30 分钟、新集合 load 超时后立即验证可能假失败 | 增加最小 `pods/delete` 权限；用统一 600 秒 wall-clock deadline 和非阻塞 Ready 查询；新集合复用有界 visibility retry；完整回归 628 passed |
| `25582ec` | standalone Pod recycle 的旧 UID snapshot 和异步 delete 请求仍可能因 apiserver/网络异常无界等待 | 两个 standalone 模板的 image/config 四条路径均增加 `--request-timeout=5s`，并补充回归断言；完整回归 628 passed |
| `cd3b126` | milvus-dev-cli 标准 7 位 short-SHA 日构建 tag 被候选版本解析器误拒绝 | 接受标准 short-SHA candidate tag，保留版本下限断言；完整回归 631 passed |
| `ca321c3` | `fouram:2.1` 内 kubectl 使用 `--request-timeout=5s` 时会丢失 in-cluster 配置并回退到 `localhost:8080` | 两个 standalone 模板的 image/config 四条路径改用外层 `/usr/bin/timeout 5s kubectl ...`，保留非阻塞 delete 和 600 秒收敛上界；完整回归 631 passed |
| `a2be266` | visibility retry 只在 validator 返回后检查 120 秒 deadline，count/PK/upsert/deleted-PK query 未设 PyMilvus timeout，单次 RPC 卡住可绕过总上界 | `_wait_for_validation` 向 existing/new callback 传入剩余 RPC 预算，每个 query 前重新计算并透传 `timeout`；sleep 也限制在剩余预算内；新增 existing/new 两条路径和 common validator 回归，完整回归 635 passed |

截至 `8400590` 的测试框架修复都从固定 SHA 推送后完成业务 Workflow 重跑；旧 SHA 卡住的临时 workflow 被显式 terminate，不计为产品结果。`240fff3`、`b946b5e`、`282ad18`、`25582ec`、`cd3b126`、`ca321c3` 和 `a2be266` 是最终 20-gate 业务矩阵之后的 review 修复，本报告不声称 20 条 Workflow 已在这些 revision 上重跑。它们的基础证据范围是定向回归、完整 pytest、CI 范围 Ruff、Argo lint 和 shell syntax；当前 head 的代表性 live E2E 另见下节，不外推为全矩阵证据。

### post-E2E 当前 head 代表性实集群验证

2026-09-01 使用功能 revision `a2be2665f23d4c657587d07cf61317103a8cc34b` 在 4am 提交一条隔离的 current-head standalone E2E：[Workflow `pr46-a2be-st-lhz5v`](https://argo-workflows.zilliz.cc/workflows/qa/pr46-a2be-st-lhz5v)，场景为 `standalone-3-0-1-vortex-disable-rollback`。运行由隔离 WorkflowTemplate `milvus-standalone-3-0-upgrade-rollback-pr46-a2be266` 启动，没有覆盖共享 WorkflowTemplate。base、target 和 rollback 都使用本报告固定的同一 `3.0.1` `tag@digest`；target 打开 LoonFFI/Vortex，rollback 关闭两者，所有阶段使用 `debug` 日志，`keep-milvus=false`。

- 终态：`Succeeded 63/63`，2026-09-01 11:09:25Z 到 12:14:27Z，共 3902 秒；`gate-final-status` 通过，无 Failed/Error 节点。这只是当前 head 的代表性 standalone 证据，不代表 20 条业务矩阵重跑。
- same-image Pod recycle：`patch-upgrade` 14 秒、`patch-rollback` 23 秒。两次日志都显示外层 `timeout 5s kubectl get`、`timeout 5s kubectl delete --wait=false` 执行，并分别收敛到新 Ready UID `668ac403-3b71-49d9-a245-725be735461e` 和 `e86a6a72-ff3e-41ac-88a2-1830b9678295`。
- upgrade phase DML/DQL：JSON 结果 `passed`，无 failure；11 个 existing + 11 个 new collection，22 个 reload 零失败，66 次 search、88 次 scalar-index query。`scalar_autoindex_formats_rollback_safe` existing visibility 实际用了 2 次 attempt 后收敛，证明新的有界 retry 不只是单测覆盖。
- rollback phase DML/DQL：JSON 结果 `passed`，无 failure；11 个 existing + 11 个 new + 11 个 carried collection，upgrade phase checkpoint 验证通过，33 个 reload 零失败，99 次 search、132 次 scalar-index query。
- 服务可用性与压力：upgrade/rollback/forward rollback serviceability 都在首次 attempt 通过；228 个 pressure sample 中 209 个直接通过，19 个失败都落在声明的 rollout/reload maintenance window 并按合同排除，最终无 gated failure/warning。steady state 共 977106/977106 次操作成功；upgrade rollout 有 4 次 delete 短暂失败，rollback rollout 有 4 次 query 短暂失败，与当前接受升级/回滚 maintenance window 内短暂不可用的合同一致，在此显式记录而不隐藏。
- 清理：onExit 完成后 Milvus CR 和依赖均不存在。已额外精确删除卡住 `pvc-protection` 的 63 个已结束 Workflow Pod、2Gi checkpoint PVC 和隔离 WorkflowTemplate；`qa`/`qa-milvus` 中按 Workflow 名称、label 和前缀复核无残留。Argo Workflow 对象、节点状态和 S3 artifacts 保留。

### QA 控制面无效轮次

多次出现 Kubernetes admission `resource quota evaluation timed out`，但 `qa` namespace 没有 ResourceQuota，API `/readyz?verbose` 恢复后通过。受影响 workflow 包括 `ff4pl` onExit、`bzxnh` create schema、`2bz2t` validation、`58f9v` patch 和 `792kz` create schema。这些轮次在业务 brick 未得到有效结论时归类为基础设施无效轮次，并以完整新 workflow 重跑。

`milvus-standalone-3-0-upgrade-rollback-76hfk` 是另一条 operator 无效轮次：Milvus CR 一直为 `Pending`，只创建了 MinIO，operator 反复报告缺少 `...-etcd-old-sts` ControllerRevision，etcd StatefulSet 和 Milvus Pod 从未创建。该半成品环境取证后清理，以新 Workflow `kdbqt` 完整重跑并通过 63/63。

`argo retry` 不适用于本模板的失败恢复：它使用 `system:serviceaccount:argo:argo`，缺少 volumeClaimTemplate PVC 的 `get` 权限，还会把动态 `keep-milvus=true` patch 恢复为默认 false。因此没有用 `argo retry` 生成正式结果，也没有给非幂等 seed/insert/upsert/delete 增加全局 retry。

## 环境保留与清理

以下 4 套产品失败环境仍实际存在，供 Milvus debug 日志、metadata 和 channel 状态继续排查：

- `milvus-standalone-2-6-upgrade-rollback-5mtr6`
- `milvus-standalone-3-0-upgrade-rollback-wtvrv`
- `milvus-standalone-2-6-upgrade-rollback-z88hx`
- `c26rb-hmtfx`

两条最早正式 #52893 gate `milvus-standalone-2-6-upgrade-rollback-xrjd2` 和 `c26to-xt8wc` 的 `keep-milvus=true` 参数与 Argo 历史仍在，但最终核对时已没有对应 Kubernetes 资源，不能声称环境保留。`5mtr6` 和 `hmtfx` 分别保留了 standalone/cluster 的同矩阵等价失败现场。

成功环境和测试框架无效环境在取证后，使用 Workflow UID / `app.kubernetes.io/instance` 标签精确删除 Milvus CR、Service、Deployment、StatefulSet、PVC、ConfigMap 和 Secret。该 Kubernetes 环境删除不可恢复；Argo workflow 历史、节点状态和日志证据保留。

## 静态验证

- `PYTHONPATH=. python3 -m pytest milvus_client/tests -q`：`635 passed in 50.97s`（post-E2E review 功能 revision `a2be266`）
- 修改文件 Ruff check：通过
- 修改文件 Ruff format check：通过
- `git diff --check`：通过
- 20 条 gate 预渲染：通过，无 placeholder，release metadata 和 `debug` log level 正确
- Argo lint：通过
- full-repo Ruff 存在与本轮无关的历史错误，因此只报告 CI 管理范围和本轮修改文件，不宣称全仓 Ruff clean

## 优化建议

1. 在 suite runner 中实现两个独立 semaphore（standalone=3、cluster=2），从 manifest 自动排队、提交、回收并生成结果表，避免人工维护并发槽位。
2. 在提交前同时验证 Git revision 已存在于 remote、Milvus/Woodpecker `tag@digest` 可拉取且为所需架构，防止 workflow 启动后才发现不可重现输入。
3. 为 release alias（例如后续 3.0.2/3.0.5）增加一次显式晋级操作：更新 manifest image catalog 和 semantic version；合同场景本身继续复用 target-only/round-trip 定义。
4. 把 phase validator 的每 collection 开始/结束、flush timeout、visibility attempts 和 reload duration 作为结构化 progress 输出；并为整个 brick 增加全局 deadline，避免“per RPC timeout × collection 数”放大总耗时。
5. pressure daemon 当前每轮产生约 170–250 个 ConfigMap，使 onExit/清理多耗时 2–3 分钟。建议聚合为单个 append-only/分片结果对象或 artifact，并只保留异常 sample 的独立记录。
6. 对配置 rollout 使用显式“期望配置 hash / Pod UID 变化 / runtime config 收敛”等待，而不是依赖 assertion retry 产生失败 child Pod 噪音。
7. 保持 retry scope：只允许 deploy/wait/patch 和严格只读 brick 使用 `OnError`；任何 seed、schema feature、phase DML 和其他非幂等写入不自动重放。
8. 为失败环境增加带原因的 retention label 和可配置 TTL；release blocker 默认保留，基础设施无效/测试框架失败在取证后自动精确清理。
9. 从 Argo node/artifact 自动生成本报告中的 immutable input、合同、结果、retry、保留环境和清理状态，减少人工转录错误。
