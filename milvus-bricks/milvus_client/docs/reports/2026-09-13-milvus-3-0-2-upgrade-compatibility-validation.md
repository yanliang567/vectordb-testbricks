# Milvus 3.0.2 RC 升级/兼容测试与分析报告

日期：2026-09-13  
环境：QA 4am Kubernetes / Argo Workflows

## 结论

**NO-GO：不建议以当前 candidate 发布 3.0.2。**

正式功能 gate 为 **19/20 PASS**：standalone 10/10，cluster 9/10。唯一正式失败是 cluster `v2.6.20 -> RC -> v2.6.20` 的 rollback pressure：一个 delete slice 8/8 请求因 `node not found` 失败；该窗口内 v2.6.20 StreamingNode 因 duplicate-field 断言累计崩溃 5 次。独立 strict cluster 又复现 6 次相同崩溃及 rollout 后 372 秒 steady-state 失败，因而不能归为测试噪声。

另有 candidate debug count 空结果越界，在两个独立 cluster workflow 的 QueryNode/StreamingNode 上复现 `exitCode=134`；虽然 v3.0.1 已有相同代码、不是本 RC 新引入，候选镜像在合法 count 压力下仍可被 debug 日志路径杀死。v2.6.20 RTREE Strong visibility 延迟是确认的 baseline 产品缺陷，测试等待已按原有 120 秒 deadline 修复，不构成本 RC 数据兼容失败。

发布前至少应：修复或明确阻止 write-before-materialization 打开后回滚到 v2.6.20 的路径，并重跑两条 cluster 2.6 场景；修复 `retrieve.go:63` 空 slice 越界并在 debug 压力下复测。不能以最终 Ready 或 steady-state 后续恢复代替这两项修复。

## 固定输入

本轮开始时查询 3.0 分支最新可部署多架构镜像，并在整个测试期间固定 tag 与 manifest digest，不跟随浮动 latest：

| 角色 | 语义版本 | 固定镜像 |
| --- | --- | --- |
| 2.6 base / rollback | `2.6.20` | `harbor.milvus.io/milvusdb/milvus:v2.6.20@sha256:e5420c36bac36a605c1ea6e04114f5d7923f114e87bbb774bd0669b9f3e41d27` |
| 3.0 base / rollback | `3.0.1` | `harbor.milvus.io/milvusdb/milvus:v3.0.1@sha256:7984f52e02fc14df553615acbf3ecf3cc131e7ce77c79e7ee23c63652919cae8` |
| 3.0.2 release candidate | `3.0.2` | `harbor.milvus.io/milvusdb/milvus:3.0-20260911-c4412246@sha256:688f10aaecc296a4ec96b973207b3e88d8451e6d916951e18da4d34ccb82b2e0` |

candidate 的 API runtime version 为 `3.0-20260911-c441224651`。正式 v3.0.1 tag 的 API runtime version 为 `3.0-20260902-658cbd1689`；上游 `v3.0.1` git tag 指向同一个 `658cbd1689...` commit。

测试代码与修复边界：

| Revision | 用途 |
| --- | --- |
| `b9a9b0ebd807453fab7d91eabbdbdb35c33a9b72` | 初始 20 条正式 gate；包含 digest-pinned release branch-build 版本校验修复 |
| `111bac5e370230d87cbba127353dcbeae1db0c62` | 修复 collection reload 窗口的 `delegator closed during wait tsafe` 压力归因；用于相关重跑和后续场景 |
| `e70e63410e23d5b5c4d2c83a8de5202c43875be9` | 为 phase scalar-index probe 增加有界可见性等待；用于两种 topology 的 2.6 gate 重跑及后续观察项 |
| `b6b9975627d3b363837971c88d1b1a32703ef6f4` | 修复 rollout 节点轮换的 pressure 归因；用于 cluster 2.6 最终重跑 |

所有 workflow 使用 `pymilvus==3.0.1`，Milvus 全阶段为 debug 日志。正式 gate 的共同强校验包括固定 image/version precheck、schema/data checkpoint、persisted index reader、phase insert/upsert/delete、新 collection、release/load 复验、持续压力、rollout serviceability 与 final status。

## 范围与并发

- 正式 gate：20 个唯一场景，standalone 10 个、cluster 10 个。
- 独立观察项：standalone/cluster strict full `2.6 -> 3.0.2 RC -> 2.6`、standalone JSON Shredding tracker、standalone LoonFFI/Vortex 到 2.6 negative control。
- 历史 `standalone/cluster-3-0-vortex-candidate-upgrade-rollback` 固定旧预发布镜像，与本轮 RC 不同，标记为 not applicable，未执行。
- standalone 实际执行并发不超过 4；cluster 实际执行并发不超过 3。

cluster 调度曾出现一次 submit 后状态尚不可见的竞态：`r302-cwp-qngq7` 在第二次 submit 后立即被 suspend 于 `0/1`，未创建子节点或 Milvus Pod；前一任务释放 slot 后才 resume。该窗口中实际执行的 cluster 始终为 3，但说明客户端轮询不是原子 semaphore，已列入优化项。

## 正式 gate 结果

| # | Topology | 路径 / 专项 | 结果与 Argo 证据 |
| ---: | --- | --- | --- |
| 1 | standalone | v3.0.1 -> RC -> v3.0.1 core | PASS — [`r302-score-dz5mr`](https://argo-workflows.zilliz.cc/workflows/qa/r302-score-dz5mr) |
| 2 | standalone | index v10/v4 | PASS — [`r302-si10-d6vx6`](https://argo-workflows.zilliz.cc/workflows/qa/r302-si10-d6vx6) |
| 3 | standalone | index v11/v4 | PASS — [`r302-si11-xl7pb`](https://argo-workflows.zilliz.cc/workflows/qa/r302-si11-xl7pb) |
| 4 | standalone | Vortex enable | PASS — [`r302-snext-8kds5`](https://argo-workflows.zilliz.cc/workflows/qa/r302-snext-8kds5) |
| 5 | standalone | JSON Shredding + Vortex | PASS — [`r302-snext-jcwfw`](https://argo-workflows.zilliz.cc/workflows/qa/r302-snext-jcwfw) |
| 6 | standalone | LoonFFI | PASS — [`r302-snext-6t56w`](https://argo-workflows.zilliz.cc/workflows/qa/r302-snext-6t56w) |
| 7 | standalone | Vortex self compatibility | PASS — [`r302-snext-cwpzc`](https://argo-workflows.zilliz.cc/workflows/qa/r302-snext-cwpzc) |
| 8 | standalone | Vortex disable, keep Loon | PASS — [`r302-snext-n2g7l`](https://argo-workflows.zilliz.cc/workflows/qa/r302-snext-n2g7l) |
| 9 | standalone | Vortex disable | PASS — [`r302-snext-mvrr9`](https://argo-workflows.zilliz.cc/workflows/qa/r302-snext-mvrr9) |
| 10 | standalone | v2.6.20 -> RC -> v2.6.20 target features | PASS — [`r302-s26fix-7bmpk`](https://argo-workflows.zilliz.cc/workflows/qa/r302-s26fix-7bmpk)，RTREE probe 在既有 deadline 内第 8 次可见 |
| 11 | cluster | v3.0.1 -> RC -> v3.0.1 core | PASS — [`r302-ccore-cs5rt`](https://argo-workflows.zilliz.cc/workflows/qa/r302-ccore-cs5rt) |
| 12 | cluster | index v10/v4 | PASS — [`r302-ci10-746kk`](https://argo-workflows.zilliz.cc/workflows/qa/r302-ci10-746kk) |
| 13 | cluster | index v11/v4 | PASS — [`r302-cnext-kjhhn`](https://argo-workflows.zilliz.cc/workflows/qa/r302-cnext-kjhhn) |
| 14 | cluster | Vortex enable | PASS — [`r302-cfix-tcvxv`](https://argo-workflows.zilliz.cc/workflows/qa/r302-cfix-tcvxv)，reload pressure classifier 修复后重跑 |
| 15 | cluster | JSON Shredding + Vortex | PASS — [`r302-cnext-54mzc`](https://argo-workflows.zilliz.cc/workflows/qa/r302-cnext-54mzc) |
| 16 | cluster | LoonFFI | PASS — [`r302-cnext-tjbxz`](https://argo-workflows.zilliz.cc/workflows/qa/r302-cnext-tjbxz) |
| 17 | cluster | Vortex self compatibility | PASS — [`r302-cnext-w2v8w`](https://argo-workflows.zilliz.cc/workflows/qa/r302-cnext-w2v8w) |
| 18 | cluster | JSON Shredding, v3.0.1 rollback | PASS — [`r302-cnext-tgc6b`](https://argo-workflows.zilliz.cc/workflows/qa/r302-cnext-tgc6b) |
| 19 | cluster | Woodpecker 2CU HA | PASS — [`r302-cwp-qngq7`](https://argo-workflows.zilliz.cc/workflows/qa/r302-cwp-qngq7) |
| 20 | cluster | v2.6.20 -> RC -> v2.6.20 target features | **FAIL** — [`r302-c26fix3-gnn49`](https://argo-workflows.zilliz.cc/workflows/qa/r302-c26fix3-gnn49)：功能/索引通过；pressure `255 total / 222 pass / 32 excluded / 1 fail`，失败 slice 为 rollback 中 delete 8/8 `node not found` |

矩阵中的 PASS 表示 workflow 自身的功能、checkpoint、索引、serviceability、压力与 final-report gate 通过；它不覆盖本文单独审计到的 Pod 非预期退出。第 20 条的 steady-state 压力为 `513099/513099`，但 rollback maintenance window 内发生了有副作用的 DML 失败，按 fail-closed policy 保留。旧的失败轮次仅用于问题分析，不重复计入 20 个唯一场景。

## 非 gate 观察项

| 观察项 | 结果与 Argo 证据 | 判定 |
| --- | --- | --- |
| standalone strict v2.6.20 -> RC -> v2.6.20 | [`r302-ng-s26fix-sh8g5`](https://argo-workflows.zilliz.cc/workflows/qa/r302-ng-s26fix-sh8g5) Succeeded；RTREE 第 18 次可见 | PASS，unsupported branch 按 known limitation 记录；本边界未复现 #52893 |
| cluster strict v2.6.20 -> RC -> v2.6.20 | [`r302-ng-c26-qrxzx`](https://argo-workflows.zilliz.cc/workflows/qa/r302-ng-c26-qrxzx) final report Failed | 功能/索引完成；已独立复现两类 crash，回滚后 steady-state 10 个失败 slice、65 次请求失败 |
| standalone JSON Shredding tracker | [`r302-ng-sjs-fpwjp`](https://argo-workflows.zilliz.cc/workflows/qa/r302-ng-sjs-fpwjp) Succeeded | PASS |
| LoonFFI/Vortex/JSON -> v2.6.20 negative control | [`r302-ng-neg-6zm2w`](https://argo-workflows.zilliz.cc/workflows/qa/r302-ng-neg-6zm2w) Failed | EXPECTED FAIL：回滚后 11 个 baseline collection 为 `0/5000`，checkpoint PK 缺失，unsupported guard 生效 |
| 固定旧预发布 Vortex candidate 场景 | 未执行 | N/A：镜像不是本轮 RC |

## 问题分析

### 1. v3.0.1 release tag runtime 版本解析阻塞

初始 precheck 把正式 `v3.0.1@digest` API 返回的 branch-build 字符串 `3.0-20260902-658cbd1689` 解析为 patch `0`，错误报 `SERVER_VERSION_TOO_OLD`。上游 `v3.0.1` tag 与 runtime commit 一致，镜像没有版本错误。

修复只接受以下严格组合：预期镜像为 digest-pinned release tag、release tag semantic version 与 expected version 完全一致、API runtime 为同 major/minor 的 daily branch build。未固定 digest、tag patch 不匹配等情况继续拒绝。初始 11 条受阻 workflow 不计入正式结果；3 个 standalone CR、9 个 Helm release、10 个 PVC 已精确清理，Argo 历史保留。

### 2. collection reload 压力归因缺口

原运行 `r302-cnext-xmx25` 的数据、索引、升级/回滚 serviceability 与 steady-state 压力全部通过，唯一失败切片发生在测试主动对同一 collection 执行 release/load 的 9.88 秒窗口内：

```text
delegator closed during wait tsafe: channel not available
```

同一切片内的 `collection not loaded` 已被 maintenance-window policy 排除，但上述同源错误未被识别，导致 final gate 失败。修复要求 MilvusException、受支持 DQL operation、failure collection 与 reload window collection 完全相同、时间零 padding 重叠，并同时匹配 `delegator closed during wait tsafe` 与 `channel not available`；正确性断言、跨 collection、窗口外和非 Milvus 错误仍严格失败。

### 3. rollout 节点轮换压力归因缺口

cluster 2.6 第二轮 `r302-c26fix-f9m7s` 的所有功能、索引、upgrade/rollback serviceability 和 steady-state 压力 `521978/521978` 均通过，但 final report 保留了 4 个失败 pressure slice。逐 failure 时间戳与 Argo maintenance window 对齐后，全部发生在 upgrade 或 rollback rollout 内，错误为：

```text
node not match[expectedNodeID=...][actualNodeID=...]
node not found
```

这些错误来自 Proxy/Delegator 在 Pod 轮换瞬间仍持有上一 QueryNode ID；相同 slice 内的 `channel not available` 已按 rollout maintenance 排除。分类器此前没有识别 node-ID rotation，构成测试判定阻塞。

修复仅接受完整边界：rollout label、failure 自身时间区间重叠、MilvusException、operation 为 `search/query/query_iterator/count`，并严格匹配完整 expected/actual node ID 对或 `failed to search/query delegator ... node not found`。DML、窗口外、非 MilvusException 和不完整文本继续失败。新增测试先稳定得到 2 个失败，再在修复后与 fail-closed 用例共同通过；全量回归为 `644 passed`。

### 4. v2.6.20 RTREE upsert 可见性与测试等待

两条初始正式 2.6 gate 在回滚后的同一 probe 完全一致地失败：

```text
collection: *_geometry_rtree_rollback_safe
runtime: 2.6.20
PK: 70000100
new value: POINT (-100 20)
filter: ST_EQUALS(location, 'POINT (-100 20)') && id == 70000100
result: []
```

专用保留环境 `repro302-s26-xbxjj` 重复了同一失败。对照证据为：

1. candidate 阶段 upsert 的 `id=50000100`，PK-only、spatial-only 和 conjunction 均命中。
2. 回滚 v2.6.20 后，candidate 写入的同一行仍被三种查询命中，排除跨版本 RTREE 数据不可读。
3. v2.6.20 回滚阶段写入的 `id=70000100` 在 workflow probe 时为空，约一分钟后同一查询自动恢复；RTREE 为 `Finished` 且 `pending_index_rows=0`。
4. 在同一 v2.6.20 环境额外对 `id=1234` 做单行 upsert，把 geometry 从 `POINT (-121.966 37.034)` 改为 `POINT (10 10)` 并 flush。Strong consistency 的 PK 查询从 1.5 秒起持续返回新 payload，但新 spatial conjunction 在 1.5、3.5、5.5、7.6、9.6、11.6、13.6 秒均为空，15.6 秒才命中；旧 spatial conjunction 同期也为空。整个窗口 index metadata 一直为 `Finished / pending_index_rows=0`。

这组结果证明两个不同层面：

- **Milvus v2.6.20 产品缺陷：** Strong spatial query 在已确认 upsert 后短暂违反[官方 Strong consistency“读取最新版本、等待最新数据可见”语义](https://milvus.io/docs/consistency.md)，且 index progress metadata 不能反映该不可见窗口。问题在 standalone/cluster 均出现，单行 Strong probe 可复现；未在 candidate 阶段出现。
- **测试阻塞：** 正式 workflow 使用 collection 默认 Bounded consistency，其他 phase visibility 已有 120 秒 deadline，但 scalar-index probe 只执行一次，把可自行恢复的短窗口错误提升为 gate failure。修复后同一个严格 probe 在既有 120 秒 deadline 内重试，超时仍保留原始 `INDEX_SCALAR_QUERY_FAILED`，不跳过 RTREE、不更改 filter、不吞错误。

该 v2.6.20 缺陷不是 3.0.2 candidate 的升级数据损坏；它应由独立 Strong-consistency RTREE regression 跟踪，不应被本轮 Bounded compatibility gate 静默遗忘。

### 5. cluster 回滚到 v2.6.20 时函数输出 WAL 兼容导致 StreamingNode 崩溃

正式复跑 `r302-c26fix-f9m7s` 在 candidate 回滚到 v2.6.20 时，Helm rollout 用时 7 分 06 秒；v2.6.20 StreamingNode 期间连续重启 5 次，最后一次 termination 为 `exitCode=134`。容器 `imageID` 是固定的 v2.6.20 digest，断言和调用路径为：

```text
Assert "!field_id_to_offset.count(field_id)"
  => duplicate field data
  at SegmentGrowingImpl.cpp:461
delegator.ProcessInsert -> growing.Insert -> panic
```

配置与源码把跨版本因果链闭合：

1. 回滚完成后 MixCoord management API 返回 `function.enableWriteBeforeMaterialization=true`，`source=EtcdSource`。
2. candidate 包含 [commit `19236379caa7...`](https://github.com/milvus-io/milvus/commit/19236379caa7e6e351eb2fbb4e1a868285e649f9)，其默认 `auto` 策略在所有在线节点稳定达到 2.6.23 以上一分钟后，把该配置一次性写为 `true`；升级到全 candidate 后条件成立。
3. 该配置让 candidate StreamingNode 在 WAL append 前写入 BM25 等 function output。崩溃前日志同时显示 v2.6.20 embedding pipeline 正在消费 candidate 阶段消息。
4. v2.6.20 tag `65db4eaa...` 不包含后续 2.6 分支的 write-before/legacy reuse 改造 `a0f27ebbb1...`，其 QueryNode embedding fallback 会再次 append output field；segcore 的 duplicate field 防御断言随后杀死进程。
5. 独立 strict cluster `r302-ng-c26-qrxzx` 在相同默认升级/回滚路径再次复现：新建的 v2.6.20 StreamingNode 首次启动 32 秒后以 `exitCode=134` 退出并进入 `CrashLoopBackOff`；容器 `imageID` 同样精确为上述固定 digest。previous log 在 `2026-09-13 07:35:50 UTC` 记录完全相同的 `SegmentGrowingImpl.cpp:461` 断言及 `ProcessInsert -> insertNode -> streamPipeline` 栈。该 Pod 最终累计 6 次重启后才恢复 Ready；回滚 rollout 结束后的 steady-state 仍有 10 个失败 slice，跨 372 秒、65 次请求失败，错误集中为 `no available shard leaders / channel not available`。
6. 正式最终重跑 `r302-c26fix3-gnn49` 第三次直接复现：v2.6.20 StreamingNode 固定为同一 digest，累计 5 次 `exitCode=134` 后恢复；最近 previous log 仍为同一 duplicate-field 断言和调用栈。三条保留到可核验日志的独立 cluster 回滚全部命中，直接复现率为 3/3。
7. 同一正式最终重跑的 rollback maintenance window 为 `08:13:42–08:22:07 UTC`。其中 `delete_pressure_167.json` 在 `08:17:52–08:18:11` 对 7 个 collection 发起 8 次 complex delete，全部因 QueryStream 路径 `node not found` 失败。该 slice 与 CrashLoop 精确重叠；分类器只允许无副作用的 DQL node rotation，不允许 DML，因此 final report 正确保留该失败。其余 steady-state 压力为 `513099/513099`。
8. 首次 cluster 2.6 运行的同一 rollback patch 也异常耗时 7 分 25 秒；该轮环境已清理，不能反推其重启数，因此不把它计入上述 3/3 直接复现证据。

StreamingNode 最终自行恢复、workflow 得以继续，不改变“受支持 rollback 期间 Milvus 进程反复崩溃”的事实。源码提供在升级前显式设 `function.enableWriteBeforeMaterialization=false` 的 escape hatch，但本轮正式 gate 验证的是默认配置，不能用手工 override 改写结果。该问题是 3.0.2 默认行为对 v2.6.20 rollback 的产品兼容缺陷。

### 6. debug count 日志分支空结果越界导致 candidate 进程崩溃

非 gate strict cluster `r302-ng-c26-qrxzx` 升级 rollout 中，candidate QueryNode 重启 2 次、candidate StreamingNode 重启 1 次；可保留的最近两次 previous log 都是同一 panic：

```text
panic: runtime error: index out of range [0] with length 0
internal/querynodev2/segments.retrieveOnSegments.func1
  /go/src/github.com/milvus-io/milvus/internal/querynodev2/segments/retrieve.go:63
```

[candidate 源码第 61–63 行](https://github.com/milvus-io/milvus/blob/c441224651/internal/querynodev2/segments/retrieve.go#L61-L63)只在 debug level 且请求为 count 时执行，并直接读取 `result.GetFieldsData()[0]...GetData()[0]`，没有检查空 slice。本次前序日志确认 count retrieve 正发生在 rollout 中的 sealed segment；测试压力模块合法发送 count，请求本身没有越权或畸形输入。v3.0.1 与 candidate 在该行代码相同，因此这是 candidate 可复现的现存 Milvus 缺陷，不是本轮新增代码回归；生产默认非 debug 日志不会走该分支，但 debug 级别不应改变请求正确性、更不应杀死进程。

正式 cluster 2.6 最终重跑 `r302-c26fix3-gnn49` 在独立数据集的 candidate rollout 中再次命中：固定 candidate digest 的 QueryNode 重启 2 次、StreamingNode 重启 1 次，最近 termination 均为 `exitCode=134`，两者 previous log 均落在完全相同的 `retrieve.go:63` 越界栈。该缺陷因此也达到两个独立 workflow 复现。

当前 gate 只按请求结果和最终 serviceability 判定，不会因为 Pod 曾经 exit 134 自动失败。因此该缺陷除产品修复外，还暴露出 workflow 缺少“非 rollout 预期退出 / rollout 新 Pod restartCount”断言。

## 优化项

1. 用 Argo/Kubernetes synchronization semaphore 代替客户端 `list -> submit` 轮询，消除状态可见性竞态。
2. scenario ID 仍带 `2-6-18`、`3-0-0` 等历史版本字样，而参数已覆盖为 v2.6.20/v3.0.1；报告和 dashboard 应优先显示解析后的真实三阶段版本。
3. final report stdout 是数万行完整 JSON，建议额外输出固定的小型 gate summary artifact，降低失败归因成本。
4. RTREE `describe_index` 的 `Finished / pending_index_rows=0` 与 Strong spatial 可见性不一致，建议产品侧增加可观测的 segment handoff/queryability 状态。
5. 为 RTREE upsert 增加独立 Strong consistency 延迟回归，记录 PK visibility 与 spatial visibility 的时间差；兼容 gate 保持 Bounded deadline 语义。
6. 对每个阶段保存组件 UID/restartCount 基线并检查非预期退出；允许 Deployment 正常替换旧 Pod，但新 revision Pod 的 panic/CrashLoop 必须进入 final gate，而不是只看最终 Ready。
7. 收敛仓库历史 Ruff 债务；当前全目录有 125 个 lint 错误、41 个文件不符合 formatter，导致本轮只能对 7 个实际改动 Python 文件做零错误校验。

## 清理与证据保留

- 无效初始轮次和正常完成 workflow 的 Milvus/Helm/PVC 资源均按 ownership 清理，Argo history/logs 保留。
- `repro302-s26-xbxjj` 取证后已删除 Milvus CR、etcd/MinIO 两个 PVC 和临时诊断 Pod；测试数据不可恢复，workflow history 保留。
- 错填 revision 的 `r302-c26fix2-f566d` 在创建 Milvus 资源前立即终止，不计入矩阵。
- 全部 workflow 结束后，以 `r302|repro302` 精确盘点 `qa-milvus` namespace：Milvus CR、Helm release、Pod、PVC、ConfigMap、Secret 均为 0；无本轮残留测试资源或数据。Argo workflow history/logs 保留。

## 最终静态验证

- `PYTHONPATH=. python3 -m pytest milvus_client/tests -q`：`644 passed in 55.12s`。
- 对相对 `origin/main` 的 7 个实际改动 Python 文件执行 Ruff lint 与 format check：全部通过，`7 files already formatted`。
- `argo lint --offline argo`：通过，无 lint error。
- 20 组渲染参数 JSON 均可被 `jq` 解析，调度脚本通过 shell syntax check；artifact 中未发现非空 token/password/secret。
- `git diff --check` 与报告 placeholder 检查：通过。
- 全目录 Ruff 基线不通过：125 个既有 lint 错误、41 个未格式化文件；错误不位于上述 7 个本轮改动 Python 文件，未在本任务中批量改写。
