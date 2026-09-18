# Milvus 3.0.2 Cluster Upgrade Compatibility Validation

日期：2026-09-18  
测试仓库：`yanliang567/vectordb-testbricks`  版本：`03fe40ea74266af3b201851914d4fe3ab45fefb9`  
验证范围：仅 cluster 模式；主批次最多并发 3 个测试集群。

## 结论

主批次 6 个 release-gate 场景完成 5 个成功、1 个失败；对失败场景使用完全相同的参数和镜像做独立复跑后成功。因此本次结果不是“全绿首跑”，但也没有形成当前可稳定复现的确定性 Milvus blocker。

建议 3.0.2 发布前保持条件 GO：

- 不要忽略主批次中观察到的 FAISS OPQ/PQ 过滤检索异常；应保留为发布风险并继续做重复运行或增加专项回归。
- 在该专项至少再取得一次连续成功，或完成 Milvus 侧根因定位后，再把结果升级为无条件 GO。
- 本次没有修改测试断言来规避失败；测试逻辑保持原样。

## 构建产物

使用 `milvus-dev-cli` 编译最新 `3.0` 分支：

- Build ID：`build-3-0-a13c7be-u3805`
- Commit：`a13c7be343e932f2dcf4b2f4741045f7aa1a592b`
- 状态：`succeeded`
- Image：`harbor.milvus.io/manta/milvus:3.0-20260918-a13c7be`
- amd64 digest：`sha256:4ef9d9bcf50785b28f6a3999f296c59d42b6d4b2d11bed9270c0cc33535d7b31`

Baseline：

- 2.6：`harbor.milvus.io/milvusdb/milvus:v2.6.22@sha256:0c4247a007eca14e539b84a11ae3970ca332595bf599c82594a5681ef0a88fb3`
- 3.0：`harbor.milvus.io/milvusdb/milvus:v3.0.1@sha256:2b2fc2cf499ad897c93d4b90ee251646f718522845261a342d74d7c0c66bb274`

## 主批次结果

工作流 label：`release-validation=milvus-3-0-2-cluster-20260918`。调度器在任意时刻最多保持 3 个 active workflow；主批次完成后 `qa-milvus` 中没有残留测试集群 pod。

| 场景 | Workflow | 结果 | 说明 |
|---|---|---|---|
| 2.6.22 → candidate → 2.6.22，target-only features | `r302-c18-p2t8z` | Succeeded | 2.6 cluster upgrade/rollback 通过 |
| 3.0.1 → candidate → 3.0.1，核心兼容 | `r302-c18-v7wbx` | Failed | `validate-phase-dml-dql-after-rollback` 失败 |
| 3.0.1 → candidate → 3.0.1，JSON shredding | `r302-c18-56smc` | Succeeded | Woodpecker 1CU |
| 3.0.1 → candidate → 3.0.1，Woodpecker 2CU HA | `r302-c18-fgcjq` | Succeeded | HA/压力窗口通过 |
| 3.0.1 → candidate → 3.0.1，Index V10/V4 | `r302-c18-8fdcs` | Succeeded | index compatibility 通过 |
| 3.0.1 → candidate → 3.0.1，Index V11/V4 | `r302-c18-txcxp` | Succeeded | index compatibility 通过 |

原始失败 workflow 用时约 54 分钟；其他长场景包括压力、回滚、服务性等待和清理，均已正常收敛。

## 失败证据与归因

失败发生在 `r302-c18-v7wbx` 的 `validate-phase-dml-dql-after-rollback`，不是部署、升级、回滚或 checkpoint reload 失败。失败结果为：

```text
collection: qa_gate_cluster_30_to_30latest_faiss_float_binary
field: float_opq_pq
filter: id == 50000999
expected_pks: 50000999
actual_pks: 10026184
message: phase vector search did not return the newly written primary key
server_version: 3.0.1
```

该 collection 使用 `FAISS OPQ16,IVF64,PQ16x4`。同一结果中：

- `phase_checkpoint_reload_collections_total=16`
- `phase_checkpoint_reload_failures_total=0`
- `phase_checkpoint_scalar_index_queries_total=16`
- 16 个 collection 的 maintenance window 均成功
- 失败是带 scalar filter 的 vector search 返回了 filter 之外的错误 PK，而不是数据写入、reload 或 scalar index query 失败

测试代码在 `milvus_client/requests/validate_phase_dml_dql.py` 中使用 `id == expected_pk` 构造过滤条件，并严格校验返回 hit 的 primary key；因此没有证据表明该失败来自测试断言错误。Milvus 3.0.1 源码 `internal/querynodev2/segments/segment.go` 的 `LocalSegment.Search` 也明确将请求交给 segment search/filter path。以上证据足以确认：主批次确实观察到一次 Milvus 返回错误检索结果的产品级异常。

但该异常未能在独立复跑中再次出现，故不能把它定性为稳定复现的 3.0.1 回滚 blocker，也不能仅凭一次失败确定根因属于 FAISS、QueryNode filter path、segment reload 状态还是压力并发交互。

## 独立复跑

复跑 workflow：`r302-c18-retry-hlhvt`，label：`release-validation=milvus-3-0-2-cluster-20260918-retry`。

- 使用与失败场景相同的 base/rollback `v3.0.1` digest、candidate digest、schema、DML/DQL 参数和 rollback 流程
- 结果：`Succeeded`
- rollback 后 phase DML/DQL：`status=passed`
- `phase_checkpoint_reload_collections_total=16`
- `scalar_index_queries_total=24`
- `searches_total=36`
- `faiss_float_binary` rollback 后 collection reload、count、DML、filtered vector search 均通过
- 复跑完成后 `qa-milvus` 中没有残留 retry pod

所以当前分类为：一次已证实发生、但未稳定复现的 Milvus 检索异常；发布风险为 intermittent/regression risk，而不是已确认的 deterministic blocker。

## Index version / compact 覆盖边界

需要特别说明：本次 index V10/V4、V11/V4 场景没有手动调用 compact。

- Workflow 只在升级/回滚配置中设置 `dataCoord.targetVecIndexVersion` 和 `dataCoord.targetScalarIndexVersion`；没有调用 `compact`、`get_compaction_state` 或等价 compaction API。
- index compatibility validator 的实际参数是 `--rebuild-index false`。代码中虽然存在 rebuild 分支，但本次不会进入该分支；该分支也只执行 flush、release、drop/create index，不包含显式 compact。
- validator 检查的是 public index metadata、vector/scalar search/query、release/load reload cycle，以及 rollback 后 metadata 与 checkpoint 的一致性。
- 没有读取 segment 列表或 segment-level index version，也没有断言存量 sealed segment 已经被 compact/rebuilt 到目标 index version。

因此，本次结果不能证明“存量数据已经 compact 到新 index version”，也不能严格保证后续 search 使用的是新 index-version segment；只能证明在当前运行时状态下 index metadata 和查询结果兼容。若要覆盖该发布风险，需要新增专项：插入并 flush 存量数据后显式 compact，等待 compaction 完成，读取 compaction/segment/index metadata，确认所有目标 segment 的 index version，再执行 search/query 回归。

## Storage v2 → Storage v3 覆盖边界

本次实际执行的 6 个 workflow 均为：

```text
base-loon-ffi-enabled=false       base-vortex-enabled=false
target-loon-ffi-enabled=false     target-vortex-enabled=false
rollback-loon-ffi-enabled=false   rollback-vortex-enabled=false
```

因此本次没有覆盖 storage v2 → storage v3。Storage v3 由 `common.storage.useLoonFFI` / `loon_ffi_enabled=true` 控制；Vortex 是独立的 `dataNode.storage.format` 存储格式能力，不是 Storage v3 的必要条件。实际场景中只有 JSON shredding 场景改变了 JSON 配置，不等价于 storage v3 迁移。

仓库中已有独立的 cluster Storage v3 场景 `cluster-3-0-1-loon-ffi-rollback`，其 target 开启 LoonFFI、Vortex 保持关闭；但它没有进入本次 6 个 cluster 测试集群调度。因此 storage v2 → v3 的存量 segment 转换、compaction、回滚读取和 Storage v3 segment search 均未被本次验证。Vortex 场景另行覆盖 Vortex format，不应与 Storage v3 概念混用。

## 指定链路：2.6.22 → 3.0.2 → 3.0.2 + Storage v3

已新增并提交专项 cluster 场景 `cluster-2-6-22-to-3-0-2-storage-v3-compaction`，目标链路为：

```text
2.6.22 / storage v2
  → 3.0.2 candidate / storage v2
  → 开启 common.storage.useLoonFFI / storage v3
```

专项 validator 的设计和实现包含以下硬性检查：

- 在 storage v3 切换后显式调用 `compact()`，轮询 `get_compaction_state()` 和 compaction plan 直到完成；
- 检查存量数据行数、persistent segment 与 loaded/serving segment 的 ID 对齐；
- 检查 compaction 后 active segment 的 storage version 为 v3，并要求连续稳定采样；
- 执行 release/load，使新 segment 重新加载；
- 对 reload 后 segment 执行完整 PK query 和 indexed vector search，作为“新 storage v3 segment 已进入内存并被 query/search 使用”的运行时证据。

本次实际提交的 Argo workflow 使用了正确参数：`base-milvus-image=v2.6.22`、`target-version=3.0.2`、`post-upgrade-loon-ffi-enabled=true`、`storage-v3-compaction-validation-enabled=true`，且 `target-vortex-enabled=false`。但是该 workflow 在切换 LoonFFI 之前的 `wait-upgrade-serviceability` 阶段被 Milvus 数据面阻塞，未到达 validator。因此本次不能声称上述 compact、storage v3 segment 加载和 search/query 使用验证已通过；准确结论是“专项已实现，但运行被 Milvus 兼容性阻塞”。

## Storage v3 专项运行阻塞与归因

工作流 `c2622-sv3-mzds9` 的已完成阶段包括：2.6.22 部署、存量数据写入、升级到 3.0.2、升级后 storage v2 配置断言、5 分钟观察和升级后 precheck。阻塞阶段反复返回：

```text
failed to query: no available shard leaders:
channel not available[channel=by-dev-rootcoord-dml_12_469163131187497216v0]
```

证据链如下：

1. MixCoord/QueryCoord 日志持续报告 `no shard leader for replica to load segment` / `leader is not available in replica`；Pod 虽然 Ready，但数据面没有可用 shard leader。
2. StreamingNode recovery 反复报 Woodpecker file-resource 读取失败，并尝试连接 `127.0.0.1:443`；重启 StreamingNode 后仍复现，排除了简单 Pod/session 残留。
3. 读取 etcd 中 `woodpecker/logs/by-dev-rootcoord-dml_12/segments/0..4` 的 protobuf 元数据：2.6.22 创建的旧 segment（包括 segment 2）为 `quorumId=18446744073709551615`（protobuf 中的 -1），且没有 field 9 `QuorumInfo`；新 segment 才带有当前 Woodpecker quorum 信息和正确的 Woodpecker endpoint。
4. 当前环境中 Woodpecker segment handle 的 `GetQuorumInfo` 对 `quorumId<=0` 的 fallback 是 `Nodes=["127.0.0.1"]`；这与 recovery 的 `127.0.0.1:443` 失败目标完全对应。该 fallback 对本地/embedded 模式有意义，但在 service-mode 恢复 2.6.22 存量 segment 时导致无法读取旧 WAL/file resource。

因此该失败归类为 Milvus/Woodpecker 存量 WAL segment 元数据兼容阻塞，而不是 LoonFFI、Vortex 或测试断言问题。它发生在 Storage v3 切换之前，所以本轮没有实际调用 compact，也没有产生“storage v3 已更新到内存并被 search/query 使用”的有效证据。应在修复旧 segment quorum 元数据兼容/迁移逻辑后重新运行该专项；不能通过跳过 serviceability gate 来判定 Storage v3 兼容。

## 测试与环境阻塞处理

- 本地测试初次被环境中的 `pytest-sugar 0.9.5` 阻塞：它依赖已从 pytest 8 移除的 `py.std`。使用 `PYTHONPATH=. pytest -q -p no:sugar` 后，`milvus_client/tests` 为 `649 passed in 60.19s`。这是测试运行环境问题，没有修改测试逻辑。
- 调度器首次运行因生成 artifact 路径错误没有提交 workflow，已修正路径并重新执行；该问题属于测试编排脚本问题，不是 Milvus 失败。
- 个别 cluster 启动时出现 QueryNode/Proxy 重启、etcd deadline、Woodpecker memberlist 端口冲突；服务最终 Ready，相关 workflow 未因此失败，归类为启动时序/环境瞬态。
- 2CU HA 场景出现过 best-effort flush 被 RateLimiter 拒绝的日志，但 workflow 成功，且测试代码对该 flush 使用有限 deadline 并由后续可见性校验决定 gate 结果；未将其误判为发布阻塞。

## 产物与复核入口

- 计划：`docs/plans/2026-09-18-milvus-3-0-2-cluster-upgrade-compatibility.md`
- 调度与最终结果：`milvus-bricks/artifacts/2026-09-18-3-0-2-cluster-upgrade-compatibility/`
- 主批次结果：`final-workflows.tsv`、`final-workflows.json`
- 复跑结果：`retry-workflow.json`
- 主批次调度脚本：`scheduler.sh`
