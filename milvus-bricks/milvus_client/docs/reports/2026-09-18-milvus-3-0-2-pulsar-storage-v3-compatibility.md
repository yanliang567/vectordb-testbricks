# Milvus 2.6.22 → 3.0.2 Pulsar / LoonFFI Storage v3 兼容性报告

日期：2026-09-18  
范围：QA 4am Kubernetes，cluster 模式，最多 3 个并发测试集群  
测试仓库：`yanliang567/vectordb-testbricks`，revision `464c193e34909a32681d2aa458979482771a1b3d`

## 结论摘要

本轮目标路径为：

```text
Milvus v2.6.22 / storage v2
        │  Pulsar WAL
        ▼
Milvus 3.0.2 candidate / storage v2
        │  开启 LoonFFI，由 LoonFFI 控制 Storage v3
        ▼
Milvus 3.0.2 / Storage v3
```

Pulsar 路径已完成端到端验证；Storage v3 验证不是只看配置开关，而是执行了手动 `compact`、核对 source→target segment lineage、release/load、QueryNode 加载版本、query 和 vector search。最终验证器结果为 **11/11 collections、7 个手动 compact jobs、20 个 compaction plans、4 个异步自动转换为 v3、11/11 持久化为 storage v3、11/11 load 后 query、33 次 search，PASS**。

当前 candidate 的发布建议仍为 **NO-GO**，原因不是 Pulsar 或 Storage v3 数据路径失败，而是已确认的共享 Milvus debug count panic（v3.0.1 与 candidate 都存在）。本报告同时记录一个 Milvus QueryCoord 观测缺陷：SDK 的 loaded segment API 对已由 QueryNode 实际加载的 storage v3 segment 返回 `storage_version=0`，不能据此判定 Storage v3 未生效。

## 固定输入

| 角色 | 固定输入 |
| --- | --- |
| 2.6 baseline | `harbor.milvus.io/milvusdb/milvus:v2.6.22@sha256:0c4247a007eca14e539b84a11ae3970ca332595bf599c82594a5681ef0a88fb3` |
| 3.0 baseline | `harbor.milvus.io/milvusdb/milvus:v3.0.1@sha256:2b2fc2cf499ad897c93d4b90ee251646f718522845261a342d74d7c0c66bb274` |
| 3.0.2 candidate | `harbor.milvus.io/manta/milvus:3.0-20260918-a13c7be@sha256:4ef9d9bcf50785b28f6a3999f296c59d42b6d4b2d11bed9270c0cc33535d7b31` |
| build | Milvus Dev CLI build `build-3-0-a13c7be-u380`，commit `a13c7be343e932f2dcf4b2f4741045f7aa1a592b` |
| deploy profile | `cluster-pulsar-1cu.yaml`，`woodpecker=false`，`pulsarv3=true`，`msgStreamType=pulsar`，Vortex=false |

LoonFFI 是 Storage v3 的控制路径；Vortex 与本场景无直接关系，未把 Vortex 当作 Storage v3 证据。

## 验证步骤与证据

### 1. v2.6.22 → 3.0.2 Pulsar 升级

Workflow 使用 cluster profile，先在 v2.6.22 创建 schema、写入存量数据并完成 baseline checkpoint，再升级到 candidate。升级前后均执行 serviceability、schema/data、index compatibility、DML/DQL 和持续 pressure 验证。升级后切换 LoonFFI 配置，等待配置生效，再进入 Storage v3 验证。

最终干净重跑 workflow：[`c2622-pulsar-final-b27pp`](https://argo-workflows.zilliz.cc/workflows/qa/c2622-pulsar-final-b27pp)，Argo `Succeeded`。前一轮主验证 [`c2622-pulsar-rerun-nsblk`](https://argo-workflows.zilliz.cc/workflows/qa/c2622-pulsar-rerun-nsblk) 的业务验证已通过，但 onExit 报告生成器因参数解析缺失退出 2；该 test-bricks bug 已在 `a9b86f4` 修复，Storage v3 混合版本收敛问题随后在 `464c193` 修复，最终 workflow 的 `generate-final-report` 和 `gate-final-status` 均成功。

### 2. 手动 compact 与存量 segment 转换

Storage v3 validator 对每个 collection 做以下判断：

1. 读取当前 live primary-key 集合，避免把 pressure 期间已 upsert/delete 的旧 baseline 当成错误；
2. 读取 persistent segment metadata，确认 compact 前为 storage v2；
3. 手动调用 `client.compact()`，等待 Completed；
4. 读取 compaction plans，要求存在 source→target transition，并确认 source 被清理；
5. 确认 active target segment stable/sealed、storage version 为 3；
6. release/load 后确认 serving segment ID 与 persistent active segment ID 完全一致；
7. 对 compact 前 live PK 做 query，并执行 vector search。

前一轮相同 validator 的 Proxy 证据已经捕获到：

```text
received ManualCompaction
compactionID=469166344946182951
compactionPlanCount=5
state=Completed
5 个 mergeInfos 均为 source→target，所有计划为 CompactionTaskStateCleaned
```

最终 workflow `c2622-pulsar-final-b27pp` 的 validator stdout 给出：

```json
{
  "status": "passed",
  "collections_checked": 11,
  "compact_jobs": 7,
  "compaction_plans": 20,
  "already_storage_v3_collections": 4,
  "storage_v3_persistent_collections": 11,
  "storage_v3_loaded_collections": 11,
  "query_collections": 11,
  "searches_total": 33,
  "failures": []
}
```

因此本次确实手动 call compact，并且确实验证了存量数据被 compact 到新的 storage version；不是只创建新 segment 或只验证配置。

### 3. 新 storage version 是否被 load 到内存并被 search/query 使用

验证器在 compact 完成后显式 release/load，并要求 serving IDs 等于 compact 后 persistent active IDs。QueryNode 日志进一步给出直接证据：同一批 target segment 的 `SetLoadInfo` 和后续 `Reopen segment` 日志中出现 `storage_version: 3`，且伴随成功 load。上一轮已保留的代表性日志为：

```text
SetLoadInfo ... segment 469166344946212955 ... storage_version: 3
SetLoadInfo ... segment 469166344946222956 ... storage_version: 3
Successfully loaded segment ...
Reopen segment ... 469166344946212955
```

这与 validator 的三层结果闭合：

```text
persistent active target IDs == serving loaded IDs
query PK 集合 == compact 前 live PK 集合
11 个 collection 均执行 query，33 次 vector search 成功
```

SDK 的 loaded segment metadata 有一个独立观测问题：QueryCoord 的 `GetLoadSegmentInfo` 经 `internal/querycoordv2/utils/types.go:MergeMetaSegmentIntoSegmentInfo` 合并时没有复制 `StorageVersion`；Proxy 只是把缺省的 `info.GetStorageVersion()` 返回给客户端。因此 SDK 看到 loaded `storage_version=0` 并不否定 QueryNode 已按 v3 加载。真正的数据路径证据是 QueryNode `SetLoadInfo(... storage_version: 3)` 和 search/query 成功。

## Woodpecker / Milvus 归因

### 结论

兼容性失败的直接根因在 **Woodpecker service-mode metadata compatibility contract**；Milvus 另有一个 release/integration 兼容性缺口，但不是“QueryCoord 无 shard leader”本身的根因。

### 确凿证据

1. v2.6.22 的 go.mod 固定 Woodpecker 为 `v0.1.14-0.20260519022627-fdf34b23e0d7`。该版本 `woodpecker/segment/segment_handle.go:91-110` 的 `NewSegmentHandle` 明确写有 `TODO: get from metadata in cluster mode`，并硬编码：

   ```go
   QuorumInfo{Id: 1, Wq: 1, Aq: 1, Es: 1, Nodes: []string{"127.0.0.1"}}
   ```

2. v2.6.22 创建的老 segment metadata 只有 deprecated `quorumId=-1`，没有 field 9 的 endpoint-bearing `QuorumInfo`；新写入的 segment 才有当前 WP endpoint 信息。

3. 当前 Woodpecker `v0.1.40` 的 `GetQuorumInfo` 在 `quorumId<=0` 时有明确 fallback：`Nodes: []string{"127.0.0.1"}`。其自身测试 `segment_handle_test.go:4908-4932` 也固定断言该行为。因此 reader 面对旧 metadata 会把 cluster segment 当成 local/standalone quorum。

4. Milvus `pkg/streaming/walimpls/impls/wp/builder.go:61-66` 在 service mode 调用 `woodpecker.NewClient(...)`，并非仅使用 embedded client；`builder.go:171` 把 Milvus storage type 传入 WP config，说明这是 cluster service-mode recovery 路径。

5. 失败 workflow [`c2622-sv3-mzds9`](https://argo-workflows.zilliz.cc/workflows/qa/c2622-sv3-mzds9) 的日志与源码闭环：WP recovery 连接 `127.0.0.1:443` 被拒绝，随后 MixCoord/QueryCoord 报 `no shard leader for replica to load segment`，客户端最终报 `no available shard leaders`。StreamingNode 重启并 Ready 后仍复现，说明不是单次 pod readiness 假象。

6. Milvus commit `321719486912d4b5b6e3f823b227fc6daffc316c` 只是将 WP `v0.1.40` bump 到 `v0.1.42`，没有 Milvus 侧 metadata migration/rejection；因此 Milvus 的责任是发布集成未在 recovery 前处理旧 metadata，而不是证明 QueryCoord 自己丢失 leader。

### 归因边界

| 现象 | 归因 |
| --- | --- |
| 旧 cluster metadata 缺少 endpoint-bearing quorum，reader fallback 到 `127.0.0.1` | Woodpecker service-mode 兼容契约缺陷 |
| Milvus 仍允许该旧 metadata 进入 3.0 recovery，未提前迁移或 fail-fast | Milvus release/integration compatibility gap |
| QueryCoord `no available shard leaders` | 下游症状，不是第一根因 |
| Pulsar profile 同 candidate 在同类升级验证中通过 | 控制变量证据，排除 candidate 的普遍 WAL/Storage v3 失败 |

## 已发现并修复的测试阻塞

- pressure daemon 未停止就开始 storage validator，导致数据持续变化；增加 stop ConfigMap 和 quiesce 窗口。
- validator 使用 stale baseline row count/PK，导致 pressure 下永远不收敛；改为 compact 前读取 live PK 集合。
- QueryCoord loaded storage version 缺省为 0；改为校验 non-zero 不冲突，同时用 QueryNode 日志证明真实 load version。
- checkpoint 返回值解包错误，导致 10 个 collection 报 `KeyError: storage_version`；已修复。
- 已由自动 conversion 提前变为 storage v3 的 collection 现在记录并继续做 release/load/query/search，不重复强制 compact 前 v2 断言。
- Argo DAG 中 disabled 的 schema-evolution task 依赖仍运行的 pressure daemon，而 stop task 又等待该 task，形成死锁；已移除该可选 task 对 pressure daemon 的不必要依赖。
- final report argparse 缺少 workflow 已传入的两个 storage/LoonFFI 参数；已补齐。

## 发布判断

Storage v2→v3 的 Pulsar 升级兼容路径：**测试通过**。  
Woodpecker 旧 metadata recovery：**不兼容，根因已定位**。  
3.0.2 release gate：**NO-GO，需先处理或明确豁免共享的 debug count panic**；该问题不是本次 Pulsar Storage v3 验证失败。

## 可复核代码与提交

- test branch：`test/3-0-2-upgrade-compat-validation`
- validation code commit：`464c193 test: wait for storage version conversion to converge`
- report commit：`dd7cf15 docs: record final pulsar storage v3 gate evidence`
- storage validator：`milvus-bricks/milvus_client/requests/validate_storage_v3_compaction.py`
- Pulsar profile：`milvus-bricks/milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml`
- report generator：`milvus-bricks/milvus_client/requests/generate_workflow_report.py`
- Milvus observation path：`internal/querycoordv2/services.go:404` → `internal/querycoordv2/utils/types.go:MergeMetaSegmentIntoSegmentInfo` → `internal/proxy/impl.go:4474-4515`
