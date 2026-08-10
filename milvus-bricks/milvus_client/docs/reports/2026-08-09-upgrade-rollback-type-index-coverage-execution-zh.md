# Milvus 升级/回滚数据类型与索引覆盖测试执行报告

日期：2026-08-09

## 1. 报告结论

PR [yanliang567/vectordb-testbricks#25](https://github.com/yanliang567/vectordb-testbricks/pull/25)
已完成代码 review、离线回归和 QA Kubernetes 集群真实升级/回滚验证。

本轮真实 Milvus 测试主要基于代码提交：

```text
a5d91d69d48672832b9bf075c1dee0ba0a664720
```

后续提交除执行报告和 Issue 草稿外，还包含 review 驱动的测试框架
fail-closed hardening；这些变更未重新执行历史 Kubernetes workflow。

总体结论：

- 后续 review 提出的 3 个 P1、4 个 P2 均已修复，未留下未处理的 P1/P2 问题。
- `2.6.18 -> 3.0 -> latest 2.6` rollback-safe 路径通过。
- index engine version `10/4` 的 sealed index 构建及回滚复用通过。
- standalone JSON Shredding 全 DML 升级/回滚通过。
- cluster JSON Shredding 的数据和索引格式兼容性通过。
- 4 类 Milvus 产品问题已稳定复现并提交正式 Issue。
- 对应 feature gate 保持严格失败，没有通过 warning、skip 或重建索引掩盖问题。

## 2. 已提交的 Milvus Issues

| 问题 | Milvus Issue | 状态 |
|---|---|---|
| StructArray FLOAT16 DISKANN 返回负的 MAX_SIM_COSINE 自相似分数 | [milvus-io/milvus#52338](https://github.com/milvus-io/milvus/issues/52338) | Open, `kind/bug`, `needs-triage` |
| SINDI growing sparse index 在有效版本 8 上触发 QueryNode SIGSEGV | [milvus-io/milvus#52339](https://github.com/milvus-io/milvus/issues/52339) | Open, `kind/bug`, `needs-triage` |
| 新版 3.0 写入的 `vortex.variant` 无法被 v3.0.0 回滚版本读取 | [milvus-io/milvus#52340](https://github.com/milvus-io/milvus/issues/52340) | Open, `kind/bug`, `needs-triage` |
| Woodpecker reader state 丢失导致 rollback 后 channel tSafe 永久卡住 | [milvus-io/milvus#52341](https://github.com/milvus-io/milvus/issues/52341) | Open, `kind/bug`, `needs-triage` |

## 3. 测试环境

- Argo namespace：`qa`
- Milvus namespace：`qa-milvus`
- PyMilvus：`3.0.1`
- Standalone WAL：RocksMQ
- Cluster WAL：Woodpecker
- Cluster profile：1 QueryNode、1 DataNode、1 MixCoord、1 StreamingNode、4 Woodpecker Pods

Milvus 镜像：

| 阶段 | 镜像 |
|---|---|
| 2.6.18 base | `harbor.milvus.io/milvusdb/milvus:v2.6.18@sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6` |
| 新版 3.0 target | `harbor.milvus.io/milvusdb/milvus:3.0-20260807-1439dc7d@sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64` |
| latest 2.6 rollback | `harbor.milvus.io/milvusdb/milvus:2.6-20260807-d85dc945@sha256:2051a754368d70f589a281fa301a12128d058e531bd6e5d82583e588bccd961e` |
| 3.0 baseline/rollback | `harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862` |

## 4. 本轮测试场景汇总

以下按测试意图归并。代码修正前的重复失败、参数验证和纯日志采集
workflow 不单独作为功能场景统计。

| 场景 | 代表 Workflow | 规模 | 最终结果 | 结论 |
|---|---|---:|---|---|
| 2.6 rollback-safe 类型/索引 | `pr25-st26-current-r1-g7wxd` | 5000 行/schema | Pass | 当前 PR SHA、三阶段 digest、完整 DML 和回滚复用通过。 |
| 3.0 core target-built indexes | `pr25-st30-forward5000-r1-tl6pm` | 5000 行/schema | Fail | 除 StructArray FLOAT16 DISKANN 分数外，其余 target-side 检查通过。 |
| index version 10/4，全写压力 | `pr25-cl30-idx104-r1-gptfx` | 持续写入 | Fail | SINDI growing index 触发 QueryNode crash。 |
| index version 10/4，read-only pressure | `pr25-cl30-idx104-f5000-r1-ff9dq` | 5000 行/schema | Pass | sealed index 10/4、JSON scalar v4、回滚复用通过。 |
| JSON Shredding standalone smoke | `pr25-st30-json-r3-4gwcs` | 100 行/schema | Pass | nested/dynamic JSON 和回滚复用通过。 |
| JSON Shredding standalone formal | `pr25-st30-json-f5000-r1-c5kcs` | 5000 行 | Pass | 全 DML、6 个索引和 forward rollback validation 通过。 |
| JSON Shredding cluster full DML | `pr25-cl30-json-r3-77cdt` | 100 行/schema | Fail | rollback 后 channel tSafe 卡住约 14 分钟。 |
| JSON Shredding cluster read-only control | `pr25-cl30-json-ro-r1-r56q6` | 100 行/schema | Pass | 同配置和 phase DML/DQL 下格式兼容性通过。 |
| JSON Shredding cluster full DML 复现 | `pr25-cl30-json-r3-vw87h` | 100 行/schema | Fail | 复现 16 分钟以上 tSafe stall，并固定 Woodpecker reader 日志。 |
| Loon/Vortex standalone | `pr25-st30-loon-r4-jsw7x` | 100 行/schema | Fail | v3.0.0 缺少 `vortex.variant` decoder。 |
| Loon/Vortex cluster | `pr25-cl30-loon-r1-jfkzb` | 100 行/schema | Fail | Woodpecker 集群复现同一 persisted-format 问题。 |

控制和隔离场景：

| Workflow | 隔离变量 | 结果 |
|---|---|---|
| `pr25-st30-baseprobe-r1-97wzm` | v3.0.0 base，100 行，无重启 | Pass |
| `pr25-st30-reload-r1-58gmd` | 同一 v3.0.0 digest 强制 reload/rollback | Pass |
| `pr25-st30-xver-r1-gx2jp` | newer 3.0 回滚 v3.0.0，无持续写压力 | Pass |
| `pr25-st30-phasedml-r1-4tw27` | 只执行 target phase DML | Pass |
| `pr25-st30-indexedbase-r1-lrtct` | v3.0.0、5000 行、真实 index build | Pass |
| `pr25-st30-pressure-r1-hzp79` | 写压力生成 33560 行 DISKANN segment | Fail，同一负分问题 |

诊断/取证 workflow 包括：

- `pr25-idx104-f5000-diag-6lq6x`
- `pr25-idx104-target-diag-bc86k`
- `pr25-loon-diag-vkmnd`
- `pr25-cl-loon-diag-5skkn`
- `pr25-cl-json-diag-7mf69`

这些 workflow 用于固定 DataNode、MixCoord、QueryNode、StreamingNode、
Woodpecker 和 index build 日志，不单独计为 gate 结果。

## 5. 数据类型覆盖

### 5.1 Scalar 和集合能力

| 类型/能力 | 覆盖内容 |
|---|---|
| `BOOL` | 普通字段、nullable、StructArray 子字段、BITMAP |
| `INT8` | BITMAP、nullable/vector data pattern |
| `INT16` | INVERTED |
| `INT32` | STL_SORT |
| `INT64` | 主键、scalar、StructArray 子字段、STL_SORT、AUTOINDEX |
| `FLOAT` | 普通 scalar、StructArray 子字段、STL_SORT、INVERTED |
| `DOUBLE` | nullable、INVERTED、JSON path cast |
| `VARCHAR` | 主键、partition key、analyzer、StructArray、TRIE、NGRAM、BITMAP、INVERTED |
| `TEXT` | NULL、empty、Unicode、64 KiB 边界、超过 64 KiB、1 MiB |
| `JSON` | 普通 JSON、nested JSON、dynamic JSON、JSON path index |
| `ARRAY` | INT64、FLOAT、BOOL、VARCHAR array |
| `GEOMETRY` | POINT、RTREE、`ST_EQUALS`、`ST_DWITHIN` |
| `TIMESTAMPTZ` | STL_SORT、Entity TTL、expired/future/NULL |
| collection 能力 | auto-id、partition key、显式 partition、nullable、dynamic field、schema evolution |

### 5.2 Vector 类型

| Vector 类型 | 覆盖索引/语义 |
|---|---|
| `FLOAT_VECTOR` | HNSW、FLAT、IVF_FLAT、IVF_SQ8、IVF_PQ、SCANN、HNSW_SQ、IVF_RABITQ、DISKANN、FAISS、AUTOINDEX |
| `FLOAT16_VECTOR` | HNSW、HNSW_SQ、StructArray DISKANN |
| `BFLOAT16_VECTOR` | DISKANN、nullable semantics |
| `INT8_VECTOR` | HNSW、AUTOINDEX、nullable semantics |
| `BINARY_VECTOR` | BIN_FLAT、BIN_IVF_FLAT、FAISS BFlat、MINHASH_LSH |
| `SPARSE_FLOAT_VECTOR` | SPARSE_WAND、SPARSE_INVERTED_INDEX、BM25、SINDI、Block-Max |

覆盖 metric：

- `COSINE`
- `L2`
- `IP`
- `HAMMING`
- `BM25`
- `MHJACCARD`
- `MAX_SIM_COSINE`

### 5.3 StructArray

2.6 rollback-safe StructArray 子字段：

- `FLOAT`
- `VARCHAR`
- `INT64`
- `BOOL`
- `FLOAT_VECTOR`

3.0 StructArray scalar index：

| 子字段 | 类型 | 索引 |
|---|---|---|
| `score_sort` | FLOAT | STL_SORT |
| `score_inverted` | FLOAT | INVERTED |
| `category_inverted` | VARCHAR | INVERTED |
| `tag_bitmap` | VARCHAR | BITMAP |
| `rank_sort` | INT64 | STL_SORT |
| `enabled_bitmap` | BOOL | BITMAP |
| `attributes[embedding]` | FLOAT_VECTOR | HNSW + MAX_SIM_COSINE |
| `embeddings[vector]` | FLOAT16_VECTOR | DISKANN + MAX_SIM_COSINE |

### 5.4 Function 和特殊索引

| 功能 | 覆盖 |
|---|---|
| BM25 Function | VARCHAR/TEXT 输入，SPARSE_FLOAT_VECTOR 输出，BM25 ranking |
| MinHash Function | VARCHAR 输入，BINARY_VECTOR 输出，MINHASH_LSH + MHJACCARD；exact self-search 为强校验，近重复召回为观测指标 |
| JSON AutoIndex | 用户配置 AUTOINDEX，内部解析 HYBRID |
| index version | vector index version 10、scalar index version 4 |
| Storage V3 | LoonFFI、Vortex、TEXT LOB |

## 6. 索引类型覆盖

Scalar/JSON 索引：

- BITMAP
- INVERTED
- STL_SORT
- TRIE
- NGRAM
- RTREE
- JSON path BITMAP/STL_SORT/INVERTED/NGRAM
- HYBRID AutoIndex

Dense/Binary 索引：

- HNSW
- HNSW_SQ
- FLAT
- IVF_FLAT
- IVF_SQ8
- IVF_PQ
- IVF_RABITQ
- SCANN
- DISKANN
- AUTOINDEX
- FAISS IVF Flat
- FAISS OPQ + IVF + PQ
- FAISS BFlat
- BIN_FLAT
- BIN_IVF_FLAT

Sparse/Function 索引：

- SPARSE_WAND
- SPARSE_INVERTED_INDEX
- SINDI
- BLOCK_MAX_MAXSCORE
- BLOCK_MAX_WAND
- BM25 sparse index
- MINHASH_LSH

## 7. 生命周期验证点

### 7.1 Base 阶段

- 校验实际 server version 和 image family。
- 创建全部 rollback-safe collection 和 index。
- 插入确定性数据并 flush/load。
- 保存 count、PK sample、scalar checksum 和 index metadata checkpoint。
- 执行所有 scalar filter、vector search 和 feature semantics。
- 启动持续压力。

### 7.2 Upgrade 后

- 等待真实 Pod rollout 和服务可用，不只检查 Helm/CR 状态。
- 校验已有 collection 的 count、checksum、PK 和 index metadata。
- 执行已有数据 insert/upsert/delete/query/search。
- 创建 target-side phase collection 和 forward-only collection。
- 验证 target 新能力和 target-built index 文件。
- 校验 JSON Shredding、Loon/Vortex、index version 等 runtime config。

### 7.3 Rollback 后

- 等待 baseline image 和数据 query serviceability。
- 对 base collection 重新执行 checksum、PK、feature 和 index query。
- 对兼容的 forward collection 执行回滚后验证。
- 比较 rollback 前后 index name/type/metric/关键参数。
- 明确要求 `indexes_rebuilt=0`、`indexes_dropped=0`。
- 验证 upgrade 阶段写入的数据仍可查询和修改。
- 创建 rollback 阶段新 collection，并执行 index/filter/search。

### 7.4 专项语义

- StructArray scalar round-trip。
- StructArray element-level `PK + offset` search。
- MaxSim `EmbeddingList` row-level search 和 score/ranking。
- Nullable vector 的 NULL 排除和非 NULL self-search。
- Dynamic JSON 的 `dyn_bucket`、`dyn_text`、`dyn_json` checksum。
- Nested JSON path 和 cast type metadata/query。
- Geometry `ST_EQUALS`、`ST_DWITHIN`。
- Entity TTL expired/future/NULL 可见性。
- TEXT bytes/chars/prefix/suffix/SHA256。
- TEXT_MATCH、PHRASE_MATCH、BM25。
- MinHash exact document self-search 为强校验；近重复和无关文本是否召回为观测指标，仅在两者都返回时强校验排序关系。

### 7.5 压力和可用性

- search、query、iterator、count。
- insert、upsert、delete、mixed read/write。
- rollout window 和 steady-state 分开统计。
- rollout 期间的连接失败不污染 steady-state 指标。
- missing、pending、unreadable pressure result 计入 incomplete sample。
- serviceability 使用真实 count/query，而不是仅依赖 Pod Ready。

## 8. 通过场景详细结果

### 8.1 2.6.18 -> 3.0 -> latest 2.6

Workflow：`pr25-st26-current-r1-g7wxd`

```text
status: passed
rows per collection: 5000
collections checked after rollback: 7
actual indexes checked after rollback: 47
vector searches after rollback: 29
scalar index queries after rollback: 18
indexes rebuilt: 0
indexes dropped: 0
```

Feature semantics：

```text
StructArray scalar round-trip: 3 passed
StructArray element search: 1 passed
nullable vector semantics: 6 passed
Geometry filter: 2 passed
```

Rollback 后 phase DML/DQL：

```text
existing collections: 7
inserted into existing collections: 7000
upserted: 6000
deleted: 700
new collections: 7
inserted into new collections: 21000
checkpoint searches: 56
total phase searches: 84
```

压力结果：

```text
steady-state operations: 377631 / 377631
success rate: 1.0
strict failures outside maintenance windows: 0
```

### 8.2 Index Engine Version 10/4

Workflow：`pr25-cl30-idx104-f5000-r1-ff9dq`

```text
collections checked: 2
actual indexes checked: 9
scalar queries: 4
vector/sparse searches: 5
steady-state operations: 345140 / 345140
```

DataNode/MixCoord 日志确认：

```text
DataNode building index ... numRows=5000 current_index_version=10
Successfully prepare indexBuildTask ...
  currentIndexVersion=10 currentScalarIndexVersion=4
Successfully build index ... currentIndexVersion=10
```

JSON AutoIndex 内部解析：

```text
UserIndexParams.index_type=AUTOINDEX
IndexParams.index_type=HYBRID
json_path=json_auto['score']
json_cast_type=DOUBLE
```

### 8.3 Standalone JSON Shredding

Workflow：`pr25-st30-json-f5000-r1-c5kcs`

Rollback checksum 包含：

```text
id, tenant, category, json_profile, json_nested, tags,
dyn_bucket, dyn_text, dyn_json
```

```text
checkpoint rows: 5000
actual indexes checked: 6
scalar index queries: 5
vector searches: 1
steady-state operations: 469844 / 469844
success rate: 1.0
```

回滚后继续使用的 nested JSON path：

```text
json_nested['nested']['score'] -> INVERTED, DOUBLE
json_profile['bucket'] -> INVERTED, DOUBLE
```

### 8.4 Cluster JSON Shredding Read-Only Control

Workflow：`pr25-cl30-json-ro-r1-r56q6`

```text
all data/index/feature/phase validations: passed
rollback serviceability: first attempt
forward rollback serviceability: first attempt
steady-state operations: 452408 / 452408
success rate: 1.0
```

该结果证明 JSON Shredding 的数据和索引格式兼容 rollback。full-DML 场景
失败来自 Woodpecker WAL reader recovery，不是 JSON 文件本身不兼容。

## 9. 失败场景与根因

### 9.1 StructArray FLOAT16 DISKANN

Issue：[milvus-io/milvus#52338](https://github.com/milvus-io/milvus/issues/52338)

现象：

```text
expected PK: 0
actual PK: 0
metric: MAX_SIM_COSINE
actual score: approximately -0.999978
```

判断：正确 PK 但 score 符号错误。问题在 target rollback 前已复现，排除
rollback metadata 和 workflow 干扰。

### 9.2 SINDI Growing Index QueryNode Crash

Issue：[milvus-io/milvus#52339](https://github.com/milvus-io/milvus/issues/52339)

现象：

```text
Unsupported sparse inverted index algorithm SINDI for index version 8
SIGNO: 11; SIGNAME: Segmentation fault
GrowableInvertedIndex<float, float>::add(...)
```

判断：sealed index 真实使用 version 10，但 QueryNode growing path 仍选择
version 8，错误处理路径存在生命周期/异步访问问题。

### 9.3 Vortex Persisted-Format Rollback

Issue：[milvus-io/milvus#52340](https://github.com/milvus-io/milvus/issues/52340)

现象：

```text
Failed to open vortex file:
Registry missing encoding with id vortex.variant
```

判断：新版 3.0 writer 写入了 v3.0.0 reader 未注册的 encoding。Standalone
和 Cluster 均复现，属于持久化格式向后兼容问题。

### 9.4 Woodpecker Reader State / tSafe Stall

Issue：[milvus-io/milvus#52341](https://github.com/milvus-io/milvus/issues/52341)

现象：

```text
reader temp info not found
update reader info failed
direct read batch failed error="no record extract"
lag(16m43.836s) max(3s): channel tsafe stalled
```

判断：持续 DML 后，rollout/rollback 重建 scanner 时无法恢复 Woodpecker
reader 临时状态。consumer position 不再推进，最终 QueryNode tSafe 永久
停滞。read-only 对照没有出现该错误。

## 10. PR #25 Review 结论

Review 范围：

- schema/data DSL
- StructArray 和特殊数据生成
- feature validators
- index compatibility validator
- phase DML/DQL
- pressure availability/reporting
- 三套 Argo WorkflowTemplate
- gate manifest、capability/feature/brick catalog
- 单元测试、CI 和真实环境结果

Review 结论：

- 后续 review 发现并修复 6 个覆盖/实现缺口：phase search 旧数据假命中、
  Auto-ID 空 ID 响应假绿、resolved index type 不可观测时 fail-open、
  MinHash 近重复召回覆盖口径过强，以及 existing collection 在 upsert 后
  仍使用旧 seed 构造 search probe，以及 index search 缺少 score/distance
  或 BM25 命中无关 PK 时假绿。
- phase search 现在按本阶段真实 PK 加 filter，并校验 PK、StructArray offset
  和适用的 self-search score/distance。
- Auto-ID 现在要求每行返回唯一 ID，按原始数据行顺序保存实际 Milvus PK，
  rollback checkpoint 同时记录 generation PK 和实际 search probe PK。
- existing collection search 使用 post-upsert seed，并将 probe seed 写入
  checkpoint，rollback 验证优先复用记录值。
- matrix 声明 `expected_resolved_index_type` 后，SDK metadata 不可观测也会
  以 `INDEX_METADATA_MISMATCH` 失败。
- 所有 index search probe 现在都要求返回期望 PK 和可观测的
  score/distance，BM25 function output index 不再接受无关的非空命中。
- upgrade 和 rollback 的 index compatibility 都先验证已 load 状态，再
  release collection、重新 load，并重复 count/PK、vector search 和 scalar
  index query；reload 后任一验证失败都会阻断 gate。
- schema evolution 现在校验演进 PK 范围 count、演进字段、StructArray
  payload、顶层向量内容和 nullable vector NULL 状态 checksum，并按 metric
  校验确定性 self-search score/distance；after-upgrade 写 checkpoint，
  after-rollback 只读复验，不重复 schema/DML mutation。
- AutoID schema evolution 现在会实际插入演进行，要求每行返回唯一实际 PK，
  rollback 阶段按 checkpoint 的实际 PK 完成 count/query/search 复验；缺失或
  重复 ID 会直接失败。
- TEXT_MATCH/PHRASE_MATCH 先以确定性 ground truth 校验完整 `count(*)`，再
  抽样验证返回内容，单个正确 posting 不再能让 gate 通过。
- Entity TTL 临时行使用独立保留 PK namespace，与 continuous pressure 的
  insert/upsert/delete 区间隔离。
- 已注册 scenario 在部署前重新校验 WorkflowTemplate、schema matrix、index
  target version、validator/storage/pressure/data-scale 参数，防止绕过 renderer
  产生假绿。
- MinHash exact self-search 保持强校验；近重复召回明确降级为观测指标，
  仅在 near/unrelated 都返回时强校验排序。
- 所有已知产品失败均保持 fail-closed。
- JSON/Loon specialty matrix 与 StructArray core matrix 已隔离，避免无关问题互相遮蔽。
- index version 场景使用 read-only continuous pressure 是已知 SINDI crash 下的明确隔离策略，phase DML/DQL 仍覆盖写操作。
- 最终 review 发现两份 issue 草稿末尾多余空行，已修复。

上述 review 修复属于真实执行完成后的测试框架 hardening，已完成离线回归；
本报告中的历史 Kubernetes workflow 没有因这些纯测试框架变更重新执行。

剩余风险：

- 4 个 Milvus Issue 未修复前，相应 feature gate 不能作为 release-green。
- GPU index 不在本轮 CPU upgrade gate 范围内。

## 11. 自动化验证

```text
offline pytest: 383 passed
Argo offline lint: passed
Ruff check: passed
Ruff format check: passed
GitHub Actions: run `31363264831` 在 review-fix commit `6114728` 上通过
```

## 12. 资源清理

以下保留现场的 release 已在日志和报告下载后清理：

- `pr25-cl30-idx104-f5000-r1-ff9dq`
- `pr25-cl30-json-ro-r1-r56q6`
- `pr25-st26-current-r1-g7wxd`
- `pr25-cl30-json-r3-vw87h`

报告完成时：

- 无运行中的 `pr25-*` Argo Workflow。
- 无遗留的 owned Deployment、StatefulSet、Service、PVC、ConfigMap、Secret。

## 13. 最终状态

| 范围 | 状态 |
|---|---|
| 测试框架实现 | Pass |
| 2.6 rollback-safe 类型/索引 | Pass |
| Standalone JSON Shredding | Pass |
| Cluster JSON Shredding 格式兼容性 | Pass |
| Cluster JSON Shredding full-DML | Blocked by [#52341](https://github.com/milvus-io/milvus/issues/52341) |
| Index engine v10/v4 sealed index | Pass |
| SINDI growing-index write pressure | Blocked by [#52339](https://github.com/milvus-io/milvus/issues/52339) |
| StructArray FLOAT/VARCHAR scalar index | Pass at target validation |
| StructArray FLOAT16 DISKANN score | Blocked by [#52338](https://github.com/milvus-io/milvus/issues/52338) |
| Loon/Vortex rollback | Blocked by [#52340](https://github.com/milvus-io/milvus/issues/52340) |

PR #25 可以作为测试基础设施变更继续 review/merge。4 个被阻塞的 Milvus
能力在对应 Issue 修复并完成回归之前，不应被标记为 release-green。
