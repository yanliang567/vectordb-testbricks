# Milvus #52341 / Woodpecker 0.1.38 升级回滚验证报告

## 结论

在 QA 4am 集群上，固定 Woodpecker server `v0.1.38`，连续完成三次以下完整路径：

`Milvus v3.0.0 -> 3.0-20260828-8bd63b88 -> Milvus v3.0.0`

三次有效 workflow 均为 `Succeeded 63/63`。rollback 后 baseline 与 forward collection 的 serviceability、数据完整性、索引、schema feature、持续 DML/DQL 和严格压力切片全部通过；未出现 #52341 相关的 reader state 丢失或 tSafe 永久卡住。

因此，在本报告限定的镜像、Woodpecker 版本、九个 rollback-safe 2.6 schema 和压力模型下，**#52341 未复现，Woodpecker 0.1.38 reader-recovery 修复验证通过**。本次未保留 Milvus 环境；Argo workflow 历史和 artifacts 保留。

基于该证据，manifest 将限定为 Woodpecker v0.1.38 与九 schema 矩阵的 #52341 合同提升为 `gate / supported_with_config_constraints / release-gate-eligible=true`。两个受独立问题 #52768 影响的 nested scalar AutoIndex schema 不属于该受支持合同，继续由专门限制场景跟踪。

## 固定输入

- scenario：`cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`
- base/rollback：`harbor.milvus.io/milvusdb/milvus:v3.0.0@sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`
- target：`harbor.milvus.io/milvusdb/milvus:3.0-20260828-8bd63b88@sha256:61479222e9229d88df0decbba388a07a4890e212573c7a739f7e9a5a7212affb`
- Woodpecker server：`harbor.milvus.io/milvusdb/woodpecker:v0.1.38@sha256:bdea08758377fea309c18087334c63d20e26ba0940a4d63369bf7794f5f2060e`
- target 源码依赖：Woodpecker client `v0.1.38`
- deploy profile：`cluster-woodpecker-v0-1-38-1cu.yaml`
- baseline matrix：`schema_matrix_2_6_woodpecker_reader_recovery.yaml`，保留九个 schema，仅隔离 #52768 的两个 nested scalar AutoIndex schema
- pressure：`search_pressure query_pressure query_iterator_scan count_pressure upsert_pressure delete_pressure mixed_rw_pressure`
- `keep-milvus=true`；每轮取证后按 workflow ownership 精确清理通过环境
- Milvus 全阶段 log level：`debug`

## 代码与模板改动

验证分支：`test/validate-52341-wp-0-1-38`

最终 E2E revision：`c862c881708466c065eefd03938909ec85a1b81c`

1. 增加固定 Woodpecker server `v0.1.38@sha256:bdea08758377fea309c18087334c63d20e26ba0940a4d63369bf7794f5f2060e` 的 cluster deploy profile；release gate 解析会拒绝未固定 digest 的显式依赖镜像。
2. 增加由 2.6 source matrix 组合出的九 schema reader-recovery matrix，并为 schema loader 增加严格的 `source_matrix` / `include_schemas` 支持。
3. 更新 #52341 场景，使用专用 profile/matrix，明确与 #52768 隔离；三次验证通过后将该受限合同提升为 release gate。
4. 为四类升级/回滚模板的 Python dependency bootstrap 增加五次退避重试。重试仅发生在业务模块启动前。
5. 为三个升级/回滚模板增加 `idempotent-run-brick` 与 `optional-idempotent-run-brick`：只读的 precheck、serviceability、data/index validation 使用 `OnError`；create/seed/phase DML/schema feature/schema evolution/drop 等可能写入的操作无 Argo retry。`validate_schema_features` 包含 `entity_ttl` insert/flush/delete，因此在 review 修正后明确使用非重试 wrapper。
6. 两次最终有效验证使用 QA `milvus-cluster-upgrade-rollback` generation 33；无 `templateDefaults`。该验证版本有 retry 的 template 仅为：
   - `deploy-milvus`
   - `wait-milvus-ready`
   - `patch-milvus-image`
   - `patch-milvus-config`
   - `idempotent-run-brick`
   - `optional-idempotent-run-brick`
7. 验证完成后，live WorkflowTemplate 已恢复到 `origin/main@4f2360d8589d43464888ed775f90a0cf570d6c8b`，generation 34；无 `templateDefaults`，仅已合并的 `deploy-milvus`、`wait-milvus-ready`、`patch-milvus-image`、`patch-milvus-config` 保留 retry，未留下未评审的集群漂移。

最终静态验证：

- `python3 -m pytest milvus_client/tests -q`：`615 passed`
- `test_argo_template.py`：`127 passed`
- Ruff check / format check：通过
- cluster、standalone 2.6、standalone 3.0、compatibility Argo lint：通过
- server-side dry-run / apply：通过
- `git diff --check`：通过

## 三次有效结果

| 轮次 | Workflow | Revision / template | 时间（UTC） | rollback serviceability | forward rollback serviceability | 结果 |
| --- | --- | --- | --- | ---: | ---: | --- |
| 1 | [c30json-vs8xw](https://argo-workflows.zilliz.cc/workflows/qa/c30json-vs8xw) | `4587ededd2bdb2fc25a24a6a4e3652c8d0d8d6a9` / gen 31 | 13:00:56–14:07:41 | 8s | 9s | Succeeded 63/63 |
| 2 | [c30json-9ccfq](https://argo-workflows.zilliz.cc/workflows/qa/c30json-9ccfq) | `c862c881708466c065eefd03938909ec85a1b81c` / gen 33 | 16:38:04–17:46:18 | 21s | 20s | Succeeded 63/63 |
| 3 | [c30json-lzq4c](https://argo-workflows.zilliz.cc/workflows/qa/c30json-lzq4c) | `c862c881708466c065eefd03938909ec85a1b81c` / gen 33 | 17:49:29–18:58:28 | 21s | 21s | Succeeded 63/63 |

gen 33 的两次有效 workflow 中，所有 retry wrapper 均只有一个 attempt；没有用重试掩盖业务失败。

gen 33 验证时 schema feature task 曾通过 retry wrapper 调度，但所有 task 都只有一个 attempt，未发生重复 DML。PR review 后三个升级/回滚模板中的全部 schema feature task 已改回非重试 wrapper，并由静态分类测试保护。

## 压力结果

| 轮次 | samples | passed | rollout transient / excluded | overall operations | overall success | steady-state operations | steady-state success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 229 | 210 | 19 | 835,633 | 99.7875% | 582,755 | 100% |
| 2 | 224 | 206 | 18 | 767,504 | 99.8021% | 537,829 | 100% |
| 3 | 234 | 220 | 14 | 831,767 | 99.8555% | 571,074 | 100% |

各 rollout window 的 success rate：

| 轮次 | upgrade | post-upgrade config | rollback |
| --- | ---: | ---: | ---: |
| 1 | 99.4604% | 99.4784% | 99.4066% |
| 2 | 99.4477% | 99.3787% | 99.5123% |
| 3 | 99.3803% | 99.5017% | 100% |

rollout window 内的失败均为部署切换期连接瞬态，按既有 maintenance-window policy 归入 `excluded_failed`。三轮 `failed=0`，所有严格压力切片和 steady-state 均通过。

## Woodpecker Pod 连续性

Woodpecker StatefulSet 全程使用 `OnDelete`；每轮四个 Woodpecker Pod 的 UID 在 base、target、配置 rollout 和 rollback 前后保持不变。

| 轮次 | Pod UID（0 / 1 / 2 / 3） | restart count |
| --- | --- | --- |
| 1 | `d0f76eae-879f-42b9-8de2-2c5b04bd0ba8` / `6fc89597-7d27-4018-8526-a368d41ee364` / `20af9444-a61b-468b-b44b-2ec72a8dd90e` / `9dd4397d-619d-4913-8a15-71bbb948b87a` | 0 / 0 / 0 / 1 |
| 2 | `94fc6f85-e5ca-4ba8-823c-bf089ea8d4e7` / `6ca9756c-8b36-47a1-aa2d-a0fc9b54d787` / `34a4dfef-fa28-4abb-a43d-d8dc8429c543` / `08e3d498-c077-41f8-8cc0-3c3655179917` | 0 / 0 / 0 / 0 |
| 3 | `40e08e5f-e465-4734-b44f-cd3f0cabb6bc` / `09f95db1-2270-4f53-a63d-6b63e493694b` / `4a33811d-c4bc-4829-bdc4-b7fa6d7a97a0` / `0f978824-512b-433d-bed9-1d7f7fa2bfb6` | 0 / 0 / 0 / 0 |

第 1 轮 Woodpecker-3 的一次重启发生在 baseline 初始化前：首次 DNS 解析失败后，进程重试遇到 memberlist 端口占用；Pod 重启后加入四节点集群，后续所有 rollout 中 restart count 不再增加。它不是 #52341 症状。

## 关键日志

| 关键字 | 轮次 1 | 轮次 2 | 轮次 3 |
| --- | ---: | ---: | ---: |
| `no record extract` | 0 | 0 | 0 |
| `reader temp info not found` | 0 | 0 | 0 |
| `update reader info failed` | 0 | 0 | 0 |
| `tsafe stalled` | 0 | 0 | 0 |
| `delegator is not serviceable` | 22 | 15 | 22 |
| `channel not available` | 1 | 0 | 3 |
| 最终阶段校验完成后上述六类日志 | 0 | 0 | 0 |

`delegator/channel unavailable` 只发生在 phase DML 引起的 collection reload 窗口。每次 query view 随后推进，collection reload 在数秒内完成，严格压力与最终数据校验均通过；没有形成 #52341 的永久不可服务状态。

## 无效轮次与测试基础设施改进

以下 workflow 不计入三次有效重复：

| Workflow | 停止点 | 根因 | 处理 |
| --- | --- | --- | --- |
| [c30json-lz6kc](https://argo-workflows.zilliz.cc/workflows/qa/c30json-lz6kc) | disabled `schema-evolution-forward` | 业务容器 exit 0、artifact 已上传；Argo executor 创建 `WorkflowTaskResult` 时 QA Kubernetes etcd timeout | 记录为控制面无效轮次 |
| [c30json-v5src](https://argo-workflows.zilliz.cc/workflows/qa/c30json-v5src) | index validation Pod 创建 | Kubernetes `resource quota evaluation timed out`；namespace 实际无 ResourceQuota | 记录为控制面无效轮次 |
| [c30json-lqpnb](https://argo-workflows.zilliz.cc/workflows/qa/c30json-lqpnb) | forward schema validation bootstrap | `pypi.org` DNS 连续失败，业务模块未启动 | 增加 dependency bootstrap 退避重试 |
| [c30json-lb26d](https://argo-workflows.zilliz.cc/workflows/qa/c30json-lb26d) | upgrade serviceability Pod 创建 | Kubernetes `resource quota evaluation timed out`，业务模块未启动 | 拆分只读幂等 wrapper，使用 `OnError` |

控制面事件中同时存在其他 QA Milvus Pod 的 ConfigMap informer timeout，且 `/readyz` 恢复后正常，支持“集群级控制面瞬态”而非 Milvus/Woodpecker 回归的判断。所有无效轮次的 Argo 日志与事件证据保留，Milvus release/PVC 已清理。

## 后续建议

1. 以本报告作为 #52341 在 Woodpecker 0.1.38 上的关闭/解除限制证据。
2. #52768 继续保留独立 known-limitation tracker；不要把两个 nested scalar AutoIndex schema 静默并回主 release gate。
3. 合并本分支的 bootstrap 与只读幂等 retry 改动，使 cluster、standalone 2.6/3.0 和通用 compatibility 路径保持一致；schema feature brick 因包含 `entity_ttl` DML，不使用 Argo retry。
4. 合并 retry hardening 后，再从合并 commit 应用代码管理的 WorkflowTemplate；当前 live template 已恢复为 `origin/main` generation 34。
