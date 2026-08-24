# Known limitation: struct-array HYBRID scalar index cannot roll back to v3.0.0

- 日期：2026-08-24

## 结论

Milvus v3.0.0（已发布、不可变）在嵌套 struct-array 子字段的 HYBRID 标量索引上存在持久化格式兼容问题：其索引工厂把 HYBRID 子字段索引走了 sort 路径，落成缺少 `index_type` key 的 `milvus_packed_stlsort_index.v3`。该问题在 3.0.1 candidate（3.0 分支）已修复（build 侧 #52385 = #52360；load 侧 #52643 = #52642）。

因此，如果升级到 3.0.1 candidate 后集合里存在 struct-array HYBRID 标量索引，**无法回滚到 v3.0.0**（v3.0.0 基线缺少 load 侧容忍，读不了修复后的索引格式）。这是向前兼容限制，不是 3.0.1 的回归。

## 受影响的 gate 场景

以下两个场景回滚目标是 v3.0.0，且 schema 使用含 struct-array HYBRID 标量索引的 2.6 matrix，因此被标记为 `known_limitation`（`release_gate_eligible=false`），不作为发布阻塞项：

- `standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`（#52768）
- `cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline`（#52341）

## 现象（#52768 日志定位）

回滚到 v3.0.0 后，集合卡在 `recovering` 状态，最终 channel 不可用：

```
target_observer.go:475 "check delegator" isServiceable=false error="segment lacks[...]"
index/HybridScalarIndex.cpp:512 "hybrid index missing index_type meta, inferred physical type: STLSORT"
collection_observer.go:387 "failed to manual check current target, skip update load status"
utils/util.go:257 "loaded collection do not found any channel in target, may be in recovery: collection on recovering"
```

## 恢复条件

未来若出现一个新的、能读取修复后索引格式的 baseline（如 3.0.1 之后的版本作为回滚目标），可将这两个场景恢复为 `gate` / `supported`。

## 相关 issue

- milvus-io/milvus#52768（standalone）
- milvus-io/milvus#52341（cluster Woodpecker）
- milvus-io/milvus#52359 / #52620（原始 HYBRID 标量索引兼容问题）
