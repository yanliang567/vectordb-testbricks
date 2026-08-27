# Index Gate Target-Only 与 Debug 日志设计

**日期：** 2026-08-27
**作者：** Codex
**状态：** Approved

## 背景

Milvus v3.0.0 已发布且不会包含 #52767 的 growing sparse index 修复。当前 index-v10/v4 与 index-v11/v4 升级回滚场景在 base 阶段就创建对应 sparse index matrix，因此会在升级到候选镜像之前触发 v3.0.0 的已知 SIGSEGV，阻断目标版本和其他升级回滚链路的验证。同时，现有工作流没有统一显式声明 Milvus 日志级别，失败后排查信息不足。

## 目标

- v3.0.0 base 和 rollback 阶段只运行稳定、回滚安全的 schema matrix。
- v10/v4、v11/v4 index matrix 只在 target 阶段创建、写入、build/load/search，并在 rollback 前删除。
- standalone 与 cluster 的所有升级回滚工作流默认使用 `log.level=debug`，同时保留显式参数以便必要时覆盖。
- 报告和渲染摘要能记录实际日志级别。

## 非目标

- 不把 #52767 转成 expected failure，也不在主 gate 内主动触发进程崩溃。
- 不验证 target-only index 集合回滚到 v3.0.0 后的可读性。
- 不改变普通数据集的严格升级、回滚、压力与数据完整性判断。

## 方案设计

四个 index gate 改用 `rollback_safe_base` 作为基础 matrix，并把原 v10/v11 matrix 设置为 `forward_schema_matrix_ref`。仅 target phase 设置 index engine version；base/rollback 使用默认版本。场景启用 forward workload，关闭 rollback forward validation，并在 rollback 前显式删除 forward collections。这样已知缺陷不会进入主 gate，target 上的 schema、seed、index compatibility 和 feature validation仍保持 fail-closed。

所有 workflow 增加 `milvus-log-level` 参数，默认值来自 gate manifest 的 `milvus_log_level: debug`。Operator CR renderer 与 Helm values renderer统一接收 `--log-level`，将其写入 `spec.config.log.level` 或 `extraConfigFiles.user.yaml`。部署摘要及 workflow 环境快照记录该值；镜像升级和配置切换采用 merge/Helm 重新渲染时持续传递同一参数，防止阶段间回落到 info。

## 替代方案

- 在 base 阶段把崩溃标成 xfail：进程级 SIGSEGV 会污染同一环境，无法可靠继续。
- 拆成独立 negative workflow：适合保留缺陷回归，但不能替代 target release gate；可后续单独增加。
- 只从 matrix 删除 SINDI/BLOCK_MAX：会削弱目标版本覆盖，且容易忘记恢复。

## 测试策略

- Manifest/renderer 测试验证四个 index gate 的 base、target、rollback matrix 和 index version 边界。
- Deploy renderer 测试验证 Operator 与 Helm 都输出 `log.level: debug`，并验证显式覆盖。
- Argo template 测试验证三个模板声明、传递并记录 `milvus-log-level`。
- 运行 YAML 解析、相关 pytest，以及可用时的 `argo lint`。
