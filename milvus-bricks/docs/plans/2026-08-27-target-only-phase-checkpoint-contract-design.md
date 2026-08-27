# Target-Only Phase Checkpoint 合同设计

**日期：** 2026-08-27
**状态：** Approved

## 背景

standalone 3.0 v10/v4 canary 在 PR #42 修正后通过了 forward index build/query/reload，但在回滚后的 phase checkpoint 验证中失败。目标阶段在全局 index engine 10/4 配置下创建的 `_after_upgrade` collection 带有目标格式索引；虽然它使用 rollback-safe schema matrix，索引 artifact 本身仍不是 baseline-compatible。现有 target-only 编译只删除显式 forward matrix collections，遗漏了 phase DML-DQL 创建的目标阶段 collections。

## 目标

- 将现有 `index_engine_contract.mode` 应用于所有目标阶段新建的索引 artifact。
- target-only 场景继续完整验证目标阶段 phase DML-DQL，但不要求这些新 collections 回滚兼容。
- 回滚后仍严格验证 baseline existing collections 的目标阶段写删结果，并继续运行 rollback 阶段 DML-DQL。
- round-trip 和无 index contract 场景保持当前跨回滚验证行为。

## 非目标

- 不关闭 target-only 场景的 phase DML-DQL。
- 不跳过基础集群回滚。
- 不把 load timeout 降级为 warning。
- 不改变 schema matrix 的 `rollback_safe` 含义；该标记描述 schema 能力，不覆盖创建时选择的 index engine artifact 格式。

## 方案设计

复用已有 workflow 参数 `index-engine-contract-mode`，不增加可独立漂移的布尔开关。三套 WorkflowTemplate 在回滚前增加 `drop-phase-new-collections`：仅当 mode 为 `target_only` 时，按 base schema matrix 删除 `${collection-prefix}_after_upgrade` collections。该节点位于 forward/schema evolution 验证之后、回滚前观察之前，保证目标阶段验证完成后才删除。

rollback phase validator 新增 `--phase-checkpoint-new-collections-contract`，接受 `none`、`target_only`、`round_trip`。`none` 与 `round_trip` 保持现状：checkpoint 的 existing/new 两组都必须 reload/query。`target_only` 仍严格校验 checkpoint 的结构、schema 覆盖和 oracle，但运行时要求 new group collections 已不存在；任何残留都失败。existing group 仍 reload/query，随后继续 baseline existing DML-DQL 和 rollback 新 collection DML-DQL。carried collection DML 只在非 target-only 模式运行。

## 错误处理和证据

target-only collection 在回滚后仍存在时，报告 `PHASE_CHECKPOINT_TARGET_ONLY_COLLECTION_PRESENT`，不静默跳过。metrics 记录合同模式、目标阶段新 collection 总数、确认不存在数量和意外存在数量。round-trip 中 reload 失败继续使用现有 `PHASE_CHECKPOINT_RELOAD_FAILED` fail-closed 行为。

## 替代方案

- 关闭 phase DML-DQL：实现简单，但损失目标与回滚阶段大量 DML/DQL 覆盖，拒绝。
- 整个 workflow 不回滚：与既定“feature target-only、baseline path 仍 round-trip”合同冲突，拒绝。
- 增加多个 drop/validate 布尔参数：表达直接但容易与 manifest contract mode 漂移，拒绝。

## 测试策略

- validator unit tests 覆盖 target-only absent/present、round-trip 现有行为及 main carried-DML 分支。
- manifest/template tests 覆盖四个 target-only index 场景、三套 WorkflowTemplate 的 drop task、参数传递和依赖顺序。
- 全量 pytest、CI Ruff check/format、三套 Argo offline lint。
