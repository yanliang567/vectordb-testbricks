# Upgrade/Rollback Availability Observation 实现计划

**目标：** 将升级/回滚期间的 availability 从 pressure failure 分类中独立出来，先生成可比较的观测指标，再用真实 2CU/cluster 运行校准硬 SLO 阈值。

**架构：** 继续复用现有 pressure daemon 和 `pressure-summary.json`。`common/pressure_maintenance.py` 汇总可解析 pressure result，按 rollout window 和 steady state 分组；三个 WorkflowTemplate 写入统一结构；最终报告单独展示 availability。现阶段不改变 gate pass/fail。

## 指标契约

每个 scope 输出：

- `sample_count` / `incomplete_sample_count` / `complete` / `calibration_eligible`
- `operations_total` / `operations_succeeded` / `requests_failed`
- `success_rate`
- `failed_sample_count`
- `impacted_bricks`
- `first_failure_at` / `last_failure_at` / `failure_span_sec`

scope 包括：

- `overall`：全部可解析 pressure result。
- `steady_state`：不与 rollout window 重叠的 result slice。
- `rollout_windows`：`upgrade-rollout`、`post-upgrade-config-rollout` 和 `rollback-rollout`。

pending、missing 和 unreadable attempt 会转换为无 metrics、无时间戳的占位样本。缺少 result 时间戳的样本不归入 steady state 或 rollout window，并通过 `unassigned_sample_count` 单独暴露。进程级失败即使没有 request metrics，也会增加 `failed_sample_count` 和 `incomplete_sample_count`。零样本或存在不完整样本时 `complete=false`、`calibration_eligible=false`，不得用于后续阈值校准。

`measurement=overlapping_pressure_result_slices` 表明统计按 result slice 与窗口是否重叠归属。单个 slice 跨越窗口边界时，其操作计数整体计入该窗口，因此该指标适合趋势比较和阈值校准，不宣称请求级精确 outage duration。

## 门禁策略

第一阶段固定输出：

```json
{
  "mode": "observational",
  "gate_enforced": false
}
```

现有 correctness、serviceability 和 strict pressure gate 保持不变。availability 失败不能被该字段自动忽略，pressure classifier 仍按原规则决定 workflow 最终状态。

## 验收

1. standalone 2.6、standalone 3.0 和 cluster 模板输出相同 availability 结构。
2. rollout 与 steady-state pressure operation 分开统计。
3. 缺失 `operations_total` 的 result 被计入 `incomplete_sample_count`。
4. 最终 Markdown 报告展示 overall、steady-state 和逐 rollout 指标。
5. 全量 pytest、Ruff、Argo offline lint 和 shell syntax check 通过。

## 后续阶段

1. 运行 Woodpecker 2CU HA、cluster target-only 和 cluster JSON Shredding gate，收集至少三轮基线。
2. 评估 pressure slice 长度对 failure span 的误差。
3. 确定 `min_success_rate`、`max_failure_span_sec` 和 incomplete sample 策略。
4. 仅对 HA promoted gate 启用硬 availability SLO。

## 校准状态

2026-08-06 已完成前三条场景的首轮 4am 校准，结果记录在
`milvus_client/docs/reports/2026-08-06-upgrade-rollback-availability-calibration.md`。

- 三条 run 的样本均完整且可用于校准。
- Woodpecker 2CU HA run 通过；target-only 和 JSON Shredding run 因 strict
  pressure 捕获到非连接类或 readiness 后持续的服务错误而失败。
- 已给出仅面向 Woodpecker 2CU HA 的候选阈值，但在重复 2CU run 建立方差前
  保持 `gate_enforced=false`。
- 现有 10 秒 result slice 会把跨越窗口边界的整段操作归入 rollout，且阻塞
  RPC 可让实际 slice 显著超过 10 秒；硬门禁前仍需重复运行或增加时间桶指标。

2026-08-07 使用固定 `v3.0.0` baseline digest 和两个不同 3.0 target build
重复运行 Woodpecker 2CU HA gate。两次都在 rollback readiness 之后因
`channel tsafe stalled` 未恢复而失败，未形成可校准样本；详见
`milvus_client/docs/reports/2026-08-07-upgrade-rollback-availability-repeat-calibration.md`
和 `milvus-io/milvus#52297`。硬 availability SLO 继续保持关闭。
