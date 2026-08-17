# Milvus 3.0.1 存储配置矩阵增强测试提案

**目标：** 在 3.0.1 发布前，补齐升级/回滚 gate 中 LoonFFI、Vortex、JSON
Shredding 三个存储开关组合的覆盖空缺，并利用「Vortex 依赖 LoonFFI」的约束把
非法组合在渲染期 fail-closed。

**背景依赖模型：** 三个开关的取值约束为

- 开启 JSON Shredding（J）不依赖 LoonFFI（L）和 Vortex（V）；
- 开启 LoonFFI（L）不依赖 J 和 V；
- 开启 Vortex（V）不依赖 J，但**开启 V 之前必须先开启 L**（记作 `V ⇒ L`）。

**架构：** 继续复用现有 3 条参数化 WorkflowTemplate 和
`upgrade_rollback_gates.yaml` 场景定义层。本轮只新增场景与两处渲染期 guard，
不新增 Workflow。

**技术栈：** Python、PyYAML、pytest、Argo WorkflowTemplate、Milvus
Operator/Helm、PyMilvus。

---

## 1. 合法状态空间

单阶段状态记为 `(J, L, V)`，其中 J=`jsonShreddingEnabled`、
L=`useLoonFFI`、V=`dataNode.storage.format=vortex`。在 `V ⇒ L` 约束下合法状态
共 6 个：

| 状态 | J | L | V | 含义 |
| --- | --- | --- | --- | --- |
| S0 | 0 | 0 | 0 | 全 legacy |
| S1 | 1 | 0 | 0 | 仅 JSON Shredding |
| S2 | 0 | 1 | 0 | 仅 LoonFFI（parquet 格式） |
| S3 | 1 | 1 | 0 | JSON + LoonFFI（parquet） |
| S4 | 0 | 1 | 1 | LoonFFI + Vortex |
| S5 | 1 | 1 | 1 | JSON + LoonFFI + Vortex |

非法状态：`V=1` 且 `L=0`，即 `(0,0,1)` 与 `(1,0,1)`。

## 2. 当前覆盖映射

按「该状态是否在某个 phase 被实际触达」标记：

| 状态 | 覆盖情况 |
| --- | --- |
| S0 | 已覆盖（所有 legacy 主路径） |
| S1 | 已覆盖（JSON Shredding 的 post-upgrade / rollback） |
| S2 | **未覆盖** |
| S3 | **未覆盖** |
| S4 | 部分（仅 candidate 的 target/rollback，base 仍是 S0） |
| S5 | 部分（仅 negative #14 的 target，无正向 gate） |

空缺集中在：(1) S2/S3 两条「LoonFFI 但非 Vortex」合法态；(2) S4/S5 的
「base 即 Vortex」与「正向全特性」。

## 3. 新增场景提案

所有场景依赖前置 `image_aliases.milvus-3-0-1`（pin 官方 multi-arch
manifest-list digest）。`(J,L,V)` 轨迹按 base → target → rollback 标注。

### P1 — Vortex baseline 全链路自洽（base=S4）

验证 3.0.1 自己写的 Vortex 数据经过升级/回滚后仍可读（#52340 契约的另一半）。

```yaml
- id: standalone-3-0-1-vortex-self-compat-upgrade-rollback
  mode: standalone
  classification: gate
  support_status: supported
  workflow_template_ref: standalone_3_0
  deploy_profile_ref: standalone
  schema_matrix_ref: 3.0_storage_v3
  forward_schema_matrix_ref: 3.0_storage_v3
  collection_prefix: qa_gate_301_vortex_self
  forward_workload_enabled: true
  schema_evolution_existing_enabled: false
  schema_evolution_forward_enabled: false
  rollback_forward_validation_enabled: true
  base:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: true          # S4
  target:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: true          # S4
  rollback:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: true          # S4
  validation_policy:
    data_integrity: strict
    serviceability: strict
    pressure_fail_on_error: true
    gate_allow_warning: false
```

cluster 版对称：`mode: cluster`、`workflow_template_ref: cluster`、
`deploy_profile_ref: cluster_woodpecker_1cu`、`submit_generate_name: c301vs-`。

### P2 — 3.0.0(legacy) → 3.0.1(开 Vortex) → 3.0.1(Vortex)

#52340 的真实用户路径：从 3.0.0 legacy 升级到 3.0.1 时开 Vortex，回滚目标
必须是 v3.0.1（guard 强制 rollback ≥ 3.0.1）。

```yaml
- id: standalone-3-0-0-to-3-0-1-vortex-enable-rollback
  mode: standalone
  classification: gate
  support_status: supported
  workflow_template_ref: standalone_3_0
  deploy_profile_ref: standalone
  schema_matrix_ref: rollback_safe_base
  forward_schema_matrix_ref: 3.0_storage_v3
  collection_prefix: qa_gate_300_to_301_vortex
  forward_workload_enabled: true
  rollback_forward_validation_enabled: true
  base:
    image_ref: milvus-3-0-baseline   # v3.0.0, S0
    vortex_enabled: false
  target:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: true          # S4
  rollback:
    image_ref: milvus-3-0-1        # 不能回 v3.0.0
    loon_ffi_enabled: true
    vortex_enabled: true          # S4
  validation_policy:
    data_integrity: strict
    serviceability: strict
    pressure_fail_on_error: true
    gate_allow_warning: false
```

> 风险：需确认「已落盘 legacy 段 + 新写 Vortex 段」混存是否受支持；若不支持，
> 本场景应降级为 negative。

### P3 — JSON Shredding × Vortex 正向（S5）

`(1,1,1)` 合法（V⇒L 满足、J 独立），直接作为 gate。

```yaml
- id: standalone-3-0-1-json-shredding-vortex-rollback
  mode: standalone
  classification: gate
  support_status: supported
  workflow_template_ref: standalone_3_0
  deploy_profile_ref: standalone
  schema_matrix_ref: 3.0_storage_v3
  forward_schema_matrix_ref: json_shredding
  collection_prefix: qa_gate_301_json_vortex
  forward_workload_enabled: true
  post_upgrade_config_toggle_enabled: true
  post_upgrade_json_shredding_enabled: true
  rollback_forward_validation_enabled: true
  base:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: true          # S4
  target:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: true          # S4
  rollback:
    image_ref: milvus-3-0-1
    json_shredding_enabled: true  # 回滚保持 J=1
    loon_ffi_enabled: true
    vortex_enabled: true          # S5
  validation_policy:
    data_integrity: strict
    serviceability: strict
    pressure_fail_on_error: true
    gate_allow_warning: false
```

### P4 — 仅 LoonFFI（parquet 格式）gate（S2）

把 LoonFFI 引擎的影响从 Vortex 中剥离开：验证 `L=1,V=0` 升级后能否安全回滚到
`L=0`。若产品确认 L=1 的 parquet 编码与 L=0 不互读，本场景会暴露该边界，届时
类比 Vortex 追加「LoonFFI 回滚安全版本」约束。

```yaml
- id: standalone-3-0-1-loon-ffi-parquet-rollback
  mode: standalone
  classification: gate
  support_status: supported
  workflow_template_ref: standalone_3_0
  deploy_profile_ref: standalone
  schema_matrix_ref: "3.0"
  collection_prefix: qa_gate_301_loon_parquet
  forward_workload_enabled: false
  rollback_forward_validation_enabled: false
  base:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: false
    vortex_enabled: false         # S0
  target:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: false         # S2
  rollback:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: false
    vortex_enabled: false         # S0
  validation_policy:
    data_integrity: strict
    serviceability: strict
    pressure_fail_on_error: true
    gate_allow_warning: false
```

### P5 — 回滚关闭 Vortex 的格式降级（negative）

同版本内把 Vortex 关掉，用户极可能误操作。两种降级形态（关 V 保留 L，或 V/L
全关）需至少选一种覆盖：

| 变体 | 轨迹 | 含义 |
| --- | --- | --- |
| P5a | `S4 → S2` | 关 Vortex、保留 LoonFFI |
| P5b | `S4 → S0` | 关 Vortex、同时关 LoonFFI |

```yaml
- id: standalone-3-0-1-vortex-disable-rollback-negative
  mode: standalone
  classification: negative
  support_status: unsupported
  workflow_template_ref: standalone_3_0
  deploy_profile_ref: standalone
  schema_matrix_ref: rollback_safe_base
  forward_schema_matrix_ref: 3.0_storage_v3
  collection_prefix: qa_negative_301_vortex_off_rb
  forward_workload_enabled: true
  base:
    image_ref: milvus-3-0-1
    vortex_enabled: false        # S0
  target:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: true
    vortex_enabled: true         # S4
  rollback:
    image_ref: milvus-3-0-1
    loon_ffi_enabled: false      # 或 true（对应 P5a）
    vortex_enabled: false        # S0 或 S2
  validation_policy:
    data_integrity: observe
    serviceability: observe
    pressure_fail_on_error: false
    gate_allow_warning: true
```

## 4. 代码增强（渲染期 guard）

### 4.1 V⇒L 校验（`milvus_client/common/gates.py`）

在 `_validate_vortex_compatibility_contract` 之前增加每 phase 的依赖校验：

```python
def _validate_vortex_loon_dependency(scenario: dict[str, Any]) -> None:
    for phase in ("base", "target", "rollback"):
        if scenario[phase].get("vortex_enabled") and not scenario[phase].get(
            "loon_ffi_enabled"
        ):
            raise ValueError(
                f"{scenario['id']}: {phase} Vortex requires LoonFFI "
                "(vortex_enabled=true implies loon_ffi_enabled=true)"
            )
```

并在 `validate_resolved_gate_scenario` 的
`_validate_vortex_compatibility_contract(scenario)` 调用处同时调用该函数。

### 4.2 回滚格式降级 guard（`gates.py`）

在 `_validate_vortex_compatibility_contract` 末尾追加：当 base/target 写入过
Vortex 数据、且回滚阶段 `vortex_enabled=false` 时，仅允许 `negative` 场景：

```python
    vortex_data_may_exist = any(
        scenario[phase].get("vortex_enabled", False) for phase in ("base", "target")
    )
    if (
        scenario.get("rollback_enabled", True)
        and vortex_data_may_exist
        and not scenario["rollback"].get("vortex_enabled", False)
        and scenario.get("classification") != "negative"
    ):
        raise ValueError(
            f"{scenario['id']}: Vortex data written before rollback cannot be "
            "read with vortex disabled; disabling Vortex at rollback requires "
            "an explicit negative scenario"
        )
```

### 4.3 渲染落盘 fail-closed（`milvus_client/common/deploy.py`）

在 `_storage_config` 中，`vortex_enabled=True` 但 `loon_ffi_enabled=False`
时直接抛错，避免写出缺失 `useLoonFFI` 的非法 CR：

```python
    if vortex_enabled and not loon_ffi_enabled:
        raise ValueError(
            "vortex storage requires useLoonFFI (Vortex depends on LoonFFI)"
        )
```

## 5. 优先级与分阶段

| 阶段 | 内容 | 分类 | 阻塞 3.0.1? |
| --- | --- | --- | --- |
| A | P1、P2（standalone/cluster）+ 现有 candidate 用 P1/P2 形态提升 | gate | 是 |
| A | P3 JSON×Vortex（standalone/cluster） | gate | 是 |
| A | 代码增强 4.1/4.2/4.3 | 代码 | 是（低成本，防非法配置） |
| B | P4 LoonFFI parquet、P5 Vortex 降级 negative | gate/negative | 建议 |
| C | S3（J+L+parquet）组合 | 待定 | 否 |

## 6. 验证命令

```bash
cd milvus-bricks

PYTHONPATH=. uv run pytest -q \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_argo_template.py

uvx ruff check milvus_client/common/gates.py milvus_client/common/deploy.py
uvx ruff format --check milvus_client/common/gates.py milvus_client/common/deploy.py

argo lint --offline argo/standalone-3-0-upgrade-rollback.yaml
argo lint --offline argo/cluster-upgrade-rollback.yaml

git diff --check
```

## 7. 完成标准

- `upgrade_rollback_gates.yaml` 新增 P1–P5（standalone/cluster 对称），promoted
  gate 数量相应更新，`test_upgrade_rollback_gates_manifest.py` 断言通过。
- 渲染期 `V=1 ⇒ L=1` 与「回滚格式降级仅限 negative」两条 guard 有单测覆盖。
- P1/P2/P3 真机跑通（含 `#52340` 复验），P4 明确给出「LoonFFI 是否独立存在
  回滚安全边界」的结论，P5 记录格式降级边界证据。
