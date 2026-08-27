# Index Engine 兼容性合同 Manifest 实现计划

**目标：** 将 v10/v4、v11/v4 的 target-only / round-trip 兼容语义建模为 `upgrade_rollback_gates.yaml` 中可编译、可校验、可报告的结构化合同，同时保证当前 26 条升级/回滚路径的执行行为除新增合同元数据外保持不变。

**架构：** 将 gate manifest 升级为 v2，并让 `index_engine_contract` 成为 index engine 场景的单一高层事实来源。`milvus_client.common.gates` 在解析期把合同编译为现有 schema matrix、phase index version 和 forward/rollback 开关；Argo DAG 继续消费现有低层参数，只新增受保护的合同元数据参数和报告字段。Round-trip 合同必须由与实际 immutable image 匹配的 baseline capability qualification 授权，禁止仅按语义版本推断支持能力。

**技术栈：** Python 3、PyYAML、pytest、Argo WorkflowTemplate YAML、Argo CLI offline lint、Ruff。

---

## 1. 背景与约束

当前四个 index engine 场景通过一组彼此独立的字段表达 target-only 合同：

```yaml
schema_matrix_ref: rollback_safe_base
forward_schema_matrix_ref: 3.0_index_v10_v4
forward_workload_enabled: true
rollback_forward_validation_enabled: false
drop_forward_before_rollback_enabled: true
target:
  target_vec_index_version: 10
  target_scalar_index_version: 4
```

这组字段当前是正确的，但 baseline 从 v3.0.0 升级到 v3.0.1 后，若能力已经 qualified，需要同时修改 matrix、三个 phase 的 index version 和多个 forward/rollback 开关。漏改任意字段都可能让报告声称 round-trip，实际却仍在 rollback 前删除 target 数据，或让不支持能力的 baseline 提前创建专项 index。

本计划遵守以下硬约束：

1. 合同只覆盖 `IndexEngineV10V4` 和 `IndexEngineV11V4`，第一版不泛化到 Vortex、LoonFFI 或 JSON Shredding。
2. 合同编译结果继续使用现有 Argo DAG，不新增第二套生命周期流程。
3. 当前四个 index 场景迁移后，展开得到的执行参数必须与提交 `d4083f4` 完全一致。
4. 其余 22 个注册场景不改变 schema matrix、镜像、phase feature flag、forward/rollback 开关或必需验证 brick。
5. 正式 round-trip gate 必须绑定 immutable baseline/rollback image 的 qualification 证据；版本号本身不是能力证据。
6. 旧 renderer 不得静默忽略新合同，因此 manifest 版本升级为 `"2"`。

## 2. 方案选择

### 2.1 采用：结构化合同并在解析期编译

```yaml
index_engine_contract:
  mode: target_only
  capability: IndexEngineV10V4
  matrix_ref: 3.0_index_v10_v4
  rollback_safe_matrix_ref: rollback_safe_base
  vector_version: 10
  scalar_version: 4
  rationale:
    issue: https://github.com/milvus-io/milvus/issues/52767
    baseline_support: unsupported
```

优点：合同是单一事实来源；低层参数可以 fail-closed 派生；能够在正式提交、Argo runtime precheck 和最终报告中保持一致。

### 2.2 不采用：只增加一个 enum，低层字段继续手写

只增加 `index_compatibility_scope: target_only|round_trip` 改动较小，但仍然允许 scope 与 phase version、drop/rollback 开关互相矛盾，不能解决漏改问题。

### 2.3 不采用：根据 baseline version 自动推断

`version >= 3.0.1` 不能证明 v10/v4 和 v11/v4 都已修复，也不能证明 standalone/cluster 都通过。该方案会把发布版本号误当成 capability evidence，因此禁止引入 `auto` 模式。

## 3. Manifest v2 Schema

### 3.1 合同允许值

第一版只允许：

```text
target_only
round_trip
```

没有 `index_engine_contract` 的普通场景在 resolved/rendered metadata 中统一表示为：

```text
mode=none
capability=none
qualification_status=not_applicable
```

使用显式 sentinel 而不是空字符串，因为 `render_argo_parameters()` 当前会过滤空值。

### 3.2 Target-only 示例

```yaml
version: "2"

capability_qualifications: {}

scenarios:
  - id: standalone-3-0-index-v10-v4-upgrade-rollback
    mode: standalone
    classification: gate
    support_status: supported
    workflow_template_ref: standalone_3_0
    deploy_profile_ref: standalone
    collection_prefix: qa_gate_30_index_v10_v4

    index_engine_contract:
      mode: target_only
      capability: IndexEngineV10V4
      matrix_ref: 3.0_index_v10_v4
      rollback_safe_matrix_ref: rollback_safe_base
      vector_version: 10
      scalar_version: 4
      rationale:
        baseline_support: unsupported
        issue: https://github.com/milvus-io/milvus/issues/52767

    base:
      image_ref: milvus-3-0-baseline
    target:
      image_ref: milvus-3-0-latest
    rollback:
      image_ref: milvus-3-0-baseline

    schema_evolution_existing_enabled: false
    schema_evolution_forward_enabled: false
    validation_policy:
      data_integrity: strict
      serviceability: strict
      pressure_fail_on_error: true
      gate_allow_warning: false
```

### 3.3 Future round-trip 示例

该示例只用于 compiler/validator 测试；在 v3.0.1 qualification 完成前不注册为正式 3.0.2 gate。

```yaml
capability_qualifications:
  milvus-3-0-1:
    immutable_image: harbor.milvus.io/milvusdb/milvus:v3.0.1@sha256:<digest>
    capabilities:
      IndexEngineV10V4:
        status: passed
        evidence:
          standalone: argo://qa/<standalone-qualification-workflow>
          cluster: argo://qa/<cluster-qualification-workflow>

scenarios:
  - id: standalone-3-0-1-to-3-0-2-index-v10-v4-upgrade-rollback
    mode: standalone
    classification: gate
    support_status: supported
    workflow_template_ref: standalone_3_0
    deploy_profile_ref: standalone
    collection_prefix: qa_gate_301_to_302_index_v10_v4

    index_engine_contract:
      mode: round_trip
      capability: IndexEngineV10V4
      matrix_ref: 3.0_index_v10_v4
      vector_version: 10
      scalar_version: 4

    base:
      image_ref: milvus-3-0-1
    target:
      image_ref: milvus-3-0-2-candidate
    rollback:
      image_ref: milvus-3-0-1

    schema_evolution_existing_enabled: false
    schema_evolution_forward_enabled: false
    validation_policy:
      data_integrity: strict
      serviceability: strict
      pressure_fail_on_error: true
      gate_allow_warning: false
```

### 3.4 Qualification 约束

`capability_qualifications` 以 manifest image alias 为入口，同时记录 resolved immutable image。Round-trip 校验必须同时满足：

1. `base.image_ref` 和 `rollback.image_ref` 都有对应 qualification。
2. qualification 中的 `immutable_image` 必须包含 `@sha256:<64 hex>`，并与 alias 当前解析出的 image 完全相同。
3. 对应 capability 的 `status` 为 `passed`。
4. standalone 场景至少有 standalone evidence；cluster 场景至少有 cluster evidence；evidence 必须是稳定的 `argo://` workflow reference 或 `https://` run URL。
5. runtime image override 若不等于 qualified immutable image，正式 registered gate 渲染失败。
6. v10 qualification 不得授权 v11，反之亦然。

## 4. 合同编译规则

### 4.1 Target-only

| 派生字段 | 值 |
| --- | --- |
| `schema_matrix` | `rollback_safe_matrix_ref` |
| `forward_schema_matrix` | `matrix_ref` |
| `forward_workload_enabled` | `true` |
| `rollback_enabled` | `true` |
| `rollback_forward_validation_enabled` | `false` |
| `drop_forward_before_rollback_enabled` | `true` |
| base vec/scalar version | `-1/-1` |
| target vec/scalar version | contract version |
| rollback vec/scalar version | `-1/-1` |
| qualification status | `unsupported` |

### 4.2 Round-trip

| 派生字段 | 值 |
| --- | --- |
| `schema_matrix` | `matrix_ref` |
| `forward_schema_matrix` | `matrix_ref` |
| `forward_workload_enabled` | `true` |
| `rollback_enabled` | `true` |
| `rollback_forward_validation_enabled` | `true` |
| `drop_forward_before_rollback_enabled` | `false` |
| base vec/scalar version | contract version |
| target vec/scalar version | contract version |
| rollback vec/scalar version | contract version |
| qualification status | `passed` |

Round-trip 同时保留两类数据：baseline 创建的专项 collection，以及 target 创建的 forward collection。现有 phase DML/DQL checkpoint 还会验证 target 创建的 phase collection 在 rollback 后能够 reload/search/query。

### 4.3 禁止双重事实来源

有 `index_engine_contract` 时，raw scenario 中禁止显式出现以下字段：

```text
schema_matrix / schema_matrix_ref
forward_schema_matrix / forward_schema_matrix_ref
forward_workload_enabled
rollback_enabled=false
rollback_forward_validation_enabled
drop_forward_before_rollback_enabled
base|target|rollback.target_vec_index_version
base|target|rollback.target_scalar_index_version
```

发现冲突时直接报错，不采用“合同优先覆盖”或“要求值相等”的宽松策略。

## 5. Fail-closed 校验

### 5.1 Raw manifest 校验

- manifest version 必须是 `"2"`。
- `index_engine_contract` 必须是 mapping。
- mode、capability、matrix ref、version 字段必须齐全且类型正确。
- vector/scalar version 必须是非负整数，bool 不视为整数。
- target-only 必须提供 `rollback_safe_matrix_ref` 和 rationale。
- round-trip 不允许 `rollback_safe_matrix_ref`，且必须能找到 qualification。
- promoted gate 必须保持 strict pressure、no warning、index compatibility 和 phase DML/DQL validation。
- 禁止合同与派生低层字段共存。

### 5.2 Resolved scenario 校验

- capability 必须存在于 `capability_catalog.yaml`。
- `matrix_ref` 解析出的所有 schema 必须声明合同 capability。
- matrix `validator_params.target_vec_index_version` / `target_scalar_index_version` 必须与合同完全一致。
- target-only 的 rollback-safe matrix 所有 schema 都必须是 `compat_mode: rollback_safe`，且不得声明任何 `IndexEngine*` capability。
- round-trip 的 base/forward matrix 必须相同。
- round-trip 的 `rollback_incompatible_specs()` 必须为空。
- round-trip base/rollback actual immutable image 必须与 qualification 一致。
- 展开的 phase config 和 forward/rollback flag 必须满足第 4 节真值表。

### 5.3 Runtime 注册参数校验

新增受保护参数：

```text
index-engine-contract-mode
index-engine-capability
index-engine-qualification-status
```

它们不得加入 `REGISTERED_SCENARIO_MUTABLE_PARAMETERS`。同时把三项安全默认值加入 `UNREGISTERED_SCENARIO_METADATA`，避免 unregistered workflow 声称自己是 round-trip gate。

## 6. 对全部升级/回滚路径的影响审计

当前 manifest 有 26 个场景：21 个 gate、2 个 candidate、2 个 known limitation、1 个 negative。

### 6.1 直接语义迁移：4 个 index 场景

以下场景改为 `index_engine_contract.mode=target_only`，但编译结果必须和当前行为相同：

```text
standalone-3-0-index-v10-v4-upgrade-rollback
standalone-3-0-index-v11-v4-upgrade-rollback
cluster-3-0-index-v10-v4-upgrade-rollback
cluster-3-0-index-v11-v4-upgrade-rollback
```

必须保持：base/rollback 使用 rollback-safe matrix；只有 target 配置 10/4 或 11/4；forward collection 在 rollback 前删除；rollback forward validation 关闭。

### 6.2 无执行语义变化：22 个非 index 场景

| 场景组 | 数量 | 是否修改场景执行字段 | 一致性处理 |
| --- | ---: | --- | --- |
| 2.6 → 3.0 → 2.6 baseline/target-only（standalone + cluster） | 4 | 否 | 合同 metadata=`none/not_applicable` |
| 3.0 baseline core gate（standalone + cluster） | 2 | 否 | 保持 3.0 matrix 与 forward rollback validation |
| JSON Shredding known limitation（standalone + cluster） | 2 | 否 | 保持现有 JSON matrix 与 classification |
| Woodpecker 2CU HA | 1 | 否 | 保持 topology、pressure、schema matrix |
| Vortex pre-release candidate（standalone + cluster） | 2 | 否 | 保持 candidate lock 和 Vortex guard |
| 2.6 rollback unsafe negative | 1 | 否 | 保持 negative classification 与 bypass guard |
| 3.0.1 storage/Vortex/JSON/LoonFFI gates | 10 | 否 | 保持现有 storage compatibility 规则 |

这些场景不增加 `index_engine_contract` block，也不修改 `index_compatibility_validation_enabled`。后者仍是所有 matrix 通用的 index metadata/search/query brick 开关，不等同于 index engine 生命周期合同。

### 6.3 所有 26 个场景都需要的一致性改动

虽然只有四个场景改变 raw manifest 表达，以下基础设施必须对全部场景保持一致：

1. manifest header 升级为 v2。
2. renderer 总是输出三个合同 metadata 参数；非 index 场景输出 sentinel。
3. registered runtime drift 校验保护这三个参数。
4. env snapshot、flow summary 和 final report 始终记录合同 metadata。
5. 所有场景的既有低层执行参数必须通过语义快照回归保持不变。

### 6.4 Argo WorkflowTemplate 影响

修改：

```text
argo/standalone-2-6-upgrade-rollback.yaml
argo/standalone-3-0-upgrade-rollback.yaml
argo/cluster-upgrade-rollback.yaml
```

三份模板都声明默认值：

```yaml
- name: index-engine-contract-mode
  value: none
- name: index-engine-capability
  value: none
- name: index-engine-qualification-status
  value: not_applicable
```

并同步到 runtime parameter map、env snapshot、flow summary、cleanup fallback 和 final report command。DAG task、`when` 条件和 patch/deploy 行为不直接读取合同 metadata，仍然只消费 compiler 生成的现有低层参数，因此其他路径执行顺序不变。

`argo/upgrade-rollback-compatibility.yaml` 不使用 gate manifest，也不负责部署这些注册 release gate，因此不修改。

### 6.5 Schema/capability 文件影响

以下文件第一版只作为校验输入，不改变内容：

```text
milvus_client/manifests/capability_catalog.yaml
milvus_client/manifests/feature_inventory.yaml
milvus_client/manifests/schema_matrix_3_0_index_v10_v4.yaml
milvus_client/manifests/schema_matrix_3_0_index_v11_v4.yaml
```

现有 capability ID 和 matrix validator params 已足够支持合同交叉校验。只有发现现有声明不一致时才单独修正，不能在合同迁移中顺带改变 index coverage。

## 7. 实施任务

### 任务 1：建立 26 条路径的语义回归护栏

**文件：**
- 创建：`milvus_client/tests/fixtures/upgrade_rollback_execution_paths_v1.yaml`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py:1-1200`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py:1-620`

**步骤 1：写入当前执行路径基线**

fixture 对每个 scenario 保存以下 resolved/rendered 字段：

```yaml
workflow_template: ...
deploy_profile: ...
schema_matrix: ...
forward_schema_matrix: ...
forward_workload_enabled: ...
rollback_enabled: ...
rollback_forward_validation_enabled: ...
drop_forward_before_rollback_enabled: ...
base_image: ...
target_image: ...
rollback_image: ...
base_version: ...
target_version: ...
rollback_version: ...
base_target_vec_index_version: ...
target_target_vec_index_version: ...
rollback_target_vec_index_version: ...
base_target_scalar_index_version: ...
target_target_scalar_index_version: ...
rollback_target_scalar_index_version: ...
index_compatibility_validation_enabled: ...
phase_dml_dql_validation_enabled: ...
```

**步骤 2：写路径不变测试**

```python
def test_manifest_v2_contract_migration_preserves_existing_execution_paths():
    expected = yaml.safe_load(EXECUTION_PATH_FIXTURE.read_text())
    actual = execution_path_signatures(load_gate_manifest(GATES))
    assert actual == expected
```

合同 metadata 不进入 v1 execution signature，避免把预期新增字段误判为行为变化。

**步骤 3：运行基线测试**

运行：

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py
```

预期：在修改 manifest 前 PASS，并固化当前 26 条路径。

**步骤 4：提交护栏**

```bash
git add milvus_client/tests/fixtures/upgrade_rollback_execution_paths_v1.yaml \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py
git commit -m "test: snapshot upgrade rollback execution paths"
```

### 任务 2：为 manifest v2 编写失败测试

**文件：**
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py:1000-1200`
- 修改：`milvus_client/common/gates.py:23-60`

**步骤 1：添加合同 schema 测试**

覆盖：

```python
def test_target_only_contract_expands_expected_execution_flags(): ...
def test_round_trip_contract_expands_expected_execution_flags(): ...
def test_manifest_rejects_unknown_index_engine_contract_mode(): ...
def test_manifest_rejects_contract_and_derived_fields_together(): ...
def test_manifest_rejects_matrix_capability_mismatch(): ...
def test_manifest_rejects_matrix_validator_version_mismatch(): ...
def test_round_trip_contract_requires_qualified_base_image(): ...
def test_round_trip_contract_requires_mode_specific_evidence(): ...
def test_v10_qualification_does_not_authorize_v11(): ...
def test_runtime_image_override_must_match_qualification(): ...
```

**步骤 2：运行并确认 red-state**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  -k 'contract or qualification'
```

预期：FAIL，错误包含缺少 compiler、manifest version 仍为 v1 或合同未被校验。

**步骤 3：提交失败测试**

```bash
git add milvus_client/tests/test_upgrade_rollback_gates_manifest.py
git commit -m "test: define index engine contract semantics"
```

### 任务 3：实现合同 parser、compiler 和 qualification guard

**文件：**
- 修改：`milvus_client/common/gates.py:23-280`
- 修改：`milvus_client/common/gates.py:436-590`
- 修改：`milvus_client/common/gates.py:620-730`

**步骤 1：增加常量和合同默认 metadata**

```python
INDEX_ENGINE_CONTRACT_MODES = {"target_only", "round_trip"}
NO_INDEX_ENGINE_CONTRACT = {
    "mode": "none",
    "capability": "none",
    "qualification_status": "not_applicable",
}
INDEX_ENGINE_DERIVED_FIELDS = {...}
```

**步骤 2：增加 raw contract validator**

实现：

```python
def _validate_raw_index_engine_contract(
    manifest: dict[str, Any], scenario: dict[str, Any], source: str
) -> None:
    ...
```

它负责类型、允许值、required fields、derived-field collision 和 qualification 结构校验。

**步骤 3：增加 compiler**

实现：

```python
def _compile_index_engine_contract(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    ...
```

返回 resolved matrix、phase index version、forward/rollback flags 和 report metadata。普通场景返回 `none/not_applicable`，不改变任何原始执行字段。

**步骤 4：增加 resolved contract validator**

实现 matrix capability、validator version、rollback compatibility、qualified immutable image 和 topology evidence 校验。

**步骤 5：接入解析顺序**

`resolve_gate_scenario()` 必须先识别 raw contract，再解析 contract-owned matrix 和 phases，最后调用 `validate_resolved_gate_scenario()`。普通场景继续走现有 `_resolve_ref()` 路径。

**步骤 6：运行合同测试**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  -k 'contract or qualification'
```

预期：PASS。

**步骤 7：提交 compiler**

```bash
git add milvus_client/common/gates.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py
git commit -m "feat: compile index engine compatibility contracts"
```

### 任务 4：升级 manifest v2 并迁移四个 index 场景

**文件：**
- 修改：`milvus_client/manifests/upgrade_rollback_gates.yaml:1-950`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py:99-1200`

**步骤 1：写迁移后的 raw manifest 测试**

断言四个 index scenario：

- 存在结构化 target-only contract。
- 不再显式声明 contract-owned 低层字段。
- v10/v11 capability、matrix 和 validator version 一致。
- resolved/rendered 结果仍满足当前 target-only 行为。

**步骤 2：运行并确认失败**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  -k index_engine
```

预期：FAIL，因为 manifest 仍使用 v1 低层字段。

**步骤 3：迁移 manifest**

- `version: "1"` 改为 `version: "2"`。
- 增加空的 `capability_qualifications: {}`。
- 四个 index 场景改为第 3.2 节结构。
- 其他 22 个 scenario block 不修改执行字段。

**步骤 4：运行语义快照与 manifest 全测**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py
```

预期：PASS；26 条 execution path signature 与 v1 fixture 完全相同。

**步骤 5：提交 manifest 迁移**

```bash
git add milvus_client/manifests/upgrade_rollback_gates.yaml \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py
git commit -m "refactor: express index gates as compatibility contracts"
```

### 任务 5：渲染受保护合同 metadata

**文件：**
- 修改：`milvus_client/common/gates.py:161-330`
- 修改：`milvus_client/common/gates.py:330-435`
- 修改：`milvus_client/tests/test_render_upgrade_rollback_params.py:44-620`
- 修改：`milvus_client/tests/test_upgrade_rollback_gates_manifest.py:63-508`

**步骤 1：写失败测试**

断言：

```python
assert target_only_params["index-engine-contract-mode"] == "target_only"
assert target_only_params["index-engine-capability"] == "IndexEngineV10V4"
assert target_only_params["index-engine-qualification-status"] == "unsupported"

assert normal_params["index-engine-contract-mode"] == "none"
assert normal_params["index-engine-capability"] == "none"
assert normal_params["index-engine-qualification-status"] == "not_applicable"
```

并验证 registered runtime drift 会拒绝修改三项字段；unregistered scenario 只能使用安全 sentinel。

**步骤 2：运行并确认失败**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  -k 'contract_mode or qualification_status or protected_parameter'
```

预期：FAIL，renderer 尚未输出 metadata。

**步骤 3：实现参数渲染和保护**

- `render_argo_parameters()` 总是渲染三项 metadata。
- 不加入 mutable parameter allowlist。
- `UNREGISTERED_SCENARIO_METADATA` 增加安全 sentinel。

**步骤 4：运行测试并提交**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py

git add milvus_client/common/gates.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py
git commit -m "feat: protect index contract workflow metadata"
```

### 任务 6：同步三份 4am WorkflowTemplate，不改变 DAG

**文件：**
- 修改：`argo/standalone-2-6-upgrade-rollback.yaml:10-120,1273-1390,1435-1490,2656-2760,2941-3005`
- 修改：`argo/standalone-3-0-upgrade-rollback.yaml:10-120,1273-1390,1435-1490,2656-2760,2941-3005`
- 修改：`argo/cluster-upgrade-rollback.yaml:10-120,1277-1395,1439-1495,2773-2890,3125-3185`
- 修改：`milvus_client/tests/test_argo_template.py:1-1200,2427-3800`

**步骤 1：写模板失败测试**

对三份模板断言：

- spec arguments 声明三项 metadata，默认 sentinel。
- deploy 前 `runtime_parameters` 包含三项 metadata。
- env snapshot、flow summary、cleanup fallback 都记录 metadata。
- final report command 传递 metadata。
- DAG task 名称、依赖、`when` 和 forward/rollback low-level 参数保持原样。

**步骤 2：运行并确认失败**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_argo_template.py \
  -k index_engine_contract
```

预期：FAIL，模板尚未声明参数。

**步骤 3：最小修改三份模板**

只增加参数传递和报告字段，不增加或删除 DAG task，不让模板根据 contract mode 自行推导低层行为。

**步骤 4：测试和 Argo lint**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests/test_argo_template.py

argo lint --offline --kinds workflowtemplates --no-color \
  argo/standalone-2-6-upgrade-rollback.yaml \
  argo/standalone-3-0-upgrade-rollback.yaml \
  argo/cluster-upgrade-rollback.yaml
```

预期：pytest PASS；`✔ no linting errors found!`。

**步骤 5：提交模板同步**

```bash
git add argo/standalone-2-6-upgrade-rollback.yaml \
  argo/standalone-3-0-upgrade-rollback.yaml \
  argo/cluster-upgrade-rollback.yaml \
  milvus_client/tests/test_argo_template.py
git commit -m "feat: report index contracts in upgrade workflows"
```

### 任务 7：在 final report 中展示合同

**文件：**
- 修改：`milvus_client/requests/generate_workflow_report.py:90-330,330-530`
- 修改：`milvus_client/tests/test_generate_workflow_report.py:1-1340`

**步骤 1：写报告失败测试**

断言 orchestrator JSON 包含：

```json
{
  "index_engine_contract": {
    "mode": "target_only",
    "capability": "IndexEngineV10V4",
    "qualification_status": "unsupported",
    "base_target_vec_index_version": -1,
    "target_target_vec_index_version": 10,
    "rollback_target_vec_index_version": -1,
    "target_created_data_required_after_rollback": false
  }
}
```

Markdown 增加 `Index Engine Compatibility Contract` 小节。非 index 场景显示 `none/not_applicable`，不影响 gate status。

**步骤 2：运行并确认失败**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_generate_workflow_report.py \
  -k index_engine_contract
```

**步骤 3：实现 parser、JSON 和 Markdown**

增加三个 CLI 参数；把合同 metadata 放在 report 顶层，同时保留现有 `parameters.config_matrix` 低层字段。required validation brick 继续由编译后的低层开关决定，不在 report 中重新解释 mode。

**步骤 4：运行测试并提交**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests/test_generate_workflow_report.py

git add milvus_client/requests/generate_workflow_report.py \
  milvus_client/tests/test_generate_workflow_report.py
git commit -m "feat: include index contracts in workflow reports"
```

### 任务 8：更新维护文档

**文件：**
- 修改：`docs/upgrade-rollback-gates/README.md:1-340`
- 修改：`milvus_client/docs/upgrade-rollback.md:90-190`

**步骤 1：记录 manifest v2 规则**

文档包括：

- target-only / round-trip 真值表。
- qualification registry 和 immutable image 规则。
- v3.0.0 当前四场景为什么是 target-only。
- v3.0.1 qualified 后如何为 3.0.2 新增 version-specific round-trip 场景。
- `index_compatibility_validation_enabled` 与合同 mode 的区别。
- 非 index 场景为什么保持 `none/not_applicable`。

**步骤 2：更新提交参数和报告说明**

记录三个新 Argo 参数是受保护 metadata，用户不应在正式 registered gate 中手工覆盖。

**步骤 3：运行文档相关测试并提交**

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_argo_template.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py

git add docs/upgrade-rollback-gates/README.md \
  milvus_client/docs/upgrade-rollback.md
git commit -m "docs: describe index engine compatibility contracts"
```

### 任务 9：全量回归和最终审计

**文件：**
- 验证：所有上述文件

**步骤 1：运行全部 Milvus client 测试**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests
```

预期：全部 PASS。

**步骤 2：运行静态检查**

```bash
ruff check \
  milvus_client/common/gates.py \
  milvus_client/requests/generate_workflow_report.py \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/tests/test_generate_workflow_report.py \
  milvus_client/tests/test_argo_template.py

git diff --check
```

预期：Ruff `All checks passed!`，`git diff --check` 无输出。

**步骤 3：再次运行 Argo offline lint**

```bash
argo lint --offline --kinds workflowtemplates --no-color \
  argo/standalone-2-6-upgrade-rollback.yaml \
  argo/standalone-3-0-upgrade-rollback.yaml \
  argo/cluster-upgrade-rollback.yaml
```

预期：`✔ no linting errors found!`。

**步骤 4：审计路径影响**

检查 execution path fixture：

- 26 个场景全部存在。
- 四个 index scenario 的 v2 compiled signature 等于 v1 target-only signature。
- 其他 22 个 scenario signature 完全不变。
- 只有合同 metadata 参数是预期新增差异。

**步骤 5：最终提交**

```bash
git status --short
git log --oneline --decorate -10
```

若仍有未提交的计划内修改：

```bash
git add <remaining-planned-files>
git commit -m "test: verify index contract migration"
```

## 8. 后续 3.0.2 发布迁移流程

合同基础设施完成不等于立即把当前场景改为 round-trip。3.0.2 发布前按 capability 独立执行：

1. 将官方 v3.0.1 image alias 固定到 manifest-list digest。
2. 运行 v10/v4 的 standalone 和 cluster `3.0.1 → 3.0.1 → 3.0.1` qualification。
3. 独立运行 v11/v4 qualification。
4. 把通过结果写入 `capability_qualifications`。
5. 为通过的 capability 新增 `3.0.1 → 3.0.2 candidate → 3.0.1` round-trip 场景。
6. 未通过的 capability 继续使用 target-only，不允许借用另一个 capability 的 evidence。
7. Release gate 必须同时包含 baseline-origin 和 target-origin collection 的 rollback 验证。

## 9. 完成标准

- manifest v2 只能由理解合同的 renderer 加载。
- 四个当前 index 场景只通过结构化合同表达，不保留派生低层字段。
- 四个场景执行语义与当前 target-only 行为完全一致。
- 其他 22 条路径的 execution signature 完全不变。
- 三份 WorkflowTemplate 和所有报告对合同 metadata 一致可见。
- 非 index 场景明确报告 `none/not_applicable`。
- 未 qualified immutable image 无法渲染正式 round-trip gate。
- pytest、Ruff、Argo offline lint、`git diff --check` 全部通过。
