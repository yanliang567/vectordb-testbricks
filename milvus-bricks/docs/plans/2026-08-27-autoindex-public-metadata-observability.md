# AUTOINDEX 公共元数据可观测性修正计划

**目标：** 消除 index v10/v4、v11/v4 target-only gate 对公共 API 不可提供元数据的错误依赖，同时保留真实兼容性回归的严格阻断能力。

**背景：** Milvus `DescribeIndex` 对 `AUTOINDEX` 返回用户声明参数，而不是 DataCoord 内部解析后的 index type。因此 JSON `AUTOINDEX` 实际解析为 `HYBRID` 时，PyMilvus 仍可能只能观测到顶层 `AUTOINDEX`。当前 validator 已能识别 `public_sdk_unavailable`，但把它当成 metadata mismatch，导致索引已成功 build/query/reload 的 target-only 场景失败。

## 验证合同

validator 按以下顺序保留证据和决定 gate 结果：

1. 索引集合、名称、声明类型、参数等公共元数据仍按 schema matrix 校验。
2. 能从 `resolved_index_type`、嵌套 `index_type` 或顶层 resolved type 观测到实际类型时，必须等于 `expected_resolved_index_type`，否则失败。
3. 仅当 schema matrix 声明类型为 `AUTOINDEX`，且来源明确为 `public_sdk_unavailable` 时，不因 resolved type 不可观测而失败；报告保留 expected、observed、source、validation 和 unobservable 计数。
4. 索引元数据缺失使用 `index_metadata_unavailable`，继续失败。
5. build、query/search、release/load 后 query/search、数据完整性和目标 index engine version 的现有验证保持不变，任何功能失败仍阻断。

该规则由 validator 统一执行，因此同时覆盖 v10/v4、v11/v4 及未来复用 `expected_resolved_index_type` 的 matrix，不修改 manifest 合同、场景路径或 Argo WorkflowTemplate。

## 实施步骤

### 任务 1：用测试锁定可观测性边界

**文件：** `milvus_client/tests/test_validate_index_compatibility.py`

- 将公共 SDK 只返回 `AUTOINDEX` 的用例改为通过，并断言完整证据指标。
- 增加索引元数据完全缺失仍失败的用例。
- 保留显式 resolved type 不匹配仍失败的用例。

### 任务 2：最小化修改 validator

**文件：** `milvus_client/requests/validate_index_compatibility.py`

- 在 resolved type 不可观测分支中，仅豁免 `AUTOINDEX` + `public_sdk_unavailable`。
- 记录 `not_observable_via_public_sdk` validation 指标。
- 不改变显式 mismatch 和 metadata missing 的失败路径。

### 任务 3：验证

```bash
python3 -m pytest milvus_client/tests/test_validate_index_compatibility.py -q
python3 -m pytest milvus_client/tests -q
python3 -m ruff check milvus_client/requests/validate_index_compatibility.py \
  milvus_client/tests/test_validate_index_compatibility.py
git diff --check
```

代码合入并更新 live WorkflowTemplate 后，重新触发 standalone 3.0 v10/v4 canary。预期 resolved metadata 记录为不可观测证据，workflow 继续完成 target 功能检查和 cleanup；该 target-only 合同不进入 rollback。
