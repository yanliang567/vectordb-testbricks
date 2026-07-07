# Milvus Client Bricks 扩展实现计划

**目标：** 在保留 `milvus-bricks/2.6/` 旧脚本目录不变的前提下，新增以 `MilvusClient` 为核心、可被 Argo Workflow 灵活编排的 `milvus-bricks/milvus_client/` test request brick 体系，并补齐 Milvus 2.6/3.0 新功能和升级回滚兼容性场景。

**架构：** 先新增独立目录和统一运行协议，避免破坏现有 `2.6/` 调用方；再抽取连接、schema、数据生成、指标输出等公共能力，最后逐步新增独立 request bricks 和 scenario orchestrator。每个 brick 都是可单独运行、可 checkpoint、可输出 JSON 结果的最小测试请求，Argo 只负责组合和生命周期编排。

**技术栈：** Python 3, pymilvus `MilvusClient` / `AsyncMilvusClient`, pytest, JSON/YAML manifests, Argo Workflow, Milvus 2.6/3.0, optional Helm/K8s upgrade hooks.

---

## 背景和范围

当前 `milvus-bricks/` 同时包含老的 Collection API 脚本和 `2.6/` 目录下的 MilvusClient 脚本。`2.6/` 已经是重要参考实现，但目录名绑定版本，脚本入口参数风格不统一，输出主要是日志，难以被 Argo 做细粒度调度、组合和结果判定。为了降低迁移风险，`2.6/` 保持原样；新的标准化运行协议和 request bricks 放到单独的 `milvus_client/` 目录。

本计划的第一目标不是一次性重写全部脚本，而是建立稳定骨架：目录、运行协议、公共库、manifest、P0 bricks 和升级回滚场景。后续 3.0 新功能以独立 bricks 增量接入。

## 前置功能梳理

正式实现 P0 request bricks 前，先维护一份 Milvus 3.0/Feishu Base 功能目录：

- 文档: `docs/plans/2026-07-07-milvus-3-feature-inventory.md`
- 作用: 整合官方 release notes、roadmap 和 Feishu Base 的 PR 级功能。
- 产物: 功能域、优先级、兼容模式、capability probe 需求、对 request 参数和 manifest 的影响。

这份 inventory 会影响公共 CLI 参数、schema matrix、brick catalog 和升级回滚场景。任何新增 3.0 feature brick 前，都应先在 inventory 或后续机器可读 `feature_inventory.yaml` 中登记。

## 非目标

- 不在第一阶段删除根目录老 Collection API 脚本。
- 不把 Milvus 升级/回滚动作硬编码进每个 brick。
- 不在第一阶段完整覆盖 Feishu Base 里的全部 108 条功能。
- 不要求 3.0-only schema 在回滚到 2.6 后仍可用；这类能力必须和 2.6 兼容集分开。

## 目标目录结构

```text
milvus-bricks/
  2.6/
    ...                 # Existing scripts kept unchanged for compatibility
  legacy_collection_api/
  milvus_client/
    README.md
    requirements.txt
    common/
      __init__.py
      args.py
      client.py
      data.py
      metrics.py
      result.py
      schema.py
      validators.py
    requests/
      __init__.py
      create_schema_matrix.py
      seed_data.py
      mixed_rw_pressure.py
      validate_data_integrity.py
      search_pressure.py
      query_pressure.py
      query_iterator_scan.py
      upsert_pressure.py
      delete_pressure.py
    scenarios/
      __init__.py
      upgrade_rollback_compatibility.py
    manifests/
      brick_catalog.yaml
      feature_inventory.yaml
      capability_catalog.yaml
      schema_matrix_2_6.yaml
      schema_matrix_3_0.yaml
      scenario_upgrade_rollback.yaml
    horizon_poc/
      ...
```

## Brick 运行协议

所有新 bricks 必须支持以下公共参数：

```bash
python -m milvus_client.requests.<brick_name> \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --db-name default \
  --collection-prefix qa_brick \
  --duration-sec 600 \
  --seed 20260707 \
  --feature-set compat_2_6 \
  --compat-mode rollback_safe \
  --capability-probe true \
  --skip-unsupported true \
  --lifecycle-phase steady_state \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/<brick_name>.json \
  --log-level INFO
```

统一结果 JSON：

```json
{
  "brick": "mixed_rw_pressure",
  "feature_set": "compat_2_6",
  "compat_mode": "rollback_safe",
  "lifecycle_phase": "steady_state",
  "status": "passed",
  "started_at": "2026-07-07T10:00:00+08:00",
  "finished_at": "2026-07-07T10:10:00+08:00",
  "target": {
    "uri": "http://localhost:19530",
    "db_name": "default",
    "collection_prefix": "qa_brick"
  },
  "metrics": {
    "requests_total": 1000,
    "requests_failed": 0,
    "entities_inserted": 50000,
    "p99_latency_ms": 120
  },
  "failures": [],
  "capabilities": {
    "server_version": "unknown",
    "sdk_version": "unknown",
    "supported": [],
    "unsupported": []
  },
  "skip_reason": null,
  "artifacts": [],
  "checkpoint": {
    "path": "/tmp/milvus-bricks/checkpoints/mixed_rw_pressure.json",
    "version": 1
  }
}
```

退出码约定：

- `0`: brick 执行成功，且 `status=passed` 或 `status=warning`。
- `1`: 测试断言失败，例如数据丢失、count drift、checksum mismatch。
- `2`: 参数错误或 manifest 不合法。
- `3`: 环境不可用，例如无法连接 Milvus。
- `4`: brick 内部异常。

## 任务 0: 功能目录和协议影响分析

**文件：**
- 创建: `docs/plans/2026-07-07-milvus-3-feature-inventory.md`
- 后续创建: `milvus-bricks/milvus_client/manifests/feature_inventory.yaml`
- 后续创建: `milvus-bricks/milvus_client/manifests/capability_catalog.yaml`

**步骤 1: 整合功能来源**

从以下来源梳理功能：

- Milvus release notes: `https://milvus.io/docs/release_notes.md`
- Milvus roadmap: `https://milvus.io/docs/roadmap.md`
- Feishu Base 视图: `Gy4VbeG8daTv88s5OtocmUuSnyh / tblOIf9XnB5mz29r / vew5tEP7E3`

**步骤 2: 按功能域分组**

至少覆盖：

- StructArray / EmbeddingList
- Query / Search Semantics
- Text / Analyzer / Expression
- Schema / Index / DML Evolution
- Storage / External / Import
- Function / Model Provider
- Ops / Security / CDC

**步骤 3: 标注对协议的影响**

为每个功能域判断：

- 是否需要新 schema 表达能力。
- 是否需要新 request 参数。
- 是否 rollback-safe。
- 是否需要 external source、secret、cluster admin 或 K8s 权限。
- validator 如何计算 ground truth。

**步骤 4: 更新本实现计划**

将公共参数、result JSON、manifest 设计同步到本文。

**步骤 5: 验收**

检查：

```bash
test -f docs/plans/2026-07-07-milvus-3-feature-inventory.md
rg -n "feature-set|compat-mode|capability-probe|lifecycle-phase" docs/plans/2026-07-07-milvus-client-bricks-expansion.md
```

预期：文档存在，且实现计划已包含 feature/capability/compat/lifecycle 设计。

## 任务 1: 新建 MilvusClient 目录和兼容边界

**文件：**
- 保留: `milvus-bricks/2.6/`
- 创建: `milvus-bricks/milvus_client/`
- 创建: `milvus-bricks/legacy_collection_api/README.md`
- 修改: `milvus-bricks/milvus_client/README.md`

**步骤 1: 新建目录而不是移动旧目录**

保留 `milvus-bricks/2.6/` 中的现有脚本，新增 `milvus-bricks/milvus_client/` 作为标准化 request brick runtime。需要复用旧脚本时，以复制或后续 wrapper/request 化的方式逐步迁移，不删除旧路径。

预期：

```text
milvus-bricks/2.6/README.md exists
milvus-bricks/milvus_client/README.md exists
```

**步骤 2: 保留旧路径兼容说明**

创建 `milvus-bricks/legacy_collection_api/README.md`：

```markdown
# Legacy Collection API Bricks

Scripts in this area use the legacy `pymilvus.Collection` and `utility` APIs.
New development should target `milvus-bricks/milvus_client`.
```

**步骤 3: 更新 README 标题和路径**

将 `milvus-bricks/milvus_client/README.md` 标题从 `Milvus 2.6 MilvusClient Test Scripts` 改成：

```markdown
# MilvusClient Test Bricks

This directory contains Milvus test bricks based on `pymilvus.MilvusClient`
and `AsyncMilvusClient`. It supports Milvus 2.6 compatible bricks and will be
extended for Milvus 3.0 feature coverage.

The existing `milvus-bricks/2.6/` scripts are kept unchanged for compatibility.
```

**步骤 4: 搜索硬编码路径**

运行：

```bash
rg -n "milvus-bricks/2\\.6|/2\\.6|2\\.6/" .
```

预期：

```text
Only intentional compatibility notes remain.
```

**步骤 5: 验证现有脚本仍可 import**

运行：

```bash
cd milvus-bricks
python -m py_compile 2.6/common.py 2.6/create_n_insert.py 2.6/search_permanently.py 2.6/query_iterator.py 2.6/upsert3.py
PYTHONPATH=. python -m py_compile milvus_client/common/client.py milvus_client/requests/precheck.py
```

预期：

```text
No output and exit code 0.
```

**步骤 6: 提交**

```bash
git add milvus-bricks
git commit -m "feat(milvus): add milvus client brick runtime"
```

## 任务 2: 建立 Python package 和公共参数解析

**文件：**
- 创建: `milvus-bricks/milvus_client/__init__.py`
- 创建: `milvus-bricks/milvus_client/common/__init__.py`
- 创建: `milvus-bricks/milvus_client/common/args.py`
- 创建: `milvus-bricks/milvus_client/tests/test_args.py`

**步骤 1: 写失败测试**

创建 `milvus-bricks/milvus_client/tests/test_args.py`：

```python
from milvus_client.common.args import build_common_parser


def test_common_parser_accepts_required_options(tmp_path):
    parser = build_common_parser("test")
    args = parser.parse_args([
        "--uri", "http://localhost:19530",
        "--token", "root:Milvus",
        "--collection-prefix", "qa",
        "--duration-sec", "60",
        "--seed", "123",
        "--feature-set", "compat_2_6",
        "--compat-mode", "rollback_safe",
        "--capability-probe", "true",
        "--skip-unsupported", "true",
        "--lifecycle-phase", "steady_state",
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--output-json", str(tmp_path / "result.json"),
    ])

    assert args.uri == "http://localhost:19530"
    assert args.token == "root:Milvus"
    assert args.collection_prefix == "qa"
    assert args.duration_sec == 60
    assert args.seed == 123
    assert args.feature_set == "compat_2_6"
    assert args.compat_mode == "rollback_safe"
    assert args.capability_probe is True
    assert args.skip_unsupported is True
    assert args.lifecycle_phase == "steady_state"
```

**步骤 2: 运行测试验证失败**

运行：

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_args.py -v
```

预期：

```text
ModuleNotFoundError: No module named 'milvus_client.common.args'
```

**步骤 3: 实现公共 parser**

创建 `milvus-bricks/milvus_client/common/args.py`：

```python
import argparse


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--db-name", default="default")
    parser.add_argument("--collection-prefix", required=True)
    parser.add_argument("--duration-sec", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-set", default="compat_2_6")
    parser.add_argument(
        "--compat-mode",
        choices=["rollback_safe", "upgrade_only", "forward_only"],
        default="rollback_safe",
    )
    parser.add_argument("--capability-probe", type=parse_bool, default=True)
    parser.add_argument("--skip-unsupported", type=parse_bool, default=True)
    parser.add_argument(
        "--lifecycle-phase",
        choices=[
            "before_upgrade",
            "after_upgrade",
            "before_rollback",
            "after_rollback",
            "steady_state",
        ],
        default="steady_state",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser
```

**步骤 4: 运行测试验证通过**

运行：

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_args.py -v
```

预期：

```text
1 passed
```

**步骤 5: 提交**

```bash
git add milvus-bricks/milvus_client
git commit -m "feat(milvus-client): add common brick argument parser"
```

## 任务 3: 统一结果 JSON 和错误码

**文件：**
- 创建: `milvus-bricks/milvus_client/common/result.py`
- 创建: `milvus-bricks/milvus_client/tests/test_result.py`

**步骤 1: 写失败测试**

```python
import json
from milvus_client.common.result import BrickResult


def test_write_result_json(tmp_path):
    path = tmp_path / "result.json"
    result = BrickResult(
        brick="unit",
        feature_set="compat_2_6",
        compat_mode="rollback_safe",
        lifecycle_phase="steady_state",
        status="passed",
        target={"uri": "http://localhost:19530"},
        metrics={"requests_total": 1},
    )

    result.write(path)

    payload = json.loads(path.read_text())
    assert payload["brick"] == "unit"
    assert payload["status"] == "passed"
    assert payload["feature_set"] == "compat_2_6"
    assert payload["compat_mode"] == "rollback_safe"
    assert payload["lifecycle_phase"] == "steady_state"
    assert payload["skip_reason"] is None
    assert payload["metrics"]["requests_total"] == 1
    assert payload["failures"] == []
```

**步骤 2: 运行测试验证失败**

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_result.py -v
```

预期：`ModuleNotFoundError` 或 `ImportError`。

**步骤 3: 实现 `BrickResult`**

```python
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


@dataclass
class BrickResult:
    brick: str
    feature_set: str
    compat_mode: str
    lifecycle_phase: str
    status: str
    target: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    artifacts: list[str] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def write(self, path: str | Path) -> None:
        self.finish()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
```

**步骤 4: 运行测试验证通过**

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_result.py -v
```

预期：`1 passed`。

**步骤 5: 提交**

```bash
git add milvus-bricks/milvus_client/common/result.py milvus-bricks/milvus_client/tests/test_result.py
git commit -m "feat(milvus-client): add structured brick result output"
```

## 任务 4: 连接封装和环境 precheck

**文件：**
- 创建: `milvus-bricks/milvus_client/common/client.py`
- 创建: `milvus-bricks/milvus_client/requests/precheck.py`
- 创建: `milvus-bricks/milvus_client/tests/test_client.py`

**步骤 1: 写单元测试**

```python
from milvus_client.common.client import normalize_uri


def test_normalize_uri_adds_http_for_host():
    assert normalize_uri("localhost") == "http://localhost:19530"


def test_normalize_uri_keeps_full_uri():
    assert normalize_uri("https://example.com") == "https://example.com"
```

**步骤 2: 实现 `normalize_uri` 和 `create_client`**

```python
from pymilvus import MilvusClient


def normalize_uri(uri: str) -> str:
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri
    return f"http://{uri}:19530"


def create_client(uri: str, token: str = "", db_name: str = "default") -> MilvusClient:
    normalized = normalize_uri(uri)
    if token:
        return MilvusClient(uri=normalized, token=token, db_name=db_name)
    return MilvusClient(uri=normalized, db_name=db_name)
```

**步骤 3: 实现 `precheck.py`**

`precheck.py` 调用 `client.get_server_version()` 或 `client.list_collections()`，写出结果 JSON。

**步骤 4: 本地编译验证**

```bash
cd milvus-bricks
PYTHONPATH=. python -m py_compile milvus_client/common/client.py milvus_client/requests/precheck.py
PYTHONPATH=. pytest milvus_client/tests/test_client.py -v
```

预期：编译通过，测试通过。

**步骤 5: 集群 smoke 验证**

```bash
cd milvus-bricks
PYTHONPATH=. python -m milvus_client.requests.precheck \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_precheck \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/precheck.json
```

预期：`precheck.json` 中 `status=passed`。

## 任务 5: Schema Matrix manifest

**文件：**
- 创建: `milvus-bricks/milvus_client/manifests/feature_inventory.yaml`
- 创建: `milvus-bricks/milvus_client/manifests/capability_catalog.yaml`
- 创建: `milvus-bricks/milvus_client/manifests/schema_matrix_2_6.yaml`
- 创建: `milvus-bricks/milvus_client/manifests/schema_matrix_3_0.yaml`
- 创建: `milvus-bricks/milvus_client/common/schema.py`
- 创建: `milvus-bricks/milvus_client/tests/test_schema_manifest.py`

**步骤 1: 定义 feature inventory 和 capability catalog**

`feature_inventory.yaml` 先从 `docs/plans/2026-07-07-milvus-3-feature-inventory.md` 提取 P0/P1 能力：

```yaml
features:
  - id: compat_basic_vector
    domain: basic
    priority: P0
    compat_mode: rollback_safe
    required_capabilities: []
    bricks:
      - create_schema_matrix
      - seed_data
      - mixed_rw_pressure
      - validate_data_integrity

  - id: query_aggregation
    domain: query_search
    priority: P1
    compat_mode: rollback_safe
    required_capabilities: ["QueryAggregation"]
    bricks:
      - query_aggregation_matrix

  - id: struct_array_element_hybrid_search
    domain: struct_array
    priority: P1
    compat_mode: forward_only
    required_capabilities: ["StructArray", "ElementHybridSearch"]
    bricks:
      - struct_array_hybrid_search
```

`capability_catalog.yaml` 定义探测方式：

```yaml
capabilities:
  - id: QueryAggregation
    detect:
      server_version_min: "3.0.0"
      sdk_probe: "query_aggregation"
    unsupported_behavior: skip

  - id: StructArray
    detect:
      server_version_min: "3.0.0"
      sdk_probe: "create_struct_array_schema"
    unsupported_behavior: skip
```

**步骤 2: 定义 2.6 schema matrix**

初始覆盖：

```yaml
version: "2.6"
schemas:
  - name: dense_float_vector
    feature_tags: ["compat_basic_vector"]
    compat_mode: rollback_safe
    required_capabilities: []
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: category, dtype: INT64}
      - {name: content, dtype: VARCHAR, max_length: 1024, nullable: true}
      - {name: flag, dtype: BOOL, nullable: true}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 128}
    indexes:
      - {field: embedding, index_type: HNSW, metric_type: COSINE}
  - name: multi_vector_numeric
    feature_tags: ["compat_multi_vector"]
    compat_mode: rollback_safe
    required_capabilities: []
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: float16_vec, dtype: FLOAT16_VECTOR, dim: 128}
      - {name: bfloat16_vec, dtype: BFLOAT16_VECTOR, dim: 128}
      - {name: int8_vec, dtype: INT8_VECTOR, dim: 128}
    indexes:
      - {field: float16_vec, index_type: AUTOINDEX, metric_type: COSINE}
      - {field: bfloat16_vec, index_type: AUTOINDEX, metric_type: COSINE}
      - {field: int8_vec, index_type: AUTOINDEX, metric_type: COSINE}
  - name: json_array_nullable
    feature_tags: ["compat_json_array"]
    compat_mode: rollback_safe
    required_capabilities: []
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: tags, dtype: ARRAY, element_type: VARCHAR, max_capacity: 32, max_length: 128}
      - {name: profile, dtype: JSON, nullable: true}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 128}
```

**步骤 3: 定义 3.0 schema matrix**

在 2.6 基础上增加：

```yaml
version: "3.0"
schemas:
  - name: nullable_vector
    feature_tags: ["nullable_vector"]
    compat_mode: forward_only
    required_capabilities: ["NullableVector"]
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 128, nullable: true}
  - name: geometry_rtree
    feature_tags: ["geometry"]
    compat_mode: forward_only
    required_capabilities: ["Geometry"]
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: location, dtype: GEOMETRY}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 128}
    indexes:
      - {field: location, index_type: RTREE}
      - {field: embedding, index_type: AUTOINDEX, metric_type: COSINE}
  - name: timestamptz_ttl
    feature_tags: ["timestamptz", "entity_ttl"]
    compat_mode: forward_only
    required_capabilities: ["TimeSeries", "EntityTTL"]
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: false}
      - {name: event_time, dtype: TIMESTAMPTZ}
      - {name: expire_at, dtype: INT64}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 128}
```

StructArray/EmbList schema 先在 manifest 预留 `experimental: true`，等实际 pymilvus API 确认后实现。

**步骤 4: 写 manifest 校验测试**

测试要求：

- 每个 schema 有唯一 `name`。
- 每个 schema 有 exactly one primary field。
- 每个 vector field 有 `dim`。
- 每个 index 指向存在字段。
- 每个 `feature_tags` 能在 `feature_inventory.yaml` 中找到。
- 每个 `required_capabilities` 能在 `capability_catalog.yaml` 中找到。
- `compat_mode` 属于 `rollback_safe|upgrade_only|forward_only`。

**步骤 5: 实现 YAML loader**

`common/schema.py` 提供：

```python
def load_schema_matrix(path: str) -> list[SchemaSpec]: ...
def validate_schema_matrix(specs: list[SchemaSpec]) -> list[str]: ...
def build_milvus_schema(spec: SchemaSpec): ...
def load_feature_inventory(path: str) -> dict[str, FeatureSpec]: ...
def load_capability_catalog(path: str) -> dict[str, CapabilitySpec]: ...
```

**步骤 6: 运行测试**

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_schema_manifest.py -v
```

预期：全部通过。

## 任务 6: `create_schema_matrix` brick

**文件：**
- 创建: `milvus-bricks/milvus_client/requests/create_schema_matrix.py`
- 测试: `milvus-bricks/milvus_client/tests/test_create_schema_matrix.py`

**步骤 1: 实现 dry-run 模式**

新增参数：

```text
--schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml
--drop-if-exists false
--dry-run
```

dry-run 只验证 manifest，不连接 Milvus。

**步骤 2: 写 dry-run 测试**

```python
from milvus_client.requests.create_schema_matrix import run_dry_run


def test_create_schema_matrix_dry_run_loads_manifest():
    result = run_dry_run("milvus_client/manifests/schema_matrix_2_6.yaml")
    assert result["schemas_total"] > 0
    assert result["errors"] == []
```

**步骤 3: 实现真实创建逻辑**

行为：

- collection name = `{collection_prefix}_{schema_name}`。
- 如果存在且 `--drop-if-exists=false`，跳过创建并记录 `skipped`。
- 创建 schema。
- 创建索引。
- 可选 `--load-after-create`。
- 输出所有 collection 名和 schema spec hash。

**步骤 4: 集群验证**

```bash
cd milvus-bricks
PYTHONPATH=. python -m milvus_client.requests.create_schema_matrix \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/create_schema_matrix.json
```

预期：

```text
exit code 0
status=passed
created_collections contains qa_schema_dense_float_vector
```

## 任务 7: 确定性数据生成和 seed_data brick

**文件：**
- 创建: `milvus-bricks/milvus_client/common/data.py`
- 创建: `milvus-bricks/milvus_client/requests/seed_data.py`
- 测试: `milvus-bricks/milvus_client/tests/test_data_generation.py`

**步骤 1: 写数据生成测试**

```python
from milvus_client.common.data import generate_rows


def test_generate_rows_is_deterministic():
    rows1 = generate_rows(schema_name="dense_float_vector", start_id=0, count=10, seed=7)
    rows2 = generate_rows(schema_name="dense_float_vector", start_id=0, count=10, seed=7)

    assert rows1 == rows2
    assert rows1[0]["id"] == 0
```

**步骤 2: 实现 generator**

生成规则：

- PK = `start_id + offset`。
- `category = id % 1024`。
- `version = writer_epoch`。
- dense vector 用 deterministic pseudo-random。
- nullable 字段按 `id % 10 == 0` 写 null。
- JSON 字段包含 `{"pk": id, "bucket": id % 16, "checksum": ...}`。
- ARRAY 字段包含可预测 tags。

**步骤 3: 实现 seed_data**

参数：

```text
--schema-matrix ...
--rows-per-collection 10000
--batch-size 1000
--start-id 0
--flush
```

输出：

- collection -> inserted rows。
- collection -> min/max pk。
- collection -> checksum。

**步骤 4: 验证**

```bash
cd milvus-bricks
PYTHONPATH=. python -m milvus_client.requests.seed_data \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --rows-per-collection 10000 \
  --batch-size 1000 \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/seed_data.json
```

预期：所有目标 collection 插入成功，`requests_failed=0`。

## 任务 8: 数据完整性 validator

**文件：**
- 创建: `milvus-bricks/milvus_client/common/validators.py`
- 创建: `milvus-bricks/milvus_client/requests/validate_data_integrity.py`
- 测试: `milvus-bricks/milvus_client/tests/test_validators.py`

**步骤 1: 实现 checkpoint 格式**

`seed_data` 写出：

```json
{
  "collections": {
    "qa_schema_dense_float_vector": {
      "schema_name": "dense_float_vector",
      "expected_count": 10000,
      "min_pk": 0,
      "max_pk": 9999,
      "checksum": "..."
    }
  }
}
```

**步骤 2: validator 校验项**

- `query count(*)` 等于 expected count。
- 抽样 PK query 命中。
- query_iterator 扫描数量和 PK 去重正确。
- 对确定性字段计算 checksum。
- search smoke: 对固定 query vector 返回非空，且无异常。
- nullable/null vector 行为按 schema spec 验证。

**步骤 3: 实现 failure 分类**

失败类型：

```text
COUNT_DRIFT
MISSING_PK
DUPLICATE_PK
CHECKSUM_MISMATCH
SEARCH_FAILED
QUERY_ITERATOR_FAILED
SCHEMA_INCOMPATIBLE
```

**步骤 4: 验证命令**

```bash
cd milvus-bricks
PYTHONPATH=. python -m milvus_client.requests.validate_data_integrity \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --checkpoint-file /tmp/milvus-bricks/checkpoints/seed_data.json \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/validate_data_integrity.json
```

预期：`status=passed`，`failures=[]`。

## 任务 9: mixed_rw_pressure brick

**文件：**
- 创建: `milvus-bricks/milvus_client/requests/mixed_rw_pressure.py`
- 修改: 可复用现有 `search_permanently.py`, `query_permanently_simplified.py`, `upsert3.py` 的核心逻辑
- 测试: `milvus-bricks/milvus_client/tests/test_mixed_rw_plan.py`

**步骤 1: 定义 workload 配置**

```yaml
weights:
  insert: 20
  upsert: 20
  delete: 5
  query: 20
  search: 25
  query_iterator: 10
limits:
  max_workers: 16
  batch_size: 100
  op_timeout_sec: 10
validation:
  sample_every_sec: 30
```

**步骤 2: 实现 operation scheduler**

规则：

- 每个 worker 随机选择操作，但使用固定 seed。
- write 操作只写入可计算 PK range，避免不同 worker 撞 key，除 upsert 专门使用固定 key。
- delete 默认只删 `delete_allowed=true` 的测试数据，不删 seed baseline。
- 所有异常分类记录，不直接吞掉。

**步骤 3: 实现在线 metrics**

每 10 秒刷新：

- op count。
- fail count。
- avg/p95/p99 latency。
- per-op throughput。

**步骤 4: 验证命令**

```bash
cd milvus-bricks
PYTHONPATH=. python -m milvus_client.requests.mixed_rw_pressure \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --duration-sec 300 \
  --seed 20260707 \
  --max-workers 16 \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/mixed_rw_pressure.json
```

预期：

- 运行 300 秒。
- 升级/重启期间允许短时连接错误，但要分类为 transient。
- 最终 `requests_total > 0`。

## 任务 10: 单功能 request bricks 拆分

**文件：**
- 创建: `milvus-bricks/milvus_client/requests/search_pressure.py`
- 创建: `milvus-bricks/milvus_client/requests/query_pressure.py`
- 创建: `milvus-bricks/milvus_client/requests/query_iterator_scan.py`
- 创建: `milvus-bricks/milvus_client/requests/upsert_pressure.py`
- 创建: `milvus-bricks/milvus_client/requests/delete_pressure.py`

**步骤 1: 从现有脚本抽取能力**

来源映射：

- `search_permanently.py` -> `search_pressure.py`
- `query_permanently_simplified.py` -> `query_pressure.py`
- `query_iterator.py` -> `query_iterator_scan.py`
- `upsert3.py` -> `upsert_pressure.py`
- `concurrent_delete_stress_test.py` 的 MilvusClient 逻辑 -> `delete_pressure.py`

**步骤 2: 每个 brick 支持统一协议**

每个 request 都必须：

- 复用 `build_common_parser`。
- 使用 `BrickResult` 输出。
- 支持 `--duration-sec`。
- 支持 `--schema-matrix` 或 `--collection-name`。
- 对 collection 不存在 fail fast。

**步骤 3: 每个 brick 做 smoke**

示例：

```bash
cd milvus-bricks
PYTHONPATH=. python -m milvus_client.requests.search_pressure \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_schema \
  --schema-matrix milvus_client/manifests/schema_matrix_2_6.yaml \
  --duration-sec 60 \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/search_pressure.json
```

预期：每个 brick 输出 JSON，exit code 符合协议。

## 任务 11: Brick catalog manifest

**文件：**
- 创建: `milvus-bricks/milvus_client/manifests/brick_catalog.yaml`
- 创建: `milvus-bricks/milvus_client/tests/test_brick_catalog.py`

**步骤 1: 定义 catalog**

```yaml
bricks:
  - name: precheck
    module: milvus_client.requests.precheck
    category: environment
    milvus_versions: ["2.6", "3.0"]
    feature_tags: []
    required_capabilities: []
    compat_mode: rollback_safe
    lifecycle_phases: ["before_upgrade", "after_upgrade", "before_rollback", "after_rollback", "steady_state"]
    destructive: false
  - name: create_schema_matrix
    module: milvus_client.requests.create_schema_matrix
    category: schema
    milvus_versions: ["2.6", "3.0"]
    feature_tags: ["compat_basic_vector", "compat_json_array"]
    required_capabilities: []
    compat_mode: rollback_safe
    lifecycle_phases: ["before_upgrade", "after_upgrade"]
    destructive: optional
  - name: mixed_rw_pressure
    module: milvus_client.requests.mixed_rw_pressure
    category: workload
    milvus_versions: ["2.6", "3.0"]
    feature_tags: ["compat_basic_vector"]
    required_capabilities: []
    compat_mode: rollback_safe
    lifecycle_phases: ["before_upgrade", "after_upgrade", "before_rollback", "after_rollback", "steady_state"]
    destructive: false
```

**步骤 2: 写 catalog 校验测试**

要求：

- `name` 唯一。
- `module` 可 import。
- `category` 属于枚举。
- `milvus_versions` 非空。
- `feature_tags` 必须存在于 `feature_inventory.yaml`，空列表除外。
- `required_capabilities` 必须存在于 `capability_catalog.yaml`，空列表除外。
- `compat_mode` 属于 `rollback_safe|upgrade_only|forward_only`。
- `lifecycle_phases` 属于已定义生命周期枚举。

**步骤 3: 运行测试**

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_brick_catalog.py -v
```

预期：全部通过。

## 任务 12: 3.0 P1 功能 bricks

**文件：**
- 创建: `milvus-bricks/milvus_client/requests/struct_array_matrix.py`
- 创建: `milvus-bricks/milvus_client/requests/json_path_index_matrix.py`
- 创建: `milvus-bricks/milvus_client/requests/text_analyzer_matrix.py`
- 创建: `milvus-bricks/milvus_client/requests/schema_evolution_matrix.py`
- 创建: `milvus-bricks/milvus_client/requests/query_search_feature_matrix.py`

**优先覆盖：**

1. StructArray / EmbList
   - 更多向量子字段类型。
   - element-level query/search/filter。
   - element-level group_by。
   - range search / iterator search。
   - hybrid search。
   - bitmap / STL_SORT / DISKANN。
   - null 和动态加字段。

2. JSON Path Index
   - Sort / Bitmap / Hybrid index。
   - nested path。
   - filter + order_by。

3. Text/Analyzer
   - BM25 highlighter。
   - semantic highlighter。
   - regex `=~` / `!~`。
   - raw string `r"..."`。
   - bitwise `& | ^`。
   - NGram、synonym、pinyin、Arabic/Thai tokenizer。

4. Schema Evolution
   - add field。
   - drop field。
   - nullable vector。
   - ARRAY_APPEND / ARRAY_REMOVE。
   - truncate collection。
   - entity-level TTL。

5. Search/Query
   - order_by。
   - aggregation count/min/max/sum/avg。
   - search_by_pk。
   - query_iterator。
   - group_by tied score。

**步骤 1: 每类先实现 dry-run 和 capability probe**

每个 brick 先判断服务端版本和 SDK 是否支持目标 API。不支持时输出：

```json
{
  "status": "skipped",
  "failures": [],
  "metrics": {
    "skip_reason": "server version does not support StructArray"
  }
}
```

**步骤 2: 每类补 L0 smoke**

每个功能先实现 1 个最小正向 case 和 1 个负向 case，避免 P1 一次性过大。

**步骤 3: 每类扩展矩阵**

按 `schema_matrix_3_0.yaml` 增补 L1/L2 组合。

## 任务 13: 升级回滚兼容性 scenario

**文件：**
- 创建: `milvus-bricks/milvus_client/scenarios/upgrade_rollback_compatibility.py`
- 创建: `milvus-bricks/milvus_client/manifests/scenario_upgrade_rollback.yaml`
- 创建: `milvus-bricks/milvus_client/tests/test_upgrade_rollback_scenario.py`

**步骤 1: 定义 scenario manifest**

```yaml
name: upgrade_rollback_compatibility
cycles: 3
observe_after_upgrade_sec: 1800
observe_after_rollback_sec: 1800
compat_schema_matrix: milvus_client/manifests/schema_matrix_2_6.yaml
forward_schema_matrix: milvus_client/manifests/schema_matrix_3_0.yaml
actions:
  upgrade:
    type: external
    wait_file: /tmp/milvus-bricks/actions/upgrade_done
  rollback:
    type: external
    wait_file: /tmp/milvus-bricks/actions/rollback_done
workloads:
  mixed_rw:
    max_workers: 32
    duration_sec: 0
  validator:
    interval_sec: 60
```

**步骤 2: 实现 scenario orchestrator**

流程：

```text
precheck
create compat_2_6_schema_set
seed compat data
start mixed_rw_pressure as background process
start validator loop as background process
for cycle in 1..N:
  wait external upgrade action
  observe upgraded cluster
  create forward_3_0_schema_set
  validate compat + forward
  wait external rollback action
  observe rollback cluster
  validate compat only
stop workloads
final validation
write scenario result
```

**步骤 3: 明确兼容性规则**

- `compat_2_6_schema_set`：必须在 2.6、3.0、回滚后都通过验证。
- `forward_3_0_schema_set`：只在 3.0 阶段验证。
- 如果 Storage V3 或 3.0-only schema 开启，不允许将其纳入 rollback 后必过集合。

**步骤 4: 实现外部动作等待**

scenario 不执行 Helm upgrade。它只等待 Argo 或人工创建文件：

```bash
touch /tmp/milvus-bricks/actions/upgrade_done
touch /tmp/milvus-bricks/actions/rollback_done
```

后续 Argo template 可以把这一步替换为真实 upgrade/rollback step。

**步骤 5: 本地 dry-run**

```bash
cd milvus-bricks
PYTHONPATH=. python -m milvus_client.scenarios.upgrade_rollback_compatibility \
  --dry-run \
  --scenario-manifest milvus_client/manifests/scenario_upgrade_rollback.yaml \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection-prefix qa_upgrade \
  --checkpoint-dir /tmp/milvus-bricks/checkpoints \
  --output-json /tmp/milvus-bricks/results/upgrade_rollback_dry_run.json
```

预期：输出 planned steps，不连接 Milvus 或不执行 destructive action。

## 任务 14: Argo Workflow 模板

**文件：**
- 创建: `milvus-bricks/argo/upgrade-rollback-compatibility.yaml`
- 创建: `milvus-bricks/argo/templates/brick-runner.yaml`

**步骤 1: 定义通用 brick-runner template**

输入参数：

```yaml
parameters:
  - name: brick_module
  - name: uri
  - name: token_secret_name
  - name: collection_prefix
  - name: args
```

执行：

```bash
PYTHONPATH=/workspace/milvus-bricks \
python -m {{inputs.parameters.brick_module}} \
  --uri "{{inputs.parameters.uri}}" \
  --token "$MILVUS_TOKEN" \
  --collection-prefix "{{inputs.parameters.collection_prefix}}" \
  {{inputs.parameters.args}}
```

**步骤 2: 定义升级回滚 workflow**

节点：

```text
precheck
create-compat-schema
seed-data
start-mixed-rw
cycle-1-upgrade
cycle-1-observe-upgrade
cycle-1-validate-upgrade
cycle-1-rollback
cycle-1-observe-rollback
cycle-1-validate-rollback
...
final-validate
collect-artifacts
```

**步骤 3: artifacts**

收集：

- `/tmp/milvus-bricks/results/*.json`
- `/tmp/milvus-bricks/checkpoints/*.json`
- `/tmp/milvus-bricks/logs/*.log`

## 任务 15: 文档和使用指南

**文件：**
- 创建: `milvus-bricks/milvus_client/README.md`
- 创建: `milvus-bricks/milvus_client/docs/brick-authoring.md`
- 创建: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`

**内容要求：**

`README.md` 包含：

- 目录结构。
- 快速运行 precheck。
- 如何创建 schema matrix。
- 如何 seed data。
- 如何跑 mixed_rw_pressure。
- JSON 输出协议。

`brick-authoring.md` 包含：

- 新 brick 必须支持的参数。
- 结果 JSON 示例。
- 错误码。
- checkpoint 规则。
- manifest 注册规则。

`upgrade-rollback.md` 包含：

- compat schema 和 forward schema 的区别。
- 升级/回滚期间 workload 不停止。
- Argo 外部动作如何接入。
- 失败定位方法。

## 任务 16: CI 和基础质量门禁

**文件：**
- 创建: `milvus-bricks/milvus_client/pytest.ini`
- 创建或修改: `.github/workflows/milvus-client-bricks.yml`

**步骤 1: pytest 配置**

```ini
[pytest]
testpaths = milvus_client/tests
python_files = test_*.py
```

**步骤 2: CI 命令**

```bash
cd milvus-bricks
PYTHONPATH=. python -m py_compile $(find milvus_client -name '*.py')
PYTHONPATH=. pytest milvus_client/tests -v
```

**步骤 3: smoke 可选门禁**

如果 CI 环境有 Milvus standalone：

```bash
PYTHONPATH=. python -m milvus_client.requests.precheck ...
PYTHONPATH=. python -m milvus_client.requests.create_schema_matrix ...
PYTHONPATH=. python -m milvus_client.requests.seed_data ...
PYTHONPATH=. python -m milvus_client.requests.validate_data_integrity ...
```

## 任务 17: 迁移现有脚本到新协议

**文件：**
- 修改: `milvus-bricks/milvus_client/search_permanently.py`
- 修改: `milvus-bricks/milvus_client/query_permanently_simplified.py`
- 修改: `milvus-bricks/milvus_client/query_iterator.py`
- 修改: `milvus-bricks/milvus_client/upsert3.py`
- 修改: `milvus-bricks/milvus_client/insert_perf_1.py`

**步骤 1: 保留旧 CLI**

先不要破坏现有 positional args。每个脚本增加：

```text
if args look like legacy positional args:
    run legacy path
else:
    run new argparse path
```

**步骤 2: 标记 deprecated**

legacy path 打印：

```text
WARNING: positional CLI is deprecated; use python -m milvus_client.requests.<name>
```

**步骤 3: 将核心逻辑迁到 requests/**

旧脚本只作为 wrapper，核心实现放到 `requests/`。

## 阶段交付顺序

### Phase 0: 目录和协议

完成任务 0-4。

验收：

- Milvus 3.0/Feishu 功能 inventory 文档存在。
- `milvus-bricks/milvus_client` 可 import。
- `precheck` 可输出包含 feature/capability/compat/lifecycle 字段的标准 JSON。
- 旧核心脚本 py_compile 通过。

### Phase 1: P0 基础 bricks

完成任务 5-11。

验收：

- 能创建 2.6 schema matrix。
- 能 seed deterministic data。
- 能独立 validate。
- 能跑 mixed rw pressure。
- `brick_catalog.yaml` 注册 P0 bricks。

### Phase 2: 升级回滚场景

完成任务 13-14。

验收：

- dry-run 输出完整步骤。
- 在真实环境中，upgrade/rollback action 由 Argo 控制，mixed workload 和 validator 持续运行。
- compat schema 在 2.6 -> 3.0 -> 2.6 后无 count/checksum drift。

### Phase 3: 3.0 功能扩展

完成任务 12 的 P1 列表。

验收：

- 每个功能 brick 至少有 L0 smoke。
- 不支持的版本输出 `status=skipped`，而不是失败。
- 3.0-only 功能不混入 rollback must-pass 集合。

## 验收标准

整体完成后，以下命令必须可运行：

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests -v
PYTHONPATH=. python -m milvus_client.requests.precheck ...
PYTHONPATH=. python -m milvus_client.requests.create_schema_matrix ...
PYTHONPATH=. python -m milvus_client.requests.seed_data ...
PYTHONPATH=. python -m milvus_client.requests.mixed_rw_pressure ...
PYTHONPATH=. python -m milvus_client.requests.validate_data_integrity ...
```

升级回滚场景通过标准：

- mixed read/write 压力在升级和回滚期间不中断。
- transient error 被分类记录，不吞异常。
- 最终 compat 集合 `COUNT_DRIFT=0`、`MISSING_PK=0`、`CHECKSUM_MISMATCH=0`。
- 3.0-only 集合只在升级成功后的 3.0 阶段验证。
- 所有 bricks 输出 JSON artifacts，Argo 可直接采集。

## 风险和处理

- **pymilvus API 版本差异：** 所有 3.0-only brick 先做 capability probe，不支持则 skip。
- **回滚兼容性边界不清：** 严格拆分 compat schema 和 forward schema。
- **旧脚本调用方被破坏：** 第一阶段保留 `milvus-bricks/2.6/` 不变；旧 CLI 迁移时再逐步增加 wrapper。
- **长压测失败难定位：** 每个 brick 都输出 failure type、operation、collection、pk/sample、exception。
- **Argo 与 brick 职责混乱：** brick 不做升级动作，只等待或响应外部 lifecycle signal。

## 建议首批 PR 拆分

1. `feat(milvus): add milvus client brick runtime alongside 2.6 scripts`
2. `feat(milvus-client): add brick protocol and result output`
3. `feat(milvus-client): add schema matrix and deterministic data seeding`
4. `feat(milvus-client): add data integrity validator`
5. `feat(milvus-client): add mixed rw pressure brick`
6. `feat(milvus-client): add upgrade rollback scenario dry-run`
7. `feat(milvus-client): add argo workflow templates`
8. `feat(milvus-client): add 3.0 feature matrix bricks`
