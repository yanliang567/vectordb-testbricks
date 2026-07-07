# Phase 2 Closed Loop and Independent Bricks Implementation Plan

**目标：** Complete the upgrade/rollback compatibility loop and split mixed workload operations into independent request bricks.

**架构：** Add a shared workload module that owns Milvus operations, counters, and JSON result behavior. Keep `mixed_rw_pressure` as a composed workload, add independent operation bricks as thin wrappers, and make the scenario runner execute a real closed loop with continuous pressure plus validation around external upgrade/rollback signals.

**技术栈：** Python 3, `pymilvus.MilvusClient`, YAML manifests, Argo WorkflowTemplate, pytest.

---

### 任务 1: Shared Workload Engine

**文件：**
- 创建: `milvus-bricks/milvus_client/common/workload.py`
- 修改: `milvus-bricks/milvus_client/requests/mixed_rw_pressure.py`
- 测试: `milvus-bricks/milvus_client/tests/test_workload.py`

**步骤：**
1. Move insert/upsert/query/search helpers into `common/workload.py`.
2. Add `run_pressure_workload()` with configurable operations, duration, workers, and batch size.
3. Keep operation failures as counters like `failed_search` rather than uncaught exceptions.
4. Update `mixed_rw_pressure` to call the shared runner with operations `insert,upsert,query,search`.

### 任务 2: Independent Request Bricks

**文件：**
- 创建: `requests/search_pressure.py`
- 创建: `requests/query_pressure.py`
- 创建: `requests/query_iterator_scan.py`
- 创建: `requests/upsert_pressure.py`
- 创建: `requests/delete_pressure.py`
- 修改: `manifests/brick_catalog.yaml`
- 测试: `tests/test_independent_bricks.py`

**步骤：**
1. Each brick uses `build_common_parser()` and writes `BrickResult`.
2. Search/query/upsert reuse shared workload operations.
3. Query iterator scans checkpoint PK ranges when checkpoint exists, otherwise runs bounded query scan.
4. Delete pressure deletes only high PK ranges reserved for pressure data, never seed baseline ranges.
5. Register all bricks in `brick_catalog.yaml`.

### 任务 3: Phase 2 Scenario Closed Loop

**文件：**
- 修改: `scenarios/upgrade_rollback_compatibility.py`
- 修改: `manifests/scenario_upgrade_rollback.yaml`
- 修改: `argo/upgrade-rollback-compatibility.yaml`
- 测试: `tests/test_upgrade_rollback_scenario.py`
- 测试: `tests/test_argo_template.py`

**步骤：**
1. Keep dry-run plan with compat setup, continuous workload, validator loop, N upgrade/rollback cycles, forward schema creation after upgrade, and compat-only validation after rollback.
2. Implement non-dry-run scenario executor using subprocess calls to request bricks.
3. Start continuous mixed RW pressure and validator loop before upgrade wait, stop them after final compat validation.
4. Validate forward schema only in after-upgrade phases.
5. Make Argo run the scenario controller as the closed-loop entrypoint while preserving result/checkpoint artifacts.

### 任务 4: Docs and Verification

**文件：**
- 修改: `milvus_client/README.md`
- 创建: `milvus_client/docs/upgrade-rollback.md`
- 修改: `docs/plans/2026-07-07-milvus-client-bricks-expansion.md`

**验证命令：**
```bash
cd milvus-bricks
PYTHONPATH=. /tmp/vectordb-testbricks-venv/bin/python -m pytest milvus_client/tests -v
PYTHONPATH=. /tmp/vectordb-testbricks-venv/bin/python -m py_compile $(find milvus_client -name '*.py' -not -path '*/.pytest_cache/*')
/tmp/vectordb-testbricks-venv/bin/python -m py_compile $(find 2.6 -name '*.py')
argo lint argo/upgrade-rollback-compatibility.yaml
python3 - <<'PY'
from pathlib import Path
import yaml
for path in list(Path('milvus_client/manifests').glob('*.yaml')) + list(Path('argo').glob('*.yaml')):
    yaml.safe_load(path.read_text())
print('yaml ok')
PY
```
