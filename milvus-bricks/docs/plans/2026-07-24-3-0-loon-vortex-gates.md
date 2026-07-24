# 3.0 LoonFFI/Vortex 回滚 Gate 实现计划

**目标：** 新增 standalone 和 cluster 两条 `3.0 baseline -> 3.0 latest + LoonFFI/Vortex -> 3.0 baseline` 升级回滚 gate path。

**架构：** 继续以 `upgrade_rollback_gates.yaml` 作为中心化场景入口，不改 Argo DAG。场景通过现有参数 `target-loon-ffi-enabled` / `target-vortex-enabled` / `rollback-loon-ffi-enabled` / `rollback-vortex-enabled` 控制 3.0-only 存储特性；2.6 rollback gate 的 StorageV3/Vortex guard 不放宽。

**技术栈：** Python manifest renderer, Pytest, Argo WorkflowTemplate YAML, Helm-based cluster deploy profile。

---

### 任务 1: 固定 3.0 baseline 镜像

**文件：**
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试: `milvus_client/tests/test_render_upgrade_rollback_params.py`

**步骤 1: 更新 image alias**

把 `milvus-3-0-baseline` 从 placeholder 改成 Harbor 可部署镜像：

```yaml
milvus-3-0-baseline:
  image: harbor.milvus.io/milvusdb/milvus:3.0-20260723-77b26a50
  version: "3.0.0"
```

**步骤 2: 更新参数渲染断言**

把 3.0 gate 的 base/rollback image 预期从 placeholder 改成上述固定 tag。

**步骤 3: 运行定向测试**

运行:

```bash
PYTHONPATH=. python3 -m pytest -q \
  milvus_client/tests/test_render_upgrade_rollback_params.py::test_render_cluster_3_0_gate_parameters
```

预期: PASS。

### 任务 2: 注册 standalone 3.0 LoonFFI/Vortex gate

**文件：**
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 测试: `milvus_client/tests/test_render_upgrade_rollback_params.py`

**步骤 1: 添加 scenario**

新增：

```yaml
- id: standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline
  mode: standalone
  classification: gate
  support_status: supported
  workflow_template_ref: standalone_3_0
  deploy_profile_ref: standalone
  schema_matrix_ref: "3.0"
  base:
    image_ref: milvus-3-0-baseline
    json_shredding_enabled: false
    vortex_enabled: false
  target:
    image_ref: milvus-3-0-latest
    json_shredding_enabled: false
    loon_ffi_enabled: true
    vortex_enabled: true
  rollback:
    image_ref: milvus-3-0-baseline
    json_shredding_enabled: false
    loon_ffi_enabled: true
    vortex_enabled: true
```

说明：rollback 阶段回退的是 baseline image，但 LoonFFI/Vortex 配置保持开启，用来验证 3.0 baseline 能否读取/服务升级阶段写入的新存储格式。

**步骤 2: 添加 manifest 断言**

断言该 scenario 是 gate，target/rollback 两阶段的 `loon_ffi_enabled` 和 `vortex_enabled` 都为 `true`，且不设置 `allow_unsafe_negative_coverage`。

**步骤 3: 添加 renderer 断言**

断言渲染出的 Argo 参数：

```text
target-loon-ffi-enabled=true
target-vortex-enabled=true
rollback-loon-ffi-enabled=true
rollback-vortex-enabled=true
allow-unsafe-negative-coverage=false
```

### 任务 3: 注册 cluster 3.0 LoonFFI/Vortex gate

**文件：**
- 修改: `milvus_client/manifests/upgrade_rollback_gates.yaml`
- 测试: `milvus_client/tests/test_upgrade_rollback_gates_manifest.py`
- 测试: `milvus_client/tests/test_render_upgrade_rollback_params.py`

**步骤 1: 添加 scenario**

新增：

```yaml
- id: cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline
  mode: cluster
  classification: gate
  support_status: supported
  workflow_template_ref: cluster
  deploy_profile_ref: cluster_woodpecker_1cu
  schema_matrix_ref: "3.0"
  base:
    image_ref: milvus-3-0-baseline
    json_shredding_enabled: false
    vortex_enabled: false
  target:
    image_ref: milvus-3-0-latest
    json_shredding_enabled: false
    loon_ffi_enabled: true
    vortex_enabled: true
  rollback:
    image_ref: milvus-3-0-baseline
    json_shredding_enabled: false
    loon_ffi_enabled: true
    vortex_enabled: true
```

**步骤 2: 更新 cluster gate 数量断言**

把 cluster gate 数量从 `2` 改成 `3`，并断言新增 scenario 使用 `milvus-cluster-upgrade-rollback` 和 `cluster-woodpecker-1cu.yaml`。

### 任务 4: 更新执行文档

**文件：**
- 创建: `docs/upgrade-rollback-gates/README.md`
- 修改: `milvus_client/README.md`

**步骤 1: 创建 gate README**

文档列出当前 7 条 scenario：

- 6 条 gate
- 1 条 negative

并说明中心化改动点：

- 版本/image：`milvus_client/manifests/upgrade_rollback_gates.yaml` 的 `image_aliases`
- path：同文件的 `scenarios`
- workload 数据量：同文件的 `defaults`
- cluster topology：`milvus_client/manifests/deploy_profiles/*.yaml`
- WorkflowTemplate：`argo/*.yaml`

**步骤 2: 补充 submit 示例**

给出 standalone 和 cluster 两条新增 path 的 `render_upgrade_rollback_params` 示例，提醒 `milvus-3-0-latest` 仍需用具体 image 覆盖或更新 alias，避免 placeholder。

### 任务 5: 验证和提交

**文件：**
- 所有改动文件

**步骤 1: 运行单测**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_upgrade_rollback_params.py
```

**步骤 2: 运行全量相关校验**

```bash
PYTHONPATH=. python3 -m pytest -q milvus_client/tests
uvx ruff check milvus_client/common/gates.py milvus_client/requests/render_upgrade_rollback_params.py milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_upgrade_rollback_params.py
uvx ruff format --check milvus_client/common/gates.py milvus_client/requests/render_upgrade_rollback_params.py milvus_client/tests/test_upgrade_rollback_gates_manifest.py milvus_client/tests/test_render_upgrade_rollback_params.py
argo lint argo/standalone-2-6-upgrade-rollback.yaml
argo lint argo/standalone-3-0-upgrade-rollback.yaml
argo lint argo/cluster-upgrade-rollback.yaml
git diff --check
```

**步骤 3: 提交**

```bash
git add milvus_client/manifests/upgrade_rollback_gates.yaml \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  milvus_client/README.md docs/upgrade-rollback-gates/README.md \
  docs/plans/2026-07-24-3-0-loon-vortex-gates.md
git commit -m "feat: add 3.0 loon vortex rollback gates"
```
