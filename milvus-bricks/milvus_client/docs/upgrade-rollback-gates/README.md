# Milvus Upgrade/Rollback Gate 执行说明

这份 README 面向需要执行或维护 4am QA 升级/回滚 gate 的同学。更完整的设计细节见
[`../upgrade-rollback.md`](../upgrade-rollback.md)。

## 目标

这套 workflow 验证 Milvus 在升级和回滚过程中的数据兼容性、服务可恢复性和持续读写稳定性。

当前覆盖两类部署形态：

- `standalone`：轻量兼容性 gate，适合快速验证数据和功能回归。
- `cluster`：分布式组件形态 gate，覆盖 `mixCoord/proxy/queryNode/dataNode/streamingNode` 以及 Woodpecker MQ。

## 当前 gate 场景

场景定义的唯一入口是
[`../../manifests/upgrade_rollback_gates.yaml`](../../manifests/upgrade_rollback_gates.yaml)。

| Scenario ID | Topology | Path | Gate 数据 | 备注 |
| --- | --- | --- | --- | --- |
| `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest` | standalone | `2.6.18 -> latest 3.0 -> latest 2.6` | 2.6 rollback-safe schema/data | 正式 gate；latest 2.6 必须包含 #50792。3.0 阶段必须禁用 storage v3/vortex。 |
| `standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | standalone | `3.0 baseline -> latest 3.0 -> 3.0 baseline` | 3.0 schema/data | 3.0 branch 内回滚 gate。 |
| `cluster-2-6-18-to-3-0-latest-rollback-2-6-latest` | cluster | `2.6.18 -> latest 3.0 -> latest 2.6` | 2.6 rollback-safe schema/data | cluster 正式 gate；latest 2.6 必须包含 #50792。3.0 阶段必须禁用 storage v3/vortex。 |
| `cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline` | cluster | `3.0 baseline -> latest 3.0 -> 3.0 baseline` | 3.0 schema/data | cluster 模式下的 3.0 branch gate。 |

另外保留以下非正式 gate 场景：

- `standalone-2-6-18-to-3-0-latest-rollback-2-6-18`
- `cluster-2-6-18-to-3-0-latest-rollback-2-6-18`

它们用于诊断 pre-#50792 的 rollback boundary。`v2.6.18` 有 coordinator
session version guard，但不包含 #50792 的 `3.0.x -> 2.6.x` rollback 例外逻辑；
recent 3.0 image 写入 `meta/session/version=3.0.0-beta` 后，回滚到 `v2.6.18`
会命中 #50694 并在 coordinator session 注册时 panic。因此它们不作为发布 gate。

另外保留一个 unsafe negative 场景：

- `standalone-3-0-loon-vortex-to-2-6-negative`

它用于记录 `3.0 LoonFFI/vortex -> 2.6` 的 unsafe boundary，不能作为发布 gate。该场景在 manifest 中显式设置 `allow_unsafe_negative_coverage: true`，渲染为 Argo 参数 `allow-unsafe-negative-coverage=true`，用于跳过运行时 gate guard 并复现 unsupported rollback boundary。运行时 guard 同时校验 `scenario-id` 必须是批准的 negative 场景 `standalone-3-0-loon-vortex-to-2-6-negative`；正式 gate 场景默认保持 `false`，手工打开会直接失败。

## 必须遵守的安全约束

`2.6.18 -> 3.0 latest -> rollback latest 2.6` 是产品正式支持路径，但前提是升级到 3.0 后不能开启 storage v3 或 vortex。rollback target 必须使用包含 #50792 的 2.6 构建；不要把 `v2.6.18` 作为正向 gate rollback target。

因此 2.6 rollback gate 必须保持：

```text
base.loon_ffi_enabled=false
target.loon_ffi_enabled=false
rollback.loon_ffi_enabled=false
base.vortex_enabled=false
target.vortex_enabled=false
rollback.vortex_enabled=false
```

Milvus 3.0 的 StorageV3 有效开关是 `common.storage.useLoonFFI`，在 manifest/Argo 参数中统一表示为 `loon_ffi_enabled` / `*-loon-ffi-enabled`；不存在单独的 `storageV3Enabled` CR 配置键。代码层面已有静态校验：如果 promoted gate 是 `2.6 -> 3.0 -> 2.6`，但 base、target 或 rollback 任一阶段开启了 `loon_ffi_enabled` 或 `vortex_enabled`，参数渲染会直接失败。manifest validator 还会强制这些开关必须是 YAML boolean，避免 `"false"` 字符串被误解析。所有可运行场景默认都会拒绝 placeholder 镜像；只有生成 dry-run/review 参数时才使用 `--allow-placeholder`。

## 目录结构

关键文件如下：

```text
milvus-bricks/
  argo/
    standalone-2-6-upgrade-rollback.yaml
    standalone-3-0-upgrade-rollback.yaml
    standalone-2-6-upgrade-rollback-rbac.yaml
    cluster-upgrade-rollback.yaml
    cluster-upgrade-rollback-rbac.yaml
  milvus_client/
    common/
      deploy.py
      gates.py
    requests/
      render_milvus_cr.py
      render_upgrade_rollback_params.py
      generate_workflow_report.py
    manifests/
      upgrade_rollback_gates.yaml
      schema_matrix_2_6.yaml
      schema_matrix_3_0.yaml
      deploy_profiles/
        standalone-rocksmq.yaml
        cluster-woodpecker-1cu.yaml
        cluster-woodpecker-2cu.yaml
```

职责边界：

- Argo YAML 负责编排流程。
- `upgrade_rollback_gates.yaml` 负责维护场景、版本路径、schema/profile/workflow 引用。
- deploy profile 负责维护部署 topology：standalone 渲染 Milvus Operator CR，cluster Woodpecker 渲染 Helm chart values。
- schema matrix 负责维护字段、索引和 workload 数据覆盖。
- renderer 负责把 manifest/profile 转成 Argo 参数、Milvus CR 或 Helm values。

## 执行前准备

确认本地在 `milvus-bricks/` 目录：

```bash
cd milvus-bricks
```

确认工具可用：

```bash
kubectl config current-context
argo version
python3 --version
```

确认 namespace：

- Argo workflow namespace：`qa`
- Milvus namespace：`qa-milvus`

应用 RBAC：

```bash
kubectl apply -f argo/standalone-2-6-upgrade-rollback-rbac.yaml
kubectl apply -f argo/cluster-upgrade-rollback-rbac.yaml
```

应用或更新 WorkflowTemplate：

```bash
argo template create argo/standalone-2-6-upgrade-rollback.yaml -n qa
argo template create argo/standalone-3-0-upgrade-rollback.yaml -n qa
argo template create argo/cluster-upgrade-rollback.yaml -n qa
```

如果模板已存在，用 `argo template update`。

## 正式运行前：先替换 latest/baseline placeholder

manifest 中的这些 alias 可能是 placeholder：

```yaml
image_aliases:
  milvus-2-6-latest:
    image: harbor.milvus.io/milvusdb/milvus:2.6-latest-placeholder
  milvus-3-0-baseline:
    image: harbor.milvus.io/milvusdb/milvus:3.0-baseline-placeholder
  milvus-3-0-latest:
    image: harbor.milvus.io/milvusdb/milvus:3.0-latest-placeholder
```

正式 gate 不要直接跑 placeholder。参数渲染器默认会拒绝 promoted gate 中的
placeholder image。先把它们替换成具体 Harbor tag，例如：

```yaml
milvus-3-0-latest:
  image: harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<sha>
  version: "3.0.0"
```

`version` 是 Milvus Operator 识别 daily build 所需的版本字段，不一定等于 Docker tag。

## 生成 Argo 参数

用 scenario id 生成提交参数：

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id standalone-2-6-18-to-3-0-latest-rollback-2-6-latest \
  --format argo-args
```

如果只是 review/dry-run 当前 manifest 中的 placeholder 参数，显式加
`--allow-placeholder`：

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id standalone-2-6-18-to-3-0-latest-rollback-2-6-latest \
  --format argo-args \
  --allow-placeholder
```

也可以输出 JSON，适合自动化系统读取：

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline \
  --format json \
  --allow-placeholder \
  --output /tmp/upgrade-rollback-params.json
```

如果临时想用 2CU cluster profile，不改 scenario 也可以 override：

```bash
PYTHONPATH=. python3 -m milvus_client.requests.render_upgrade_rollback_params \
  --scenario-id cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline \
  --deploy-profile milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml \
  --format argo-args \
  --allow-placeholder
```

## 提交 workflow

不要把 `--format argo-args` 的输出放进 shell substitution，因为 `pressure-modules` 参数包含空格。推荐先生成并检查参数，再复制执行。

示例：standalone `2.6.18 -> 3.0 latest -> latest 2.6`

```bash
argo submit -n qa \
  --from workflowtemplate/milvus-standalone-2-6-upgrade-rollback \
  -p scenario-id=standalone-2-6-18-to-3-0-latest-rollback-2-6-latest \
  -p deploy-profile=milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml \
  -p base-milvus-image=harbor.milvus.io/milvusdb/milvus:v2.6.18 \
  -p base-version=2.6.18 \
  -p target-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<sha> \
  -p target-version=3.0.0 \
  -p rollback-milvus-image=harbor.milvus.io/milvusdb/milvus:2.6-YYYYMMDD-<sha> \
  -p rollback-version=2.6.0 \
  -p base-loon-ffi-enabled=false \
  -p target-loon-ffi-enabled=false \
  -p rollback-loon-ffi-enabled=false \
  -p base-vortex-enabled=false \
  -p target-vortex-enabled=false \
  -p rollback-vortex-enabled=false \
  -p target-json-shredding-enabled=false \
  -p rollback-enabled=true \
  -p 'pressure-modules=search_pressure query_pressure query_iterator_scan count_pressure upsert_pressure delete_pressure mixed_rw_pressure' \
  -p keep-milvus=false
```

示例：cluster `3.0 baseline -> 3.0 latest -> 3.0 baseline`

```bash
argo submit -n qa \
  --from workflowtemplate/milvus-cluster-upgrade-rollback \
  -p scenario-id=cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline \
  -p deploy-profile=milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml \
  -p base-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<baseline-sha> \
  -p base-version=3.0.0 \
  -p target-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<latest-sha> \
  -p target-version=3.0.0 \
  -p rollback-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<baseline-sha> \
  -p rollback-version=3.0.0 \
  -p rollback-enabled=true \
  -p keep-milvus=false
```

## Workflow 会做什么

每个 promoted gate 的主流程：

1. 部署 base Milvus。standalone 使用 Milvus Operator CR；cluster Woodpecker 使用 Milvus Helm chart。
2. 等待 Milvus ready。
3. 创建 schema matrix 中的集合。
4. 写入 deterministic seed data，并保存 checkpoint。
5. 升级前做数据完整性校验。
6. 启动 background pressure daemon。
7. 升级到 target image。
8. 升级后做 precheck、数据完整性校验、foreground pressure。
9. 按场景决定是否执行 schema evolution / forward workload。
10. 回滚到 rollback image。
11. 回滚后等待数据 serviceability 恢复。
12. 回滚后做数据完整性校验和 foreground pressure。
13. 收集 K8s snapshot、pressure result、checkpoint、最终报告。
14. 按 `keep-milvus` 决定是否清理 Milvus CR/Helm release 和依赖资源。

## 报告和 artifacts

workflow 会导出这些核心 artifact：

- `final_report.md`
- `orchestrator_report.json`
- `flow_summary.json`
- `env_snapshot.json`
- `deploy_topology.json`
- foreground brick result JSON
- pressure result JSON
- `pressure-summary.json`
- checkpoint 文件
- K8s resource snapshot

最终报告会记录：

- scenario id
- workflow template
- deploy profile
- deploy topology
- base/target/rollback image 和 version
- schema matrix
- storage/jsonShredding/LoonFFI 参数
- rollback serviceability 恢复耗时
- pressure 和 data integrity 结果

## 如何新增 3.1、4.0 或改版本

优先只改 manifest，不改 workflow。

### 只换具体 image tag

改
[`../../manifests/upgrade_rollback_gates.yaml`](../../manifests/upgrade_rollback_gates.yaml)
里的 `image_aliases`：

```yaml
image_aliases:
  milvus-3-0-latest:
    image: harbor.milvus.io/milvusdb/milvus:3.0-YYYYMMDD-<sha>
    version: "3.0.0"
```

所有引用 `milvus-3-0-latest` 的 scenario 会自动使用新 tag。

### 新增 3.1 branch gate

通常改三处：

1. 新增 image alias：

   ```yaml
   image_aliases:
     milvus-3-1-baseline:
       image: harbor.milvus.io/milvusdb/milvus:3.1-YYYYMMDD-<baseline-sha>
       version: "3.1.0"
     milvus-3-1-latest:
       image: harbor.milvus.io/milvusdb/milvus:3.1-YYYYMMDD-<latest-sha>
       version: "3.1.0"
   ```

2. 如果 schema 变了，新增 matrix 并注册：

   ```yaml
   schema_matrices:
     "3.1": milvus_client/manifests/schema_matrix_3_1.yaml
   ```

3. 新增 scenario：

   ```yaml
   - id: standalone-3-1-baseline-to-3-1-latest-rollback-3-1-baseline
     mode: standalone
     classification: gate
     support_status: supported
     workflow_template_ref: standalone_3_0
     deploy_profile_ref: standalone
     schema_matrix_ref: "3.1"
     collection_prefix: qa_gate_31_to_31latest
     base:
       image_ref: milvus-3-1-baseline
     target:
       image_ref: milvus-3-1-latest
       loon_ffi_enabled: false
       vortex_enabled: false
   rollback:
     image_ref: milvus-3-1-baseline
   validation_policy:
     data_integrity: strict
     serviceability: strict
     pressure_fail_on_error: true
     gate_allow_warning: false
   ```

如果 3.1 的升级/回滚流程和 3.0 一样，workflow YAML 不需要改。

### 新增 4.0 branch gate

按 3.1 同样方式新增 alias、schema matrix 和 scenario。只有出现以下情况才改 workflow：

- 需要新增 DAG 阶段。
- 需要新增 runtime 参数。
- Milvus Operator CR 或 Helm chart values 结构变化，deploy renderer 需要新字段。
- 回滚安全边界变化，需要新增静态校验。

### 新增 deployment topology

新增或修改 deploy profile：

```text
milvus_client/manifests/deploy_profiles/
```

例如新增 `cluster-woodpecker-4cu.yaml` 后，在 manifest 的 `deploy_profiles` 注册：

```yaml
deploy_profiles:
  cluster_woodpecker_4cu: milvus_client/manifests/deploy_profiles/cluster-woodpecker-4cu.yaml
```

然后 scenario 改 `deploy_profile_ref` 即可。

## 本地校验

改 manifest、profile、renderer 或 workflow 后，至少跑：

```bash
python3 -m pytest milvus_client/tests -q
argo template lint \
  argo/standalone-2-6-upgrade-rollback.yaml \
  argo/standalone-3-0-upgrade-rollback.yaml \
  argo/cluster-upgrade-rollback.yaml \
  -n qa \
  -o simple
```

只改 gate manifest 时，快速检查：

```bash
python3 -m pytest \
  milvus_client/tests/test_upgrade_rollback_gates_manifest.py \
  milvus_client/tests/test_render_upgrade_rollback_params.py \
  -q
```

## 常见问题

### 为什么 2.6 rollback gate 不跑 3.0-only schema？

因为 3.0-only schema/data 不要求回滚到 2.6 后仍可用。2.6 rollback gate 的硬门禁是 2.6 rollback-safe 数据在升级和回滚后仍完整、可查询、可承压。

### 为什么 latest 要提前解析成具体 tag？

正式 gate 需要可复现。`latest` 是运行时变化的概念，报告和失败定位需要具体 Docker image tag。

### standalone 和 cluster 都要跑吗？

需要。standalone 快速覆盖数据兼容性和功能回归；cluster 额外覆盖分布式组件滚动、serviceability 和 Woodpecker 拓扑。

### 什么时候设置 `keep-milvus=true`？

只在调试失败现场时设置。默认 `keep-milvus=false`，workflow 会清理自己创建的 Milvus CR 或 Helm release、PVC 和依赖资源。
