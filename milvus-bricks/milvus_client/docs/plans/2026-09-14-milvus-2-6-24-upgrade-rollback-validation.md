# Milvus 2.6.24 升级回滚验证计划

**目标：** 使用 4am Harbor 最新 2.6 分支镜像作为 2.6.24 candidate，验证 `v2.6.18 -> candidate -> 2.6-latest` 的 standalone 与 cluster 升级、回滚和数据/服务可用性。

**架构：** 复用 vectordb-testbricks 现有 Argo 升级/回滚场景，跳过所有 3.0 相关场景；固定镜像 tag 与 digest，使用相同 schema、压力和检查点对两种 topology 执行验证。standalone 并发上限 3，cluster 并发上限 2；失败时按日志、Pod 状态和监控逐层定位，区分 Milvus 产品缺陷与测试阻塞。

**技术栈：** Argo Workflows、Kubernetes、Grafana Loki/Prometheus、Milvus Python client、Harbor v2 API。

---

### 任务 1：锁定 candidate 镜像与测试输入

**文件：**
- 读取: `milvus-bricks/milvus_client/argo/` 下现有升级回滚模板与参数
- 创建: 本报告 `docs/reports/2026-09-14-milvus-2-6-24-upgrade-rollback-validation.md`

**步骤 1：** 查询 4am Harbor 最新可部署 `2.6-` 多架构镜像，排除 e2e/架构专用 tag。

**步骤 2：** 固定 `v2.6.18` baseline、candidate digest、`2.6-latest` rollback digest、测试 revision，并记录在报告。

**步骤 3：** 渲染参数并验证 JSON、shell syntax 与并发限制。

### 任务 2：运行 standalone 升级/回滚矩阵

**步骤 1：** 仅提交 2.6 相关 standalone 场景，client-side 并发不超过 3。

**步骤 2：** 观察 Argo 状态、Pod rollout、压力和最终报告；失败时保留 workflow、日志和监控证据。

**步骤 3：** 对每个完成 workflow 核对功能检查、回滚后数据/索引、serviceability、restartCount 和失败 slice。

### 任务 3：运行 cluster 升级/回滚矩阵

**步骤 1：** 仅提交 2.6 相关 cluster 场景，client-side 并发不超过 2。

**步骤 2：** 同步核对 QueryNode/DataNode/StreamingNode/Proxy 状态、pressure 结果、Loki panic 和 Prometheus restart 指标。

### 任务 4：问题归因与必要修复

**步骤 1：** 对每个失败建立单一根因假设，使用最小复现和源码/日志/监控交叉验证。

**步骤 2：** 测试 harness 阻塞问题直接修复并回归；Milvus 产品问题只提交 issue，不将测试绕过当作修复；优化项单独记录。

### 任务 5：完成报告与最终验证

**步骤 1：** 汇总镜像、场景、Argo 链接、结果、失败归因、证据和发布建议。

**步骤 2：** fresh 验证所有 workflow terminal status、并发约束、报告内容、`git diff --check`，再交付报告路径。
