# Cluster pressure maintenance-window exclusion 实现计划

**目标：** 让 cluster upgrade/rollback gate 的 pressure 汇总排除 workflow 自己制造的 rollout/schema 维护窗口失败，避免健康集群被误判。

**架构：** 复用 standalone WorkflowTemplate 已验证的 `check-pressure-results` 逻辑：读取 Argo Workflow node 时间，构造 maintenance windows，对窗口内的 transient connectivity/request failure 标记为 excluded。业务 validation、serviceability 和 pressure 原始结果保持不变。

**技术栈：** Argo Workflow YAML、Python inline script、pytest、argo lint。

---

### 任务 1: 同步 cluster pressure exclusion

**文件：**
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 修改: `milvus_client/tests/test_argo_template.py`

**步骤 1: 更新测试期望**

将 standalone-only 断言扩展到 cluster 模板，要求三个模板都包含 maintenance window exclusion 字段。

**步骤 2: 更新 cluster check-pressure-results**

把 `argo/standalone-3-0-upgrade-rollback.yaml` 中的 `_maintenance_windows()`、`_overlap_window()`、`_is_transient_connectivity_failure()` 和 `excluded_failed_results` 汇总逻辑同步到 `argo/cluster-upgrade-rollback.yaml`。

**步骤 3: 运行验证**

运行：

```bash
python3 -m pytest milvus_client/tests/test_argo_template.py -q
argo lint argo/cluster-upgrade-rollback.yaml
git diff --check
```

预期：全部通过。

**步骤 4: 应用并重跑**

运行：

```bash
kubectl apply -n qa -f argo/cluster-upgrade-rollback.yaml
```

然后重新提交 cluster 3.0 gate。
