# 4am Milvus 2.6 Upgrade/Rollback Workflow 实现计划

**目标：** 固化一个 4am Argo workflow，用 4c16g standalone 从 v2.6.18 升级到 4am 最新 2.6 镜像，再回滚并验证存量数据完整性。

**架构：** 新增独立 WorkflowTemplate 负责 Milvus CR 生命周期，复用 `milvus_client.requests` 做 schema、seed、pressure 和 validation。Pressure 以 daemon 方式在升级和回滚期间持续运行，同步 validation 在升级前、升级后和回滚后作为强校验。

**技术栈：** Argo Workflows, Milvus Operator CRD, Python 3.11, pymilvus, pytest, pyyaml。

---

### 任务 1: 新增 4am standalone WorkflowTemplate

**文件：**
- 创建: `milvus-bricks/argo/standalone-2-6-upgrade-rollback.yaml`

**步骤 1: 固化 Milvus CR 创建和 image patch**

从实跑 workflow 提取 `deploy-milvus`、`wait-milvus-ready`、`patch-milvus-image` 和 `maybe-cleanup` 模板。

**步骤 2: 接入 request bricks**

复用 `run-brick` 模板依次运行：
- `precheck`
- `create_schema_matrix`
- `seed_data`
- `validate_data_integrity`

**步骤 3: 接入 daemon pressure**

新增 `pressure-daemon`，在升级前启动并覆盖升级、观察、回滚、回滚后观察阶段。`validate_data_integrity` 保持前台同步步骤，避免 pressure 和 validation 并发挂载同一个 ReadWriteOnce checkpoint PVC。

### 任务 2: 补模板测试

**文件：**
- 修改: `milvus-bricks/milvus_client/tests/test_argo_template.py`

**步骤 1: 断言新模板参数**

校验 base image、target image、schema matrix、cleanup、pressure 参数存在。

**步骤 2: 断言闭环任务**

校验 DAG 包含 deploy、wait、seed、validate、pressure daemon、upgrade、rollback、final validation。

### 任务 3: 更新文档

**文件：**
- 修改: `milvus-bricks/milvus_client/docs/upgrade-rollback.md`
- 修改: `milvus-bricks/milvus_client/README.md`

**步骤 1: 说明 2.6-only workflow**

记录这次 workflow 不覆盖 3.0 schema，因为 3.0 新数据不支持回滚。

**步骤 2: 说明已覆盖 request 和类型**

列出 `precheck`、`create_schema_matrix`、`seed_data`、`validate_data_integrity`、pressure bricks，以及当前 2.6 schema 覆盖的数据类型。

### 任务 4: 验证

**命令：**

```bash
cd milvus-bricks
PYTHONPATH=. pytest milvus_client/tests/test_argo_template.py milvus_client/tests/test_independent_bricks.py milvus_client/tests/test_upgrade_rollback_scenario.py -v
PYTHONPATH=. pytest milvus_client/tests -v
```

**预期：** 所有测试通过。
