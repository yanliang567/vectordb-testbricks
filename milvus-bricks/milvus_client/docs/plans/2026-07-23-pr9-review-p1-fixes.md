# PR9 Review P1 Fixes 实现计划

**目标：** 修复 PR9 review 提出的三个 P1：index rebuild 与 pressure 冲突、upsert 未验证更新语义、rollback 后未验证同一批实际索引元数据。

**架构：** Promoted gate 不在 pressure-daemon 运行期间重建 baseline collection 索引；`validate_index_compatibility` 默认只 flush/load/search/query 并记录实际 index metadata，回滚后按 checkpoint 比较同一批索引。`validate_phase_dml_dql` 在显式 PK schema 上查询 upsert 后样本字段，证明服务端确实更新了数据。

**技术栈：** Python request brick、MilvusClient、Argo WorkflowTemplate、pytest、Argo lint。

---

### 任务 1: 去除 promoted gate 的索引重建窗口

**文件：**
- 修改: `argo/standalone-2-6-upgrade-rollback.yaml`
- 修改: `argo/standalone-3-0-upgrade-rollback.yaml`
- 修改: `argo/cluster-upgrade-rollback.yaml`
- 修改: `milvus_client/tests/test_argo_template.py`
- 修改: `milvus_client/docs/upgrade-rollback.md`
- 修改: `milvus_client/docs/upgrade-rollback-gates/README.md`

**步骤：**
1. 将 after-upgrade `validate-index-compatibility` task 的 `--rebuild-index true` 改为 `false`。
2. 保留 CLI 参数 `--rebuild-index` 作为手工诊断能力，不作为 promoted gate 默认。
3. 测试断言 promoted gate 不会 drop/create index。

### 任务 2: 验证 upsert 确实更新

**文件：**
- 修改: `milvus_client/requests/validate_phase_dml_dql.py`
- 修改: `milvus_client/tests/test_validate_phase_dml_dql.py`

**步骤：**
1. 对显式 PK collection，upsert 后查询未删除样本 PK。
2. 比较 query 返回字段和 `seed + 101` 生成的预期值。
3. fake client 默认维护行状态；新增 no-op upsert 负向测试，修复前应失败。

### 任务 3: 记录和比较实际索引元数据

**文件：**
- 修改: `milvus_client/requests/validate_index_compatibility.py`
- 修改: `milvus_client/tests/test_validate_index_compatibility.py`

**步骤：**
1. after-upgrade 使用 `list_indexes` / `describe_index` 收集实际 index name、field、index_type、metric_type、params。
2. checkpoint 保存实际索引列表，不再只保存 schema expected fields。
3. after-rollback 重新收集实际索引列表，与 checkpoint 逐项比较。
4. 对 schema 中标量 index 字段执行字段过滤 query，覆盖 BITMAP/INVERTED/STL_SORT/TRIE/NGRAM/JSON/ARRAY 等非向量索引服务路径。

### 任务 4: 验证和提交

运行：

```bash
PYTHONPATH=. pytest milvus_client/tests -q
uvx ruff check <changed python files>
uvx ruff format --check <changed python files>
argo lint argo/standalone-2-6-upgrade-rollback.yaml argo/standalone-3-0-upgrade-rollback.yaml argo/cluster-upgrade-rollback.yaml
git diff --check
```

预期：全部通过后提交并推送 PR #9，更新 PR description。
