# Release Candidate 验证工作流（3.0.x 发布前验证）

**日期：** 2026-08-18

## 1. 背景与决策

正式发布 3.0.x 后再做升级/回滚验证来不及。因此**后续统一用「3.0 分支最新的
daily image」作为「下一个 3.0.x 版本的 release candidate」来做测试验证**；
只有当 candidate 跑通升级/回滚 gate（数据/索引/serviceability/pressure 全绿）
之后，才正式发布 3.0.x。

这意味着：测试时使用的 Milvus 镜像，其**服务端语义版本号仍然是 `3.0.0`**
（3.0.x 尚未 bump/打 tag），但它已经包含下一个 3.0.x 的功能（如 Vortex 0.75+、
#52359 修复等）。我们把这种镜像称为 **release candidate build**。

## 2. 当前框架的阻断点

`upgrade_rollback_gates.yaml` 里 `milvus-3-0-1` 别名声明 `version: "3.0.1"`，
这是为了满足 Vortex guard（`Vortex 要求 >= 3.0.1`）的下限。但当把它覆盖成
`3.0-20260817-beb93ec2` 这类 daily build 时：

1. **precheck** 里 `version_at_least(server_version, expected_version)` 用
   `version_core("3.0-20260817-beb93ec2a7") == (3,0,0)` 与 `(3,0,1)` 比较，
   触发 `SERVER_VERSION_TOO_OLD`。
2. 因此声明 version=`3.0.1` 的新 gate 场景（vortex/loon 等）在 3.0.x 正式
   bump 版本号之前无法跑 E2E。

这正是当初设计「3.0.1 gate 不能用 3.0.0 冒充」的保护，但对 release candidate
验证流程是误伤。

## 3. 目标框架语义

引入「release candidate 镜像」概念，使声明 `version: "3.0.1"` 的 gate 场景可以
对「版本号仍是 3.0.0、但已具备 3.0.1 功能」的 candidate daily build 运行，同时
保留跨版本硬边界（2.6 读不了 v3/vortex；v3.0.0 读不了 v3.0.1 升级后的 vortex
编码 #52340）。

## 4. 框架改动（已实现）

### 4.1 `version.py`：识别 daily/branch build tag

新增 `DAILY_BUILD_TAG` 正则与 `image_tag` / `is_daily_build_image`：

```text
^v?\d+\.\d+-\d{8}-[0-9a-fA-F]{8,40}(?:-[a-zA-Z0-9]+)?$
```

匹配 `3.0-20260817-beb93ec2`、`3.0-20260805-ad3ba1ea-amd64` 这类
`major.minor-YYYYMMDD-<commit>` 的 daily build tag。它们就是下一个 3.0.x 的
release candidate。

### 4.2 `precheck.py`：daily build 放宽版本下限

`version_at_least` 之前判定 `is_daily_build_image(expected_server_image)`：
命中（family 已匹配、只是 patch 低）则跳过 `SERVER_VERSION_TOO_OLD`，并记录
`server_version_validation_mode=release_candidate_build` 与
`candidate_server_version`。

要点：

- 放宽**仅对 daily build tag 生效**；正式 release tag（如 `v3.0.0`）仍会被
  `SERVER_VERSION_TOO_OLD` 拒绝，不破坏「普通 gate 用 v3.0.0 冒充 v3.0.1」的
  既有保护。
- Vortex guard 无需改：`milvus-3-0-1` 声明 version=`3.0.1`，`_phase_supports_vortex`
  按 `version >= 3.0.1` 通过。

### 4.3 记录证据

precheck 结果里记录 `server_version_validation_mode=release_candidate_build` 与
`candidate_server_version`；最终报告仍按场景分类计算 `release_gate_eligible`，
candidate daily build 的运行证据不冒充正式 release gate。

## 5. 实施任务（已完成）

1. `version.py`：新增 `DAILY_BUILD_TAG`、`image_tag`、`is_daily_build_image`。
2. `precheck.py`：`version_at_least` 前判定 daily build，命中跳过
   `SERVER_VERSION_TOO_OLD`。
3. 单测：daily build tag 通过 precheck、release tag（`v3.0.0`）仍被拒绝。
4. 文档：本工作流。

## 6. 验证命令

```bash
cd milvus-bricks
PYTHONPATH=. uv run pytest -q milvus_client/tests/test_precheck.py milvus_client/tests/test_version.py
uvx ruff check milvus_client/common/version.py milvus_client/requests/precheck.py
uvx ruff format --check milvus_client/common/version.py milvus_client/requests/precheck.py
```
