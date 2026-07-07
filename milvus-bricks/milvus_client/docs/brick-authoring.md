# Brick Authoring Guide

New MilvusClient bricks should be small request units that can run locally,
inside Argo, or under a higher-level scenario controller.

## Required CLI

Use `milvus_client.common.args.build_common_parser()` and keep these common
arguments available:

- `--uri`
- `--token`
- `--db-name`
- `--collection-prefix`
- `--duration-sec`
- `--seed`
- `--feature-set`
- `--compat-mode`
- `--capability-probe`
- `--skip-unsupported`
- `--lifecycle-phase`
- `--checkpoint-dir`
- `--output-json`
- `--log-level`

## Result JSON

Use `milvus_client.common.result.BrickResult` or `result_from_args()`.

Every result must include:

- `brick`
- `status`
- `feature_set`
- `compat_mode`
- `lifecycle_phase`
- `target`
- `metrics`
- `failures`
- `capabilities`
- `skip_reason`
- `checkpoint`

## Exit Codes

- `0`: request completed successfully, including clean `skipped` results.
- `1`: test assertion or data validation failure.
- `2`: invalid arguments, invalid manifest, or unsupported direct mode.
- `3`: environment unavailable.
- `4`: unexpected internal error.

## Compatibility Rules

Use `rollback_safe` only for schemas and operations that must survive
2.6 -> 3.0 -> 2.6 validation. Mark 3.0-only features as `forward_only` unless
rollback behavior has been explicitly proven.

