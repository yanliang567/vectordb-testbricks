# Rollback Serviceability Gate

## Context

Recent 2.6/3.0 upgrade rollback runs showed rollback validation can fail with transient query serviceability errors after the Milvus CR becomes ready:

- `channel distribution is not serviceable: channel not available`
- `no available shard leaders: channel not available`

Manual revalidation with the same checkpoint later passed, which means the workflow needs a smart rollback data serviceability gate before strict checksum validation. The gate must not hide data loss: count drift, missing PK, checksum mismatch, or timeout still fail the workflow.

## Implementation

- Add `milvus_client.requests.wait_data_serviceability`.
- Check each checkpoint collection with lightweight count and PK sample queries.
- Retry only recognized transient serviceability query failures.
- Fail fast on non-transient validation failures.
- Record recovery metrics in the brick result:
  - `recovered`
  - `recovery_duration_sec`
  - `attempts`
  - `transient_failure_attempts`
  - `timeout_sec`
  - `interval_sec`
- Add rollback serviceability gates to both standalone 2.6 and standalone 3.0 Argo templates:
  - `wait-rollback-serviceability` before `validate-after-rollback`
  - `wait-forward-rollback-serviceability` before `validate-forward-after-rollback` when forward rollback validation is enabled
- Keep pressure daemon running through the wait and validation gates.
- Make final report require rollback serviceability result files when rollback is enabled.

## Defaults

- `rollback-serviceability-timeout-sec`: `900`
- `rollback-serviceability-interval-sec`: `10`

## Reporting

`generate_workflow_report` now includes a `serviceability` section and Markdown `Serviceability Recovery` section. Recoverable scenarios are reported as passed with recovery duration; unrecovered scenarios fail with `SERVICEABILITY_TIMEOUT`.
