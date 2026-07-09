from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from milvus_client.common.data import stable_checksum


COUNT_DRIFT = "COUNT_DRIFT"
MISSING_PK = "MISSING_PK"
QUERY_FAILED = "QUERY_FAILED"
SEARCH_FAILED = "SEARCH_FAILED"
CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
SERVICEABILITY_TIMEOUT = "SERVICEABILITY_TIMEOUT"

TRANSIENT_SERVICEABILITY_PATTERNS = (
    "channel not available",
    "channel distribution is not serviceable",
    "no available shard leaders",
)


@dataclass
class ValidationReport:
    passed: bool = True
    failures: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def fail(self, failure_type: str, message: str, **details: Any) -> None:
        self.passed = False
        failure = {"type": failure_type, "message": message}
        failure.update(details)
        self.failures.append(failure)


def format_filter_value(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


def is_transient_serviceability_error(error: str) -> bool:
    normalized = error.lower()
    return any(pattern in normalized for pattern in TRANSIENT_SERVICEABILITY_PATTERNS)


def is_transient_serviceability_failure(failure: dict[str, Any]) -> bool:
    return (
        failure.get("type") == QUERY_FAILED
        and is_transient_serviceability_error(str(failure.get("error", "")))
    )


def pk_range_filter(primary_field: str, min_pk: Any, max_pk: Any) -> str:
    return f"{primary_field} >= {format_filter_value(min_pk)} && {primary_field} <= {format_filter_value(max_pk)}"


def query_count(client: Any, collection_name: str, filter_expr: str = "") -> int:
    result = client.query(collection_name=collection_name, filter=filter_expr, output_fields=["count(*)"])
    if not result:
        return 0
    return int(result[0].get("count(*)", 0))


def validate_collection_count(
    client: Any,
    collection_name: str,
    expected_count: int,
    report: ValidationReport,
    filter_expr: str = "",
    metric_suffix: str = "count",
) -> None:
    try:
        actual_count = query_count(client, collection_name, filter_expr=filter_expr)
    except Exception as exc:
        report.fail(QUERY_FAILED, "count query failed", collection=collection_name, error=str(exc))
        return
    report.metrics[f"{collection_name}.{metric_suffix}"] = actual_count
    if actual_count != expected_count:
        report.fail(
            COUNT_DRIFT,
            "checkpoint row count differs from expected range",
            collection=collection_name,
            expected=expected_count,
            actual=actual_count,
            filter=filter_expr,
        )


def validate_pk_samples(
    client: Any,
    collection_name: str,
    primary_field: str,
    sample_pks: list[Any],
    report: ValidationReport,
) -> None:
    for pk in sample_pks:
        try:
            rows = client.query(
                collection_name=collection_name,
                filter=f"{primary_field} == {format_filter_value(pk)}",
                output_fields=[primary_field],
                limit=1,
            )
        except Exception as exc:
            report.fail(QUERY_FAILED, "pk query failed", collection=collection_name, pk=pk, error=str(exc))
            continue
        if not rows:
            report.fail(MISSING_PK, "expected primary key is missing", collection=collection_name, pk=pk)


def query_rows_by_pk_range(
    client: Any,
    collection_name: str,
    primary_field: str,
    min_pk: int,
    max_pk: int,
    output_fields: list[str],
    batch_size: int,
    pk_value_fn: Callable[[int], Any] | None = None,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pk_value_fn is None:
        pk_value_fn = lambda pk: pk
    rows = []
    start_pk = min_pk
    while start_pk <= max_pk:
        end_pk = min(start_pk + batch_size - 1, max_pk)
        batch = client.query(
            collection_name=collection_name,
            filter=pk_range_filter(primary_field, pk_value_fn(start_pk), pk_value_fn(end_pk)),
            output_fields=output_fields,
            limit=batch_size,
        )
        rows.extend(batch)
        start_pk = end_pk + 1
    return rows


def query_rows_by_pk_values(
    client: Any,
    collection_name: str,
    primary_field: str,
    pk_values: list[Any],
    output_fields: list[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = []
    for offset in range(0, len(pk_values), batch_size):
        batch_values = pk_values[offset : offset + batch_size]
        values = ", ".join(format_filter_value(value) for value in batch_values)
        batch = client.query(
            collection_name=collection_name,
            filter=f"{primary_field} in [{values}]",
            output_fields=output_fields,
            limit=len(batch_values),
        )
        rows.extend(batch)
    return rows


def validate_scalar_checksum(
    client: Any,
    collection_name: str,
    primary_field: str,
    min_pk: int,
    max_pk: int,
    expected_checksum: str,
    checksum_fields: list[str],
    report: ValidationReport,
    batch_size: int = 1000,
    pk_value_fn: Callable[[int], Any] | None = None,
    pk_values: list[Any] | None = None,
) -> None:
    output_fields = list(dict.fromkeys([primary_field, *checksum_fields]))
    try:
        if pk_values is not None:
            rows = query_rows_by_pk_values(
                client,
                collection_name,
                primary_field,
                pk_values,
                output_fields,
                batch_size,
            )
        else:
            rows = query_rows_by_pk_range(
                client,
                collection_name,
                primary_field,
                min_pk,
                max_pk,
                output_fields,
                batch_size,
                pk_value_fn=pk_value_fn,
            )
    except Exception as exc:
        report.fail(QUERY_FAILED, "checksum query failed", collection=collection_name, error=str(exc))
        return

    actual_checksum = stable_checksum(rows, fields=checksum_fields, primary_field=primary_field)
    report.metrics[f"{collection_name}.checksum_rows"] = len(rows)
    report.metrics[f"{collection_name}.checksum"] = actual_checksum
    if actual_checksum != expected_checksum:
        report.fail(
            CHECKSUM_MISMATCH,
            "checkpoint checksum differs from queried scalar fields",
            collection=collection_name,
            expected=expected_checksum,
            actual=actual_checksum,
            fields=checksum_fields,
        )
