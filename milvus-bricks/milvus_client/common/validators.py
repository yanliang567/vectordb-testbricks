from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from milvus_client.common.data import stable_checksum


COUNT_DRIFT = "COUNT_DRIFT"
MISSING_PK = "MISSING_PK"
QUERY_FAILED = "QUERY_FAILED"
SEARCH_FAILED = "SEARCH_FAILED"
CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"


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


def pk_range_filter(primary_field: str, min_pk: int, max_pk: int) -> str:
    return f"{primary_field} >= {min_pk} && {primary_field} <= {max_pk}"


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
    sample_pks: list[int],
    report: ValidationReport,
) -> None:
    for pk in sample_pks:
        try:
            rows = client.query(
                collection_name=collection_name,
                filter=f"{primary_field} == {pk}",
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
) -> list[dict[str, Any]]:
    rows = []
    offset = 0
    expr = pk_range_filter(primary_field, min_pk, max_pk)
    while True:
        kwargs = {
            "collection_name": collection_name,
            "filter": expr,
            "output_fields": output_fields,
            "limit": batch_size,
        }
        if offset:
            kwargs["offset"] = offset
        batch = client.query(**kwargs)
        rows.extend(batch)
        if len(batch) < batch_size:
            break
        offset += len(batch)
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
) -> None:
    output_fields = list(dict.fromkeys([primary_field, *checksum_fields]))
    try:
        rows = query_rows_by_pk_range(
            client,
            collection_name,
            primary_field,
            min_pk,
            max_pk,
            output_fields,
            batch_size,
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
