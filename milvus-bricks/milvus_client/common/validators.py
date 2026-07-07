from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


def query_count(client: Any, collection_name: str) -> int:
    result = client.query(collection_name=collection_name, filter="", output_fields=["count(*)"])
    if not result:
        return 0
    return int(result[0].get("count(*)", 0))


def validate_collection_count(
    client: Any,
    collection_name: str,
    expected_count: int,
    report: ValidationReport,
) -> None:
    try:
        actual_count = query_count(client, collection_name)
    except Exception as exc:
        report.fail(QUERY_FAILED, "count query failed", collection=collection_name, error=str(exc))
        return
    report.metrics[f"{collection_name}.count"] = actual_count
    if actual_count != expected_count:
        report.fail(
            COUNT_DRIFT,
            "collection count differs from checkpoint",
            collection=collection_name,
            expected=expected_count,
            actual=actual_count,
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

