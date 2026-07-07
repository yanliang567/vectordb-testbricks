import re

from milvus_client.common.data import stable_checksum
from milvus_client.common.validators import (
    CHECKSUM_MISMATCH,
    COUNT_DRIFT,
    ValidationReport,
    pk_range_filter,
    query_rows_by_pk_range,
    validate_collection_count,
    validate_pk_samples,
    validate_scalar_checksum,
)


class FakeMilvusClient:
    def __init__(self, rows):
        self.rows = rows
        self.query_calls = []

    def query(self, collection_name, filter="", output_fields=None, limit=None, offset=0):
        del collection_name
        self.query_calls.append({"filter": filter, "limit": limit, "offset": offset})
        matched = self._match_filter(filter)
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(matched)}]
        sliced = matched[offset : offset + limit if limit is not None else None]
        if output_fields is None:
            return sliced
        return [{field: row.get(field) for field in output_fields} for row in sliced]

    def _match_filter(self, filter_expr):
        if not filter_expr:
            return list(self.rows)
        equals = re.fullmatch(r"id == (\d+)", filter_expr)
        if equals:
            pk = int(equals.group(1))
            return [row for row in self.rows if row["id"] == pk]
        string_equals = re.fullmatch(r'pk == "([^"]+)"', filter_expr)
        if string_equals:
            pk = string_equals.group(1)
            return [row for row in self.rows if row["pk"] == pk]
        between = re.fullmatch(r"id >= (\d+) && id <= (\d+)", filter_expr)
        if between:
            min_pk = int(between.group(1))
            max_pk = int(between.group(2))
            return [row for row in self.rows if min_pk <= row["id"] <= max_pk]
        raise AssertionError(f"unexpected filter: {filter_expr}")


def test_checkpoint_count_ignores_pressure_rows_outside_pk_range():
    client = FakeMilvusClient(
        [
            {"id": 0, "category": 0},
            {"id": 1, "category": 1},
            {"id": 2, "category": 2},
            {"id": 10_000_000, "category": 99},
        ]
    )
    report = ValidationReport()

    validate_collection_count(
        client,
        "qa_dense",
        3,
        report,
        filter_expr=pk_range_filter("id", 0, 2),
        metric_suffix="checkpoint_count",
    )

    assert report.passed
    assert report.metrics["qa_dense.checkpoint_count"] == 3


def test_checkpoint_count_reports_baseline_drift():
    client = FakeMilvusClient([{"id": 0, "category": 0}, {"id": 10_000_000, "category": 99}])
    report = ValidationReport()

    validate_collection_count(
        client,
        "qa_dense",
        3,
        report,
        filter_expr=pk_range_filter("id", 0, 2),
        metric_suffix="checkpoint_count",
    )

    assert not report.passed
    assert report.failures[0]["type"] == COUNT_DRIFT
    assert report.failures[0]["actual"] == 1


def test_scalar_checksum_queries_checkpoint_rows():
    rows = [
        {"id": 2, "category": 2, "embedding": [0.2]},
        {"id": 0, "category": 0, "embedding": [0.0]},
        {"id": 1, "category": 1, "embedding": [0.1]},
        {"id": 10_000_000, "category": 99, "embedding": [9.9]},
    ]
    checksum = stable_checksum(rows[:3], fields=["id", "category"], primary_field="id")
    client = FakeMilvusClient(rows)
    report = ValidationReport()

    validate_scalar_checksum(
        client,
        "qa_dense",
        "id",
        0,
        2,
        checksum,
        ["id", "category"],
        report,
        batch_size=2,
    )

    assert report.passed
    assert report.metrics["qa_dense.checksum_rows"] == 3
    assert [call["offset"] for call in client.query_calls] == [0, 0]
    assert [call["filter"] for call in client.query_calls] == ["id >= 0 && id <= 1", "id >= 2 && id <= 2"]


def test_scalar_checksum_reports_mismatch():
    client = FakeMilvusClient([{"id": 0, "category": 10}])
    report = ValidationReport()

    validate_scalar_checksum(
        client,
        "qa_dense",
        "id",
        0,
        0,
        "bad-checksum",
        ["id", "category"],
        report,
    )

    assert not report.passed
    assert report.failures[0]["type"] == CHECKSUM_MISMATCH


def test_validate_pk_samples_quotes_string_primary_keys():
    client = FakeMilvusClient([{"pk": "pk_00000000000000000007"}])
    report = ValidationReport()

    validate_pk_samples(client, "qa_string", "pk", ["pk_00000000000000000007"], report)

    assert report.passed
    assert client.query_calls[0]["filter"] == 'pk == "pk_00000000000000000007"'


def test_query_rows_by_pk_range_formats_generated_string_primary_keys():
    class CapturingClient:
        def __init__(self):
            self.query_calls = []

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return [{"pk": "pk_00000000000000000007"}]

    client = CapturingClient()

    rows = query_rows_by_pk_range(
        client,
        "qa_string",
        "pk",
        7,
        8,
        ["pk"],
        batch_size=2,
        pk_value_fn=lambda pk: f"pk_{pk:020d}",
    )

    assert rows == [{"pk": "pk_00000000000000000007"}]
    assert client.query_calls[0]["filter"] == 'pk >= "pk_00000000000000000007" && pk <= "pk_00000000000000000008"'
