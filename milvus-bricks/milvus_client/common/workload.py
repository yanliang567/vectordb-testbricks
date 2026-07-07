from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from random import Random
import time
from typing import Any

from milvus_client.common.data import generate_primary_key_value, generate_rows, stable_vector_value, vector_fields
from milvus_client.common.schema import SchemaSpec, collection_name, load_schema_matrix
from milvus_client.common.validators import format_filter_value


PRESSURE_INSERT_BASE = 10_000_000
PRESSURE_UPSERT_BASE = 20_000_000
PRESSURE_DELETE_BASE = 30_000_000


@dataclass
class WorkloadSummary:
    counts: dict[str, int] = field(default_factory=dict)
    operations_total: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def failed(self) -> bool:
        return any(key.startswith("failed_") for key in self.counts)

    def record(self, operation: str, count: int) -> None:
        self.counts[operation] = self.counts.get(operation, 0) + count
        self.operations_total += 1

    def metrics(self) -> dict[str, Any]:
        finished_at = self.finished_at if self.finished_at is not None else time.time()
        metrics: dict[str, Any] = dict(self.counts)
        metrics["operations_total"] = self.operations_total
        metrics["duration_sec"] = max(0.0, finished_at - self.started_at)
        metrics["requests_failed"] = sum(count for key, count in self.counts.items() if key.startswith("failed_"))
        return metrics


def primary_field(spec: SchemaSpec):
    for field in spec.fields:
        if field.primary:
            return field
    return None


def metric_type_for_field(spec: SchemaSpec, field_name: str) -> str:
    for index in spec.indexes:
        if index.field == field_name and index.metric_type:
            return index.metric_type
    return "COSINE"


def assert_search_result(result: Any, collection: str, field_name: str) -> None:
    if not isinstance(result, list) or len(result) != 1:
        raise AssertionError(f"{collection}.{field_name}: unexpected search result shape")
    if not result[0]:
        raise AssertionError(f"{collection}.{field_name}: search returned no hits")


def _query_operation(client: Any, spec: SchemaSpec, collection: str) -> tuple[str, int]:
    primary = primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    min_pk = generate_primary_key_value(primary, 0) if primary is not None else 0
    client.query(
        collection_name=collection,
        filter=f"{primary_name} >= {format_filter_value(min_pk)}",
        output_fields=[primary_name],
        limit=10,
    )
    return ("query", 1)


def _search_operation(client: Any, spec: SchemaSpec, collection: str, seed: int, op_index: int) -> tuple[str, int]:
    fields = vector_fields(spec)
    if not fields:
        return ("search_skipped", 0)
    searches = 0
    for vector_field in fields:
        query_vector = stable_vector_value(vector_field, op_index + 1, seed)
        result = client.search(
            collection_name=collection,
            data=[query_vector],
            anns_field=vector_field.name,
            limit=5,
            search_params={"metric_type": metric_type_for_field(spec, vector_field.name), "params": {}},
        )
        assert_search_result(result, collection, vector_field.name)
        searches += 1
    return ("search", searches)


def _delete_operation(client: Any, spec: SchemaSpec, collection: str, batch_size: int, op_index: int) -> tuple[str, int]:
    primary = primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    start_id = PRESSURE_DELETE_BASE + op_index * batch_size
    min_pk = generate_primary_key_value(primary, start_id) if primary is not None else start_id
    max_pk = generate_primary_key_value(primary, start_id + batch_size - 1) if primary is not None else start_id + batch_size - 1
    client.delete(
        collection_name=collection,
        filter=f"{primary_name} >= {format_filter_value(min_pk)} && {primary_name} <= {format_filter_value(max_pk)}",
    )
    return ("delete", batch_size)


def _query_iterator_operation(client: Any, spec: SchemaSpec, collection: str, batch_size: int) -> tuple[str, int]:
    primary = primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    min_pk = generate_primary_key_value(primary, 0) if primary is not None else 0
    filter_expr = f"{primary_name} >= {format_filter_value(min_pk)}"
    if hasattr(client, "query_iterator"):
        iterator = client.query_iterator(
            collection_name=collection,
            filter=filter_expr,
            output_fields=[primary_name],
            batch_size=batch_size,
        )
        rows = 0
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows += len(batch)
                if rows >= batch_size:
                    break
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        return ("query_iterator", rows)
    rows = client.query(collection_name=collection, filter=filter_expr, output_fields=[primary_name], limit=batch_size)
    return ("query_iterator", len(rows))


def run_operation(client: Any, spec: SchemaSpec, collection: str, operation: str, seed: int, batch_size: int, op_index: int) -> tuple[str, int]:
    try:
        if operation == "insert":
            start_id = PRESSURE_INSERT_BASE + op_index * batch_size
            rows = generate_rows(spec, start_id=start_id, count=batch_size, seed=seed)
            client.insert(collection_name=collection, data=rows)
            return ("insert", len(rows))
        if operation == "upsert":
            start_id = PRESSURE_UPSERT_BASE + op_index * batch_size
            rows = generate_rows(spec, start_id=start_id, count=batch_size, seed=seed)
            client.upsert(collection_name=collection, data=rows)
            return ("upsert", len(rows))
        if operation == "delete":
            return _delete_operation(client, spec, collection, batch_size, op_index)
        if operation == "query":
            return _query_operation(client, spec, collection)
        if operation == "query_iterator":
            return _query_iterator_operation(client, spec, collection, batch_size)
        if operation == "search":
            return _search_operation(client, spec, collection, seed, op_index)
        raise ValueError(f"unknown operation: {operation}")
    except Exception:
        return (f"failed_{operation}", 1)


def run_pressure_workload(
    client: Any,
    schema_matrix: str,
    collection_prefix: str,
    operations: list[str],
    seed: int,
    duration_sec: int,
    max_workers: int,
    batch_size: int,
    operation_interval_sec: float = 0.0,
) -> WorkloadSummary:
    specs = load_schema_matrix(schema_matrix)
    rng = Random(seed)
    deadline = time.time() + duration_sec if duration_sec > 0 else time.time()
    op_index = 0
    summary = WorkloadSummary()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = set()

        def collect(done) -> None:
            for future in done:
                operation, count = future.result()
                summary.record(operation, count)

        while time.time() <= deadline or (duration_sec == 0 and not futures):
            spec = rng.choice(specs)
            collection = collection_name(collection_prefix, spec)
            operation = rng.choice(operations)
            futures.add(pool.submit(run_operation, client, spec, collection, operation, seed, batch_size, op_index))
            op_index += 1
            if operation_interval_sec > 0:
                time.sleep(operation_interval_sec)
            if len(futures) >= max_workers:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                collect(done)
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            collect(done)
    summary.finished_at = time.time()
    return summary
