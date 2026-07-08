from __future__ import annotations

from dataclasses import replace
import sys
from typing import Any

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.data import generate_primary_key_value, generate_rows, stable_vector_value, vector_fields
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import (
    FieldSpec,
    FunctionSpec,
    SchemaSpec,
    auto_id_enabled,
    collection_name,
    dtype_to_milvus,
    function_output_fields,
)
from milvus_client.common.validators import format_filter_value, query_count
from milvus_client.common.workload import metric_type_for_field, primary_field, search_params_for_field


EVOLUTION_FIELD = FieldSpec(
    name="evo_nullable_varchar",
    dtype="VARCHAR",
    nullable=True,
    max_length=256,
)
EVOLUTION_DROP_FIELD = FieldSpec(
    name="evo_drop_candidate",
    dtype="INT64",
    nullable=True,
)


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--rows-per-collection", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-id", type=int, default=40_000_000)


def _field_kwargs(field: FieldSpec) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"nullable": True}
    if field.max_length is not None:
        kwargs["max_length"] = field.max_length
    if field.dim is not None:
        kwargs["dim"] = field.dim
    if field.element_type is not None:
        from pymilvus import DataType

        kwargs["element_type"] = getattr(DataType, field.element_type)
    if field.max_capacity is not None:
        kwargs["max_capacity"] = field.max_capacity
    if field.dtype == "VARCHAR":
        kwargs["default_value"] = ""
    if field.dtype in {"INT64", "INT32", "INT16", "INT8"}:
        kwargs["default_value"] = 0
    if field.dtype in {"FLOAT", "DOUBLE"}:
        kwargs["default_value"] = 0.0
    if field.dtype == "BOOL":
        kwargs["default_value"] = False
    return kwargs


def _add_field(client: Any, collection: str, field: FieldSpec) -> str:
    if not hasattr(client, "add_collection_field"):
        return "skipped"
    try:
        client.add_collection_field(
            collection_name=collection,
            field_name=field.name,
            data_type=dtype_to_milvus(field.dtype),
            **_field_kwargs(field),
        )
        return "added"
    except Exception as exc:
        if "exist" in str(exc).lower() or "duplicate" in str(exc).lower():
            return "exists"
        raise


def _drop_field(client: Any, collection: str, field_name: str) -> str:
    drop = getattr(client, "drop_collection_field", None)
    if drop is None:
        return "skipped"
    drop(collection_name=collection, field_name=field_name)
    return "dropped"


def _function_cycle(client: Any, collection: str, function: FunctionSpec) -> str:
    if not hasattr(client, "drop_collection_function") or not hasattr(client, "add_collection_function"):
        return "skipped"
    from pymilvus import Function, FunctionType

    client.drop_collection_function(collection_name=collection, function_name=function.name)
    client.add_collection_function(
        collection_name=collection,
        function=Function(
            name=function.name,
            function_type=getattr(FunctionType, function.function_type),
            input_field_names=function.input_fields,
            output_field_names=function.output_fields,
            description=function.description,
            params=function.params,
        ),
    )
    return "cycled"


def _evolved_spec(spec: SchemaSpec) -> SchemaSpec:
    existing = {field.name for field in spec.fields}
    extra_fields = []
    if EVOLUTION_FIELD.name not in existing:
        extra_fields.append(EVOLUTION_FIELD)
    return replace(spec, fields=[*spec.fields, *extra_fields])


def _nullable_vector_update_rows(spec: SchemaSpec, rows: list[dict[str, Any]]) -> int:
    nullable_vectors = [field for field in vector_fields(spec) if field.nullable]
    if not nullable_vectors:
        return 0
    updated = 0
    for offset, row in enumerate(rows):
        for field in nullable_vectors:
            if offset % 2 == 0:
                row[field.name] = None
            else:
                pk_value = row.get(primary_field(spec).name if primary_field(spec) else "id", offset)
                row[field.name] = stable_vector_value(field, int(pk_value) if isinstance(pk_value, int) else offset, 17)
            updated += 1
    return updated


def _upsert_evolution_rows(
    client: Any,
    spec: SchemaSpec,
    collection: str,
    rows_per_collection: int,
    batch_size: int,
    start_id: int,
    seed: int,
) -> tuple[int, int]:
    if auto_id_enabled(spec):
        return (0, 0)
    evolved = _evolved_spec(spec)
    upserted = 0
    nullable_updates = 0
    for start in range(start_id, start_id + rows_per_collection, batch_size):
        count = min(batch_size, start_id + rows_per_collection - start)
        rows = generate_rows(evolved, start_id=start, count=count, seed=seed)
        for row in rows:
            row[EVOLUTION_FIELD.name] = f"evo_{row.get(primary_field(spec).name if primary_field(spec) else 'id')}"
        nullable_updates += _nullable_vector_update_rows(evolved, rows)
        client.upsert(collection_name=collection, data=rows)
        upserted += len(rows)
    return (upserted, nullable_updates)


def _read_validate(client: Any, spec: SchemaSpec, collection: str, start_id: int) -> tuple[int, int, int]:
    primary = primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    min_pk = generate_primary_key_value(primary, start_id) if primary is not None else start_id
    output_fields = [primary_name, EVOLUTION_FIELD.name]
    client.query(
        collection_name=collection,
        filter=f"{primary_name} >= {format_filter_value(min_pk)}",
        output_fields=output_fields,
        limit=10,
    )
    count = query_count(client, collection)
    searches = 0
    function_outputs = function_output_fields(spec)
    for vector_field in vector_fields(spec):
        if vector_field.nullable:
            continue
        metric_type = metric_type_for_field(spec, vector_field.name)
        if vector_field.name in function_outputs and metric_type == "BM25":
            query_vector = f"milvus schema evolution token_{start_id % 16}"
        else:
            query_vector = stable_vector_value(vector_field, start_id + 1, 23)
        client.search(
            collection_name=collection,
            data=[query_vector],
            anns_field=vector_field.name,
            limit=5,
            search_params={"metric_type": metric_type, "params": search_params_for_field(spec, vector_field.name)},
        )
        searches += 1
    return (1, count, searches)


def run_schema_evolution(
    client: Any,
    specs: list[SchemaSpec],
    collection_prefix: str,
    rows_per_collection: int,
    batch_size: int,
    start_id: int,
    seed: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "collections_total": len(specs),
        "field_add_total": 0,
        "field_add_exists_total": 0,
        "field_add_skipped_total": 0,
        "drop_field_skipped_total": 0,
        "function_cycles_total": 0,
        "function_cycle_skipped_total": 0,
        "upserted_total": 0,
        "nullable_updates_total": 0,
        "queries_total": 0,
        "searches_total": 0,
        "count_checks_total": 0,
        "failed_total": 0,
        "collections": [],
    }
    for spec in specs:
        collection = collection_name(collection_prefix, spec)
        collection_metrics: dict[str, Any] = {"schema": spec.name, "collection": collection}
        try:
            if not client.has_collection(collection):
                raise RuntimeError(f"{collection} does not exist")
            add_status = _add_field(client, collection, EVOLUTION_FIELD)
            collection_metrics["add_field"] = add_status
            metrics["field_add_total"] += 1
            metrics[f"field_add_{add_status}_total"] = metrics.get(f"field_add_{add_status}_total", 0) + 1

            drop_add_status = _add_field(client, collection, EVOLUTION_DROP_FIELD)
            drop_status = _drop_field(client, collection, EVOLUTION_DROP_FIELD.name)
            collection_metrics["drop_field"] = drop_status if drop_add_status != "skipped" else "skipped"
            if collection_metrics["drop_field"] == "skipped":
                metrics["drop_field_skipped_total"] += 1

            cycled = 0
            skipped = 0
            for function in spec.functions:
                status = _function_cycle(client, collection, function)
                if status == "cycled":
                    cycled += 1
                else:
                    skipped += 1
            metrics["function_cycles_total"] += cycled
            metrics["function_cycle_skipped_total"] += skipped
            collection_metrics["function_cycles"] = cycled
            collection_metrics["function_cycle_skipped"] = skipped

            upserted, nullable_updates = _upsert_evolution_rows(
                client, spec, collection, rows_per_collection, batch_size, start_id, seed
            )
            collection_metrics["upserted"] = upserted
            collection_metrics["nullable_updates"] = nullable_updates
            metrics["upserted_total"] += upserted
            metrics["nullable_updates_total"] += nullable_updates

            queries, count, searches = _read_validate(client, spec, collection, start_id)
            collection_metrics["query_checks"] = queries
            collection_metrics["count"] = count
            collection_metrics["searches"] = searches
            metrics["queries_total"] += queries
            metrics["count_checks_total"] += 1
            metrics["searches_total"] += searches
        except Exception as exc:
            metrics["failed_total"] += 1
            collection_metrics["error"] = str(exc)
        metrics["collections"].append(collection_metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Run schema evolution workload against existing Milvus collections")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "schema_evolution_workload")
    try:
        from milvus_client.common.schema import load_schema_matrix

        client = create_client(args.uri, args.token, args.db_name)
        metrics = run_schema_evolution(
            client,
            load_schema_matrix(args.schema_matrix),
            args.collection_prefix,
            args.rows_per_collection,
            args.batch_size,
            args.start_id,
            args.seed,
        )
        result.metrics = metrics
        result.status = FAILED if metrics["failed_total"] else PASSED
        for collection in metrics["collections"]:
            if "error" in collection:
                result.mark_failed(
                    "SCHEMA_EVOLUTION_FAILED",
                    "schema evolution workload failed",
                    collection=collection["collection"],
                    schema=collection["schema"],
                    error=collection["error"],
                )
        result.write(args.output_json)
        return 1 if result.status == FAILED else 0
    except Exception as exc:
        result.status = FAILED
        result.mark_failed("SCHEMA_EVOLUTION_FAILED", "unexpected schema evolution failure", error=str(exc))
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
