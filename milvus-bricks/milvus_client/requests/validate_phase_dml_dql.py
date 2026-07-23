from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.client import create_client
from milvus_client.common.data import (
    generate_primary_key_value,
    generate_rows,
    stable_vector_value,
    vector_fields,
)
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import (
    FieldSpec,
    SchemaSpec,
    auto_id_enabled,
    build_index_params,
    build_milvus_schema,
    collection_name,
    create_collection_kwargs,
    function_output_fields,
    load_schema_matrix,
)
from milvus_client.common.validators import (
    ValidationReport,
    format_filter_value,
    pk_range_filter,
    validate_collection_count,
    validate_pk_samples,
)
from milvus_client.common.workload import (
    assert_search_result,
    metric_type_for_field,
    search_params_for_field,
)


PHASE_DML_FAILED = "PHASE_DML_FAILED"
PHASE_DQL_FAILED = "PHASE_DQL_FAILED"
PHASE_NEW_COLLECTION_FAILED = "PHASE_NEW_COLLECTION_FAILED"
PHASE_UPSERT_NOT_APPLIED = "PHASE_UPSERT_NOT_APPLIED"
CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--new-collection-prefix", required=True)
    parser.add_argument("--carried-collection-prefix", default="")
    parser.add_argument("--new-collection-rows", type=int, default=3000)
    parser.add_argument("--existing-dml-rows", type=int, default=1000)
    parser.add_argument("--existing-delete-rows", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--existing-start-id", type=int, default=50_000_000)
    parser.add_argument("--new-start-id", type=int, default=60_000_000)
    parser.add_argument(
        "--drop-new-collections-if-exist", type=parse_bool, default=True
    )


def _primary_field(spec: SchemaSpec) -> FieldSpec | None:
    primary = [field for field in spec.fields if field.primary]
    if primary:
        return primary[0]
    return None


def _spec_by_schema(schema_matrix: str) -> dict[str, SchemaSpec]:
    return {spec.name: spec for spec in load_schema_matrix(schema_matrix)}


def _checkpoint_path(args) -> Path:
    if args.checkpoint_file:
        return Path(args.checkpoint_file)
    return Path(args.checkpoint_dir) / "seed_data.json"


def _extract_insert_ids(response: Any) -> list[Any]:
    if response is None:
        return []
    if isinstance(response, dict):
        for key in ("ids", "primary_keys", "primaryKeys"):
            if key in response:
                return list(response[key])
    for attr in ("ids", "primary_keys", "primaryKeys"):
        if hasattr(response, attr):
            return list(getattr(response, attr))
    return []


def _partition_for_id(partitions: list[str], pk: int) -> str | None:
    if not partitions:
        return None
    return partitions[pk % len(partitions)]


def _insert_rows(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    rows: list[dict[str, Any]],
    start_id: int,
) -> list[Any]:
    responses = []
    if spec.partitions:
        partition_rows: dict[str, list[dict[str, Any]]] = {}
        for offset, row in enumerate(rows):
            partition = _partition_for_id(spec.partitions, start_id + offset)
            partition_rows.setdefault(partition or "", []).append(row)
        for partition, batch in partition_rows.items():
            responses.append(
                client.insert(
                    collection_name=target_collection,
                    data=batch,
                    partition_name=partition or None,
                )
            )
    else:
        responses.append(client.insert(collection_name=target_collection, data=rows))

    ids: list[Any] = []
    for response in responses:
        ids.extend(_extract_insert_ids(response))
    return ids


def _call_best_effort(method: Any, *args, **kwargs) -> str:
    if method is None:
        return "not_available"
    try:
        method(*args, **kwargs)
        return "done"
    except TypeError:
        try:
            method(*args)
            return "done"
        except Exception as exc:
            return f"failed: {exc}"
    except Exception as exc:
        return f"failed: {exc}"


def _flush_and_load_best_effort(client: Any, target_collection: str) -> dict[str, str]:
    return {
        "flush": _call_best_effort(getattr(client, "flush", None), target_collection),
        "load": _call_best_effort(
            getattr(client, "load_collection", None),
            target_collection,
        ),
    }


def _create_new_collection(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    drop_if_exists: bool,
) -> str:
    if client.has_collection(target_collection):
        if not drop_if_exists:
            raise RuntimeError(f"{target_collection} already exists")
        release = getattr(client, "release_collection", None)
        if release is not None:
            _call_best_effort(release, target_collection)
        client.drop_collection(target_collection)

    client.create_collection(
        collection_name=target_collection,
        schema=build_milvus_schema(spec),
        **create_collection_kwargs(spec),
    )
    for partition in spec.partitions:
        has_partition = False
        if hasattr(client, "has_partition"):
            has_partition = client.has_partition(
                collection_name=target_collection,
                partition_name=partition,
            )
        if not has_partition:
            client.create_partition(
                collection_name=target_collection,
                partition_name=partition,
            )
    if spec.indexes:
        client.create_index(
            collection_name=target_collection,
            index_params=build_index_params(spec),
        )
    _flush_and_load_best_effort(client, target_collection)
    return target_collection


def _delete_pk_values(
    client: Any,
    target_collection: str,
    primary_name: str,
    pk_values: list[Any],
) -> int:
    if not pk_values:
        return 0
    values = ", ".join(format_filter_value(value) for value in pk_values)
    client.delete(
        collection_name=target_collection,
        filter=f"{primary_name} in [{values}]",
    )
    return len(pk_values)


def _validate_deleted_pk_values(
    client: Any,
    target_collection: str,
    primary_name: str,
    pk_values: list[Any],
    report: ValidationReport,
) -> None:
    for pk in pk_values[:3]:
        try:
            rows = client.query(
                collection_name=target_collection,
                filter=f"{primary_name} == {format_filter_value(pk)}",
                output_fields=[primary_name],
                limit=1,
            )
        except Exception as exc:
            report.fail(
                PHASE_DQL_FAILED,
                "deleted primary key query failed",
                collection=target_collection,
                pk=pk,
                error=str(exc),
            )
            continue
        if rows:
            report.fail(
                PHASE_DQL_FAILED,
                "deleted primary key is still queryable",
                collection=target_collection,
                pk=pk,
            )


def _run_searches(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    seed: int,
    pk: int,
    report: ValidationReport,
) -> int:
    searches = 0
    function_outputs = function_output_fields(spec)
    for vector_field in vector_fields(spec):
        metric_type = metric_type_for_field(spec, vector_field.name)
        if vector_field.name in function_outputs and metric_type == "BM25":
            query_vector = f"milvus phase dml dql token_{pk % 16}"
        else:
            query_vector = stable_vector_value(vector_field, pk, seed)
        try:
            result = client.search(
                collection_name=target_collection,
                data=[query_vector],
                anns_field=vector_field.name,
                limit=5,
                search_params={
                    "metric_type": metric_type,
                    "params": search_params_for_field(spec, vector_field.name),
                },
            )
            assert_search_result(result, target_collection, vector_field.name)
            searches += 1
        except Exception as exc:
            report.fail(
                PHASE_DQL_FAILED,
                "phase vector search failed",
                collection=target_collection,
                field=vector_field.name,
                error=str(exc),
            )
    return searches


def _normalize_for_compare(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_compare(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_compare(item) for item in value]
    return value


def _upsert_validation_field(spec: SchemaSpec) -> FieldSpec | None:
    function_outputs = function_output_fields(spec)
    float_vectors = [
        field
        for field in vector_fields(spec)
        if field.name not in function_outputs and field.dtype == "FLOAT_VECTOR"
    ]
    if float_vectors:
        return float_vectors[0]
    non_function_vectors = [
        field for field in vector_fields(spec) if field.name not in function_outputs
    ]
    if non_function_vectors:
        return non_function_vectors[0]
    return None


def _validate_upserted_values(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    primary: FieldSpec,
    start_id: int,
    sample_offsets: list[int],
    seed: int,
    report: ValidationReport,
) -> None:
    validation_field = _upsert_validation_field(spec)
    if validation_field is None:
        return
    primary_name = primary.name
    sample_values = [
        generate_primary_key_value(primary, start_id + offset)
        for offset in sample_offsets
    ]
    values = ", ".join(format_filter_value(value) for value in sample_values)
    try:
        rows = client.query(
            collection_name=target_collection,
            filter=f"{primary_name} in [{values}]",
            output_fields=[primary_name, validation_field.name],
            limit=len(sample_values),
        )
    except Exception as exc:
        report.fail(
            PHASE_DQL_FAILED,
            "upserted value query failed",
            collection=target_collection,
            field=validation_field.name,
            error=str(exc),
        )
        return
    rows_by_pk = {row.get(primary_name): row for row in rows}
    for offset, pk_value in zip(sample_offsets, sample_values):
        row = rows_by_pk.get(pk_value)
        if not row:
            report.fail(
                PHASE_UPSERT_NOT_APPLIED,
                "upserted primary key is missing",
                collection=target_collection,
                pk=pk_value,
                field=validation_field.name,
            )
            continue
        actual = _normalize_for_compare(row.get(validation_field.name))
        expected_rows = generate_rows(
            spec, start_id=start_id + offset, count=1, seed=seed + 101
        )
        expected = _normalize_for_compare(expected_rows[0].get(validation_field.name))
        if actual != expected:
            report.fail(
                PHASE_UPSERT_NOT_APPLIED,
                "upserted field value does not match expected updated value",
                collection=target_collection,
                pk=pk_value,
                field=validation_field.name,
                expected=expected,
                actual=actual,
            )


def _run_existing_collection_dml_dql(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    rows: int,
    delete_rows: int,
    batch_size: int,
    start_id: int,
    seed: int,
    report: ValidationReport,
) -> dict[str, Any]:
    primary = _primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    metrics: dict[str, Any] = {
        "collection": target_collection,
        "inserted": 0,
        "upserted": 0,
        "deleted": 0,
        "searches": 0,
        "upsert_skipped_auto_id": False,
    }
    inserted_ids: list[Any] = []

    try:
        for start in range(start_id, start_id + rows, batch_size):
            count = min(batch_size, start_id + rows - start)
            batch = generate_rows(spec, start_id=start, count=count, seed=seed)
            ids = _insert_rows(client, spec, target_collection, batch, start)
            inserted_ids.extend(ids)
            metrics["inserted"] += len(batch)

        if auto_id_enabled(spec):
            metrics["upsert_skipped_auto_id"] = True
        else:
            for start in range(start_id, start_id + rows, batch_size):
                count = min(batch_size, start_id + rows - start)
                batch = generate_rows(
                    spec, start_id=start, count=count, seed=seed + 101
                )
                client.upsert(collection_name=target_collection, data=batch)
                metrics["upserted"] += len(batch)

        if auto_id_enabled(spec):
            deleted_values = inserted_ids[: min(delete_rows, len(inserted_ids))]
        else:
            deleted_values = [
                generate_primary_key_value(primary, start_id + offset)
                for offset in range(min(delete_rows, rows))
            ]
        metrics["deleted"] = _delete_pk_values(
            client,
            target_collection,
            primary_name,
            deleted_values,
        )
        _flush_and_load_best_effort(client, target_collection)
    except Exception as exc:
        report.fail(
            PHASE_DML_FAILED,
            "existing collection phase DML failed",
            collection=target_collection,
            schema=spec.name,
            error=str(exc),
        )
        return metrics

    if auto_id_enabled(spec):
        remaining_values = inserted_ids[metrics["deleted"] : metrics["deleted"] + 3]
        validate_pk_samples(
            client, target_collection, primary_name, remaining_values, report
        )
    else:
        remaining_start_id = start_id + metrics["deleted"]
        min_pk = generate_primary_key_value(primary, remaining_start_id)
        max_pk = generate_primary_key_value(primary, start_id + rows - 1)
        validate_collection_count(
            client,
            target_collection,
            rows - metrics["deleted"],
            report,
            filter_expr=pk_range_filter(primary_name, min_pk, max_pk),
            metric_suffix="phase_existing_dml_count",
        )
        sample_values = [
            generate_primary_key_value(primary, remaining_start_id),
            generate_primary_key_value(primary, start_id + rows - 1),
        ]
        validate_pk_samples(
            client, target_collection, primary_name, sample_values, report
        )
        _validate_upserted_values(
            client,
            spec,
            target_collection,
            primary,
            start_id,
            [metrics["deleted"], rows - 1],
            seed,
            report,
        )
    _validate_deleted_pk_values(
        client,
        target_collection,
        primary_name,
        deleted_values,
        report,
    )
    metrics["searches"] = _run_searches(
        client, spec, target_collection, seed, start_id + rows - 1, report
    )
    return metrics


def _run_new_collection_dml_dql(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    rows: int,
    batch_size: int,
    start_id: int,
    seed: int,
    drop_if_exists: bool,
    report: ValidationReport,
) -> dict[str, Any]:
    primary = _primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    metrics: dict[str, Any] = {
        "collection": target_collection,
        "inserted": 0,
        "searches": 0,
    }
    inserted_ids: list[Any] = []
    try:
        _create_new_collection(client, spec, target_collection, drop_if_exists)
        for start in range(start_id, start_id + rows, batch_size):
            count = min(batch_size, start_id + rows - start)
            batch = generate_rows(spec, start_id=start, count=count, seed=seed)
            ids = _insert_rows(client, spec, target_collection, batch, start)
            inserted_ids.extend(ids)
            metrics["inserted"] += len(batch)
        _flush_and_load_best_effort(client, target_collection)
    except Exception as exc:
        report.fail(
            PHASE_NEW_COLLECTION_FAILED,
            "new collection phase setup/DML failed",
            collection=target_collection,
            schema=spec.name,
            error=str(exc),
        )
        return metrics

    if auto_id_enabled(spec):
        validate_collection_count(
            client,
            target_collection,
            rows,
            report,
            metric_suffix="phase_new_collection_count",
        )
        validate_pk_samples(
            client, target_collection, primary_name, inserted_ids[:3], report
        )
    else:
        min_pk = generate_primary_key_value(primary, start_id)
        max_pk = generate_primary_key_value(primary, start_id + rows - 1)
        validate_collection_count(
            client,
            target_collection,
            rows,
            report,
            filter_expr=pk_range_filter(primary_name, min_pk, max_pk),
            metric_suffix="phase_new_collection_count",
        )
        validate_pk_samples(
            client,
            target_collection,
            primary_name,
            [
                generate_primary_key_value(primary, start_id),
                generate_primary_key_value(primary, start_id + rows - 1),
            ],
            report,
        )
    metrics["searches"] = _run_searches(
        client, spec, target_collection, seed, start_id + rows - 1, report
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser(
        "Validate phase DML/DQL against existing and new collections"
    )
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "validate_phase_dml_dql")

    try:
        checkpoint_file = _checkpoint_path(args)
        if not checkpoint_file.exists():
            result.status = FAILED
            result.mark_failed(
                CHECKPOINT_NOT_FOUND,
                "seed checkpoint file does not exist",
                path=str(checkpoint_file),
            )
            result.write(args.output_json)
            return 2

        checkpoint = json.loads(checkpoint_file.read_text())
        specs = _spec_by_schema(args.schema_matrix)
        client = create_client(args.uri, args.token, args.db_name)
        report = ValidationReport()
        metrics: dict[str, Any] = {
            "phase": args.phase,
            "existing_collections_total": 0,
            "new_collections_total": 0,
            "existing_inserted_total": 0,
            "existing_upserted_total": 0,
            "existing_deleted_total": 0,
            "existing_upsert_skipped_auto_id_total": 0,
            "carried_collections_total": 0,
            "carried_inserted_total": 0,
            "carried_upserted_total": 0,
            "carried_deleted_total": 0,
            "new_collection_inserted_total": 0,
            "searches_total": 0,
            "existing_collections": [],
            "carried_collections": [],
            "new_collections": [],
        }

        for existing_collection, meta in checkpoint.get("collections", {}).items():
            spec = specs.get(meta["schema_name"])
            if spec is None:
                report.fail(
                    PHASE_DML_FAILED,
                    "schema from checkpoint is not present in schema matrix",
                    collection=existing_collection,
                    schema=meta["schema_name"],
                )
                continue
            metrics["existing_collections_total"] += 1
            existing_metrics = _run_existing_collection_dml_dql(
                client,
                spec,
                existing_collection,
                args.existing_dml_rows,
                args.existing_delete_rows,
                args.batch_size,
                args.existing_start_id,
                args.seed,
                report,
            )
            metrics["existing_collections"].append(existing_metrics)
            metrics["existing_inserted_total"] += existing_metrics["inserted"]
            metrics["existing_upserted_total"] += existing_metrics["upserted"]
            metrics["existing_deleted_total"] += existing_metrics["deleted"]
            metrics["searches_total"] += existing_metrics["searches"]
            if existing_metrics["upsert_skipped_auto_id"]:
                metrics["existing_upsert_skipped_auto_id_total"] += 1

        if args.carried_collection_prefix:
            carried_start_id = args.existing_start_id + 10_000_000
            for spec in specs.values():
                carried_collection = collection_name(
                    args.carried_collection_prefix, spec
                )
                metrics["carried_collections_total"] += 1
                carried_metrics = _run_existing_collection_dml_dql(
                    client,
                    spec,
                    carried_collection,
                    args.existing_dml_rows,
                    args.existing_delete_rows,
                    args.batch_size,
                    carried_start_id,
                    args.seed + 31,
                    report,
                )
                metrics["carried_collections"].append(carried_metrics)
                metrics["carried_inserted_total"] += carried_metrics["inserted"]
                metrics["carried_upserted_total"] += carried_metrics["upserted"]
                metrics["carried_deleted_total"] += carried_metrics["deleted"]
                metrics["searches_total"] += carried_metrics["searches"]

        for spec in specs.values():
            new_collection = collection_name(args.new_collection_prefix, spec)
            metrics["new_collections_total"] += 1
            new_metrics = _run_new_collection_dml_dql(
                client,
                spec,
                new_collection,
                args.new_collection_rows,
                args.batch_size,
                args.new_start_id,
                args.seed + 17,
                args.drop_new_collections_if_exist,
                report,
            )
            metrics["new_collections"].append(new_metrics)
            metrics["new_collection_inserted_total"] += new_metrics["inserted"]
            metrics["searches_total"] += new_metrics["searches"]

        result.status = PASSED if report.passed else FAILED
        result.failures = report.failures
        result.metrics = {**report.metrics, **metrics}
        result.write(args.output_json)
        return 0 if report.passed else 1
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "PHASE_DML_DQL_VALIDATION_FAILED",
            "unexpected error during phase DML/DQL validation",
            error=str(exc),
        )
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
