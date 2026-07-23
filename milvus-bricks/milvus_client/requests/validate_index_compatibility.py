from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.client import create_client
from milvus_client.common.data import generate_primary_key_value, stable_vector_value
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import (
    FieldSpec,
    SchemaSpec,
    VECTOR_TYPES,
    build_index_params,
    function_output_fields,
    load_schema_matrix,
)
from milvus_client.common.validators import (
    ValidationReport,
    pk_range_filter,
    validate_collection_count,
    validate_pk_samples,
)
from milvus_client.common.workload import (
    assert_search_result,
    metric_type_for_field,
    search_params_for_field,
)


INDEX_SEARCH_FAILED = "INDEX_SEARCH_FAILED"
INDEX_REBUILD_FAILED = "INDEX_REBUILD_FAILED"
INDEX_COMPATIBILITY_CHECKPOINT_NOT_FOUND = "INDEX_COMPATIBILITY_CHECKPOINT_NOT_FOUND"
INDEX_COMPATIBILITY_CHECKPOINT_EMPTY = "INDEX_COMPATIBILITY_CHECKPOINT_EMPTY"


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--index-checkpoint-file", default="")
    parser.add_argument(
        "--phase", choices=["after-upgrade", "after-rollback"], required=True
    )
    parser.add_argument("--rebuild-index", type=parse_bool, default=False)
    parser.add_argument("--timeout-sec", type=int, default=900)


def _spec_by_schema(schema_matrix: str) -> dict[str, SchemaSpec]:
    return {spec.name: spec for spec in load_schema_matrix(schema_matrix)}


def _primary_field(spec: SchemaSpec) -> FieldSpec | None:
    primary = [field for field in spec.fields if field.primary]
    if primary:
        return primary[0]
    return None


def _field_by_name(spec: SchemaSpec) -> dict[str, FieldSpec]:
    return {field.name: field for field in spec.fields}


def _indexed_fields(spec: SchemaSpec) -> list[str]:
    return list(dict.fromkeys(index.field for index in spec.indexes))


def _indexed_vector_fields(spec: SchemaSpec) -> list[FieldSpec]:
    fields = _field_by_name(spec)
    return [
        fields[field_name]
        for field_name in _indexed_fields(spec)
        if field_name in fields and fields[field_name].dtype in VECTOR_TYPES
    ]


def _call_with_optional_timeout(method, *args, timeout_sec: int, **kwargs):
    try:
        return method(*args, timeout=timeout_sec, **kwargs)
    except TypeError:
        return method(*args, **kwargs)


def _flush_collection(client: Any, collection: str, timeout_sec: int) -> None:
    try:
        _call_with_optional_timeout(
            client.flush,
            collection_name=collection,
            timeout_sec=timeout_sec,
        )
    except TypeError:
        _call_with_optional_timeout(client.flush, collection, timeout_sec=timeout_sec)


def _release_collection_best_effort(
    client: Any, collection: str, timeout_sec: int
) -> str:
    release = getattr(client, "release_collection", None)
    if release is None:
        return "release_collection_not_available"
    try:
        _call_with_optional_timeout(
            release,
            collection_name=collection,
            timeout_sec=timeout_sec,
        )
        return "released"
    except TypeError:
        try:
            _call_with_optional_timeout(release, collection, timeout_sec=timeout_sec)
            return "released"
        except Exception as exc:
            return f"release_failed: {exc}"
    except Exception as exc:
        return f"release_failed: {exc}"


def _load_collection(client: Any, collection: str, timeout_sec: int) -> None:
    try:
        _call_with_optional_timeout(
            client.load_collection,
            collection_name=collection,
            timeout_sec=timeout_sec,
        )
    except TypeError:
        _call_with_optional_timeout(
            client.load_collection, collection, timeout_sec=timeout_sec
        )


def _index_names_for_field(client: Any, collection: str, field_name: str) -> list[str]:
    list_indexes = getattr(client, "list_indexes", None)
    if list_indexes is None:
        return [field_name]
    try:
        names = list_indexes(collection_name=collection, field_name=field_name)
    except TypeError:
        names = list_indexes(collection, field_name)
    except Exception:
        return [field_name]
    return list(names or [field_name])


def _drop_indexes_for_spec(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    report: ValidationReport,
    timeout_sec: int,
) -> int:
    dropped = 0
    for field_name in _indexed_fields(spec):
        for index_name in _index_names_for_field(client, collection, field_name):
            try:
                _call_with_optional_timeout(
                    client.drop_index,
                    collection_name=collection,
                    index_name=index_name,
                    timeout_sec=timeout_sec,
                )
                dropped += 1
            except TypeError:
                _call_with_optional_timeout(
                    client.drop_index,
                    collection,
                    index_name,
                    timeout_sec=timeout_sec,
                )
                dropped += 1
            except Exception as exc:
                message = str(exc).lower()
                if "not exist" in message or "not found" in message:
                    continue
                report.fail(
                    INDEX_REBUILD_FAILED,
                    "failed to drop existing index before compatibility rebuild",
                    collection=collection,
                    field=field_name,
                    index=index_name,
                    error=str(exc),
                )
    return dropped


def _create_indexes_for_spec(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    timeout_sec: int,
) -> None:
    index_params = build_index_params(spec)
    _call_with_optional_timeout(
        client.create_index,
        collection_name=collection,
        index_params=index_params,
        timeout_sec=timeout_sec,
    )


def _validate_index_searches(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    seed: int,
    report: ValidationReport,
) -> int:
    searches = 0
    function_outputs = function_output_fields(spec)
    for vector_field in _indexed_vector_fields(spec):
        metric_type = metric_type_for_field(spec, vector_field.name)
        if vector_field.name in function_outputs and metric_type == "BM25":
            query_vector = "milvus compatibility upgrade rollback token_1"
        else:
            query_vector = stable_vector_value(vector_field, 1, seed)
        try:
            response = client.search(
                collection_name=collection,
                data=[query_vector],
                anns_field=vector_field.name,
                limit=5,
                search_params={
                    "metric_type": metric_type,
                    "params": search_params_for_field(spec, vector_field.name),
                },
            )
            assert_search_result(response, collection, vector_field.name)
            searches += 1
        except Exception as exc:
            report.fail(
                INDEX_SEARCH_FAILED,
                "indexed vector search failed",
                collection=collection,
                field=vector_field.name,
                metric_type=metric_type,
                error=str(exc),
            )
    return searches


def _validate_query_serviceability(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    report: ValidationReport,
) -> None:
    primary = _primary_field(spec)
    primary_field = meta.get("primary_field") or (
        primary.name if primary is not None else "id"
    )
    pk_values = meta.get("pk_values")
    pk_value_fn = (
        (lambda pk, field=primary: generate_primary_key_value(field, pk))
        if primary is not None and not pk_values
        else (lambda pk: pk)
    )
    min_pk = int(meta["min_pk"])
    max_pk = int(meta["max_pk"])
    validate_collection_count(
        client,
        collection,
        int(meta["expected_count"]),
        report,
        filter_expr=pk_range_filter(
            primary_field, pk_value_fn(min_pk), pk_value_fn(max_pk)
        ),
        metric_suffix="index_compatibility_count",
    )
    mid_pk = min_pk + (max_pk - min_pk) // 2
    sample_pks = meta.get("pk_samples") or [
        pk_value_fn(min_pk),
        pk_value_fn(mid_pk),
        pk_value_fn(max_pk),
    ]
    validate_pk_samples(client, collection, primary_field, sample_pks, report)


def _index_checkpoint_path(args) -> Path:
    if args.index_checkpoint_file:
        return Path(args.index_checkpoint_file)
    return Path(args.checkpoint_dir) / "index_compatibility.json"


def _seed_checkpoint_path(args) -> Path:
    if args.checkpoint_file:
        return Path(args.checkpoint_file)
    return Path(args.checkpoint_dir) / "seed_data.json"


def _collection_items_for_phase(
    seed_checkpoint: dict[str, Any],
    index_checkpoint: dict[str, Any],
    phase: str,
) -> dict[str, dict[str, Any]]:
    seed_collections = seed_checkpoint.get("collections", {})
    if phase == "after-upgrade":
        return seed_collections
    index_collections = index_checkpoint.get("collections", {})
    return {
        collection: seed_collections[collection]
        for collection in index_collections
        if collection in seed_collections
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser(
        "Validate rollback compatibility of target-version rebuilt indexes"
    )
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "validate_index_compatibility")
    try:
        seed_checkpoint_file = _seed_checkpoint_path(args)
        if not seed_checkpoint_file.exists():
            result.status = FAILED
            result.mark_failed(
                "CHECKPOINT_NOT_FOUND",
                "seed checkpoint file does not exist",
                path=str(seed_checkpoint_file),
            )
            result.write(args.output_json)
            return 2

        index_checkpoint_file = _index_checkpoint_path(args)
        if args.phase == "after-rollback" and not index_checkpoint_file.exists():
            result.status = FAILED
            result.mark_failed(
                INDEX_COMPATIBILITY_CHECKPOINT_NOT_FOUND,
                "index compatibility checkpoint file does not exist",
                path=str(index_checkpoint_file),
            )
            result.write(args.output_json)
            return 2

        seed_checkpoint = json.loads(seed_checkpoint_file.read_text())
        index_checkpoint = (
            json.loads(index_checkpoint_file.read_text())
            if index_checkpoint_file.exists()
            else {"collections": {}}
        )
        if args.phase == "after-rollback" and not index_checkpoint.get("collections"):
            result.status = FAILED
            result.mark_failed(
                INDEX_COMPATIBILITY_CHECKPOINT_EMPTY,
                "index compatibility checkpoint has no collections to validate",
                path=str(index_checkpoint_file),
            )
            result.write(args.output_json)
            return 2

        specs = _spec_by_schema(args.schema_matrix)
        client = create_client(args.uri, args.token, args.db_name)
        report = ValidationReport()
        output_checkpoint = {
            "version": 1,
            "phase": args.phase,
            "source_seed_checkpoint": str(seed_checkpoint_file),
            "collections": {},
        }
        metrics = {
            "collections_checked": 0,
            "collections_with_index": 0,
            "indexes_rebuilt": 0,
            "indexes_dropped": 0,
            "searches_total": 0,
        }

        for collection, meta in _collection_items_for_phase(
            seed_checkpoint,
            index_checkpoint,
            args.phase,
        ).items():
            schema_name = meta["schema_name"]
            spec = specs.get(schema_name)
            if spec is None:
                report.fail(
                    "SCHEMA_NOT_FOUND",
                    "schema from checkpoint is not present in schema matrix",
                    collection=collection,
                    schema=schema_name,
                )
                continue
            metrics["collections_checked"] += 1
            indexed_fields = _indexed_fields(spec)
            if indexed_fields:
                metrics["collections_with_index"] += 1
            try:
                _flush_collection(client, collection, args.timeout_sec)
                release_status = "not_requested"
                if args.rebuild_index:
                    release_status = _release_collection_best_effort(
                        client,
                        collection,
                        args.timeout_sec,
                    )
                    failures_before_drop = len(report.failures)
                    metrics["indexes_dropped"] += _drop_indexes_for_spec(
                        client,
                        collection,
                        spec,
                        report,
                        args.timeout_sec,
                    )
                    if len(report.failures) == failures_before_drop and indexed_fields:
                        _create_indexes_for_spec(
                            client, collection, spec, args.timeout_sec
                        )
                        metrics["indexes_rebuilt"] += len(indexed_fields)
                _load_collection(client, collection, args.timeout_sec)
                _validate_query_serviceability(client, collection, spec, meta, report)
                metrics["searches_total"] += _validate_index_searches(
                    client,
                    collection,
                    spec,
                    args.seed,
                    report,
                )
                output_checkpoint["collections"][collection] = {
                    "schema_name": schema_name,
                    "indexed_fields": indexed_fields,
                    "indexed_vector_fields": [
                        field.name for field in _indexed_vector_fields(spec)
                    ],
                    "release_status": release_status,
                }
            except Exception as exc:
                report.fail(
                    "INDEX_COMPATIBILITY_VALIDATION_FAILED",
                    "index compatibility validation failed for collection",
                    collection=collection,
                    schema=schema_name,
                    error=str(exc),
                )

        index_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        index_checkpoint_file.write_text(
            json.dumps(output_checkpoint, indent=2, sort_keys=True)
        )
        result.status = PASSED if report.passed else FAILED
        result.failures = report.failures
        result.metrics = {
            **report.metrics,
            **metrics,
            "index_checkpoint_path": str(index_checkpoint_file),
        }
        result.checkpoint = {"path": str(index_checkpoint_file), "version": 1}
        result.write(args.output_json)
        return 0 if report.passed else 1
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "INDEX_COMPATIBILITY_FAILED",
            "unexpected error during index compatibility validation",
            error=str(exc),
        )
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
