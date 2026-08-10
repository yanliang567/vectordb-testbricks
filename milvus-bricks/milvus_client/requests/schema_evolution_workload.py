from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
import sys
from typing import Any

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.client import create_client
from milvus_client.common.data import (
    generate_primary_key_value,
    generate_rows,
    indexed_vector_fields,
    prepare_struct_vector_query,
    stable_checksum,
    stable_vector_value,
    vector_fields,
)
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import (
    VECTOR_TYPES,
    FieldSpec,
    FunctionSpec,
    SchemaSpec,
    auto_id_enabled,
    collection_name,
    dtype_to_milvus,
    function_output_fields,
    struct_array_for_field,
)
from milvus_client.common.validators import (
    format_filter_value,
    pk_range_filter,
    query_count,
)
from milvus_client.common.workload import (
    metric_type_for_field,
    primary_field,
    search_params_for_field,
)


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
SCHEMA_EVOLUTION_CHECKPOINT_VERSION = 2


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--rows-per-collection", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-id", type=int, default=40_000_000)
    parser.add_argument("--function-field-cycle-enabled", type=parse_bool, default=True)
    parser.add_argument(
        "--phase", choices=["after-upgrade", "after-rollback"], default="after-upgrade"
    )
    parser.add_argument("--evolution-checkpoint-file", default="")


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


def _function_cycle(
    client: Any, collection: str, spec: SchemaSpec, function: FunctionSpec
) -> str:
    if not hasattr(client, "drop_collection_function") or not hasattr(
        client, "add_collection_function"
    ):
        return "skipped"
    drop_function_field = getattr(client, "drop_function_field", None)
    add_function_field = getattr(client, "add_function_field", None)
    from pymilvus import Function, FunctionType, MilvusClient

    function_schema = Function(
        name=function.name,
        function_type=getattr(FunctionType, function.function_type),
        input_field_names=function.input_fields,
        output_field_names=function.output_fields,
        description=function.description,
        params=function.params,
    )
    if drop_function_field is not None and add_function_field is not None:
        output_names = set(function.output_fields)
        if not any(
            field_name not in output_names
            for field_name, _ in indexed_vector_fields(spec)
        ):
            return "skipped_only_vector_field"
        if len(function.output_fields) != 1:
            return "skipped_unsupported_output_count"
        output_field = next(
            (field for field in spec.fields if field.name == function.output_fields[0]),
            None,
        )
        output_index = next(
            (
                index
                for index in spec.indexes
                if index.field == function.output_fields[0]
            ),
            None,
        )
        if output_field is None or output_index is None:
            return "skipped_missing_function_field_index"
        field_kwargs: dict[str, Any] = {"nullable": output_field.nullable}
        if output_field.dim is not None:
            field_kwargs["dim"] = output_field.dim
        field_schema = MilvusClient.create_field_schema(
            name=output_field.name,
            data_type=dtype_to_milvus(output_field.dtype),
            **field_kwargs,
        )
        index_params = MilvusClient.prepare_index_params()
        index_kwargs: dict[str, Any] = {
            "field_name": output_index.field,
            "index_type": output_index.index_type,
            "metric_type": output_index.metric_type,
            "params": dict(output_index.params) or None,
        }
        if output_index.index_name:
            index_kwargs["index_name"] = output_index.index_name
        index_params.add_index(**index_kwargs)
        drop_function_field(
            collection_name=collection,
            function_name=function.name,
        )
        add_function_field(
            collection_name=collection,
            field_schema=field_schema,
            func=function_schema,
            index_params=index_params,
        )
        return "cycled"

    try:
        client.drop_collection_function(
            collection_name=collection, function_name=function.name
        )
    except Exception as exc:
        message = str(exc)
        if "drop_function_field" in message or "output field" in message:
            if drop_function_field is None:
                return "skipped_drop_function_field_api_missing"
            drop_function_field(
                collection_name=collection,
                function_name=function.name,
                field_name=function.output_fields[0] if function.output_fields else "",
            )
        else:
            raise
    client.add_collection_function(
        collection_name=collection,
        function=function_schema,
    )
    return "cycled"


def _evolved_spec(spec: SchemaSpec) -> SchemaSpec:
    existing = {field.name for field in spec.fields}
    extra_fields = []
    if EVOLUTION_FIELD.name not in existing:
        extra_fields.append(EVOLUTION_FIELD)
    return replace(spec, fields=[*spec.fields, *extra_fields])


def _nullable_vector_update_rows(
    spec: SchemaSpec, rows: list[dict[str, Any]], start_id: int
) -> int:
    nullable_vectors = [field for field in vector_fields(spec) if field.nullable]
    if not nullable_vectors:
        return 0
    updated = 0
    for offset, row in enumerate(rows):
        data_pk = start_id + offset
        for field in nullable_vectors:
            if data_pk % 2 == 0:
                row[field.name] = None
            else:
                row[field.name] = stable_vector_value(field, data_pk, 17)
            updated += 1
    return updated


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


def _evolution_field_value(
    spec: SchemaSpec,
    row: dict[str, Any],
    data_pk: int,
) -> str:
    primary = primary_field(spec)
    if auto_id_enabled(spec):
        return f"evo_auto_{data_pk}"
    primary_name = primary.name if primary else "id"
    return f"evo_{row.get(primary_name)}"


def _write_evolution_rows(
    client: Any,
    spec: SchemaSpec,
    collection: str,
    rows_per_collection: int,
    batch_size: int,
    start_id: int,
    seed: int,
) -> tuple[int, int, list[Any]]:
    evolved = _evolved_spec(spec)
    written = 0
    nullable_updates = 0
    inserted_ids: list[Any] = []
    for start in range(start_id, start_id + rows_per_collection, batch_size):
        count = min(batch_size, start_id + rows_per_collection - start)
        rows = generate_rows(evolved, start_id=start, count=count, seed=seed)
        for offset, row in enumerate(rows):
            row[EVOLUTION_FIELD.name] = _evolution_field_value(
                spec, row, start + offset
            )
        nullable_updates += _nullable_vector_update_rows(evolved, rows, start)
        if auto_id_enabled(spec):
            batch_ids = _extract_insert_ids(
                client.insert(collection_name=collection, data=rows)
            )
            if len(batch_ids) != len(rows) or any(pk is None for pk in batch_ids):
                raise AssertionError(
                    f"{collection}: auto-id schema evolution insert returned "
                    f"{len(batch_ids)} primary keys for {len(rows)} rows"
                )
            inserted_ids.extend(batch_ids)
        else:
            client.upsert(collection_name=collection, data=rows)
        written += len(rows)
    if auto_id_enabled(spec) and len(set(inserted_ids)) != len(inserted_ids):
        raise AssertionError(
            f"{collection}: auto-id schema evolution insert returned duplicate primary keys"
        )
    return (written, nullable_updates, inserted_ids)


def _prepare_collection_for_read(
    client: Any,
    collection: str,
    *,
    flush: bool,
) -> None:
    if flush and hasattr(client, "flush"):
        try:
            client.flush(collection_name=collection)
        except TypeError:
            client.flush(collection)
    if hasattr(client, "load_collection"):
        try:
            client.load_collection(collection_name=collection)
        except TypeError:
            client.load_collection(collection)


def _expected_evolution_row(
    spec: SchemaSpec,
    data_pk: int,
    seed: int,
    actual_pk: Any | None = None,
) -> dict[str, Any]:
    evolved = _evolved_spec(spec)
    row = generate_rows(evolved, start_id=data_pk, count=1, seed=seed)[0]
    primary = primary_field(spec)
    if auto_id_enabled(spec):
        if actual_pk is None:
            raise AssertionError(
                f"{spec.name}: auto-id evolution row requires an actual primary key"
            )
        row[primary.name if primary else "id"] = actual_pk
    row[EVOLUTION_FIELD.name] = _evolution_field_value(spec, row, data_pk)
    _nullable_vector_update_rows(evolved, [row], data_pk)
    return row


def _validation_fields(spec: SchemaSpec) -> list[str]:
    primary = primary_field(spec)
    function_outputs = function_output_fields(spec)
    return [
        primary.name if primary else "id",
        *(
            field.name
            for field in spec.fields
            if field.dtype in VECTOR_TYPES and field.name not in function_outputs
        ),
        *(struct_array.name for struct_array in spec.struct_arrays),
        EVOLUTION_FIELD.name,
    ]


def _expected_primary_value(
    spec: SchemaSpec,
    data_pk: int,
    start_id: int,
    pk_values: list[Any],
) -> Any:
    primary = primary_field(spec)
    if auto_id_enabled(spec):
        offset = data_pk - start_id
        if offset < 0 or offset >= len(pk_values):
            raise AssertionError(
                f"{spec.name}: auto-id checkpoint has no primary key for data PK {data_pk}"
            )
        return pk_values[offset]
    return generate_primary_key_value(primary, data_pk) if primary else data_pk


def _pk_values_filter(primary_name: str, pk_values: list[Any]) -> str:
    return (
        f"{primary_name} in ["
        + ", ".join(format_filter_value(value) for value in pk_values)
        + "]"
    )


def _sample_data_pks(start_id: int, rows_per_collection: int) -> list[int]:
    if rows_per_collection <= 0:
        return []
    candidates = [
        start_id,
        start_id + (rows_per_collection - 1) // 2,
        start_id + rows_per_collection - 1,
    ]
    return list(dict.fromkeys(candidates))


def _hit_value(hit: Any, key: str) -> Any:
    if isinstance(hit, dict):
        if key in hit:
            return hit[key]
        entity = hit.get("entity")
        if isinstance(entity, dict):
            return entity.get(key)
    if hasattr(hit, key):
        return getattr(hit, key)
    return None


def _hit_primary_key(hit: Any, primary_name: str) -> Any:
    for key in ("id", "pk", primary_name):
        value = _hit_value(hit, key)
        if value is not None:
            return value
    return None


def _hit_offset(hit: Any) -> int | None:
    value = _hit_value(hit, "offset")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hit_distance(hit: Any) -> float | None:
    for key in ("distance", "score"):
        value = _hit_value(hit, key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _search_probe(
    spec: SchemaSpec,
    field_name: str,
    vector_field: FieldSpec,
    start_id: int,
    rows_per_collection: int,
    seed: int,
    preferred_data_pk: int | None = None,
) -> tuple[int, Any, int | None] | None:
    function_outputs = function_output_fields(spec)
    data_pks = (
        [preferred_data_pk]
        if preferred_data_pk is not None
        else range(start_id, start_id + rows_per_collection)
    )
    for data_pk in data_pks:
        if data_pk is None:
            continue
        row = _expected_evolution_row(
            spec,
            data_pk,
            seed,
            actual_pk=0 if auto_id_enabled(spec) else None,
        )
        if vector_field.name in function_outputs:
            function = next(
                item
                for item in spec.functions
                if vector_field.name in item.output_fields
            )
            query = row.get(function.input_fields[0])
            if query is not None and query != "":
                return data_pk, query, None
            continue
        struct_array = struct_array_for_field(spec, field_name)
        if struct_array is not None:
            values = row.get(struct_array.name)
            if not values:
                continue
            for offset, element in enumerate(values):
                vector = element.get(vector_field.name)
                if vector is not None:
                    query, expected_offset = prepare_struct_vector_query(
                        metric_type_for_field(spec, field_name), vector, offset
                    )
                    return data_pk, query, expected_offset
            continue
        query = row.get(vector_field.name)
        if query is not None:
            return data_pk, query, None
    return None


def _assert_metric_self_match(
    collection: str,
    field_name: str,
    metric_type: str,
    index_type: str,
    distance: float,
) -> None:
    metric = metric_type.upper().removeprefix("MAX_SIM_")
    if metric == "BM25":
        return
    lossy_index = index_type.upper() in {
        "IVF_PQ",
        "IVF_SQ8",
        "HNSW_SQ",
        "IVF_RABITQ",
        "SCANN",
    }
    if metric in {"L2", "HAMMING", "JACCARD"}:
        max_distance = 0.5 if lossy_index and metric == "L2" else 1e-3
        if distance < 0 or distance > max_distance:
            raise AssertionError(
                f"{collection}.{field_name}: self-match distance {distance} is "
                f"outside [0, {max_distance}] for {metric_type}/{index_type}"
            )
        return
    if metric in {"COSINE", "IP", "MHJACCARD"}:
        min_score = 0.5 if lossy_index and metric in {"COSINE", "IP"} else 0.9
        if distance < min_score or (
            metric in {"COSINE", "MHJACCARD"} and distance > 1.001
        ):
            raise AssertionError(
                f"{collection}.{field_name}: self-match score {distance} is "
                f"outside the expected range for {metric_type}/{index_type}"
            )


def _validate_search_probe(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    field_name: str,
    vector_field: FieldSpec,
    start_id: int,
    rows_per_collection: int,
    seed: int,
    pk_values: list[Any],
    preferred_data_pk: int | None = None,
) -> dict[str, Any]:
    probe = _search_probe(
        spec,
        field_name,
        vector_field,
        start_id,
        rows_per_collection,
        seed,
        preferred_data_pk,
    )
    if probe is None:
        raise AssertionError(f"{collection}.{field_name}: no non-null search probe")
    data_pk, query_vector, expected_offset = probe
    primary = primary_field(spec)
    primary_name = primary.name if primary else "id"
    expected_pk = _expected_primary_value(spec, data_pk, start_id, pk_values)
    metric_type = metric_type_for_field(spec, field_name)
    index_type = next(
        index.index_type for index in spec.indexes if index.field == field_name
    )
    result = client.search(
        collection_name=collection,
        data=[query_vector],
        anns_field=field_name,
        filter=f"{primary_name} == {format_filter_value(expected_pk)}",
        limit=5,
        search_params={
            "metric_type": metric_type,
            "params": search_params_for_field(spec, field_name),
        },
    )
    if not isinstance(result, list) or len(result) != 1 or not result[0]:
        raise AssertionError(f"{collection}.{field_name}: search returned no hits")
    expected_hit = next(
        (
            hit
            for hit in result[0]
            if _hit_primary_key(hit, primary_name) == expected_pk
            and (expected_offset is None or _hit_offset(hit) == expected_offset)
        ),
        None,
    )
    if expected_hit is None:
        raise AssertionError(
            f"{collection}.{field_name}: search missed expected primary key "
            f"{expected_pk!r} and offset {expected_offset!r}"
        )
    distance = _hit_distance(expected_hit)
    if distance is None or not math.isfinite(distance):
        raise AssertionError(
            f"{collection}.{field_name}: expected hit has no finite score/distance"
        )
    _assert_metric_self_match(
        collection,
        field_name,
        metric_type,
        index_type,
        distance,
    )
    return {
        "field": field_name,
        "data_pk": data_pk,
        "expected_pk": expected_pk,
        "expected_offset": expected_offset,
        "metric_type": metric_type,
        "index_type": index_type,
    }


def _read_validate(
    client: Any,
    spec: SchemaSpec,
    collection: str,
    start_id: int,
    rows_per_collection: int,
    seed: int,
    pk_values: list[Any] | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
) -> tuple[int, int, int, dict[str, Any]]:
    primary = primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    pk_values = list(pk_values or [])
    if auto_id_enabled(spec):
        if len(pk_values) != rows_per_collection:
            raise AssertionError(
                f"{collection}: auto-id checkpoint has {len(pk_values)} primary keys "
                f"for {rows_per_collection} evolved rows"
            )
        count = 0
        for offset in range(0, len(pk_values), 100):
            count += query_count(
                client,
                collection,
                filter_expr=_pk_values_filter(
                    primary_name, pk_values[offset : offset + 100]
                ),
            )
        min_pk = min(pk_values)
        max_pk = max(pk_values)
        count_filter = "auto-id checkpoint primary-key batches"
    else:
        min_pk = generate_primary_key_value(primary, start_id) if primary else start_id
        max_data_pk = start_id + rows_per_collection - 1
        max_pk = (
            generate_primary_key_value(primary, max_data_pk) if primary else max_data_pk
        )
        count_filter = pk_range_filter(primary_name, min_pk, max_pk)
        count = query_count(client, collection, filter_expr=count_filter)
    if count != rows_per_collection:
        raise AssertionError(
            f"{collection}: expected {rows_per_collection} evolved rows in "
            f"{count_filter}, got {count}"
        )

    expected_validation_fields = _validation_fields(spec)
    validation_fields = list(
        (checkpoint_meta or {}).get("validation_fields") or expected_validation_fields
    )
    if checkpoint_meta is not None and validation_fields != expected_validation_fields:
        raise AssertionError(
            f"{collection}: schema evolution checkpoint validation fields differ "
            f"from the schema contract: {validation_fields!r} != "
            f"{expected_validation_fields!r}"
        )
    sample_data_pks = list(
        (checkpoint_meta or {}).get("sample_data_pks")
        or _sample_data_pks(start_id, rows_per_collection)
    )
    expected_checkpoint_checksums = {
        int(item["data_pk"]): item["checksum"]
        for item in (checkpoint_meta or {}).get("sample_checksums", [])
    }
    sample_checksums = []
    for data_pk in sample_data_pks:
        expected_pk = _expected_primary_value(spec, data_pk, start_id, pk_values)
        rows = client.query(
            collection_name=collection,
            filter=f"{primary_name} == {format_filter_value(expected_pk)}",
            output_fields=validation_fields,
            limit=2,
        )
        matching = [row for row in rows if row.get(primary_name) == expected_pk]
        if len(matching) != 1:
            raise AssertionError(
                f"{collection}: expected exactly one evolved row for primary key "
                f"{expected_pk!r}, got {len(matching)}"
            )
        actual_checksum = stable_checksum(
            matching,
            fields=validation_fields,
            primary_field=primary_name,
        )
        if checkpoint_meta is None:
            expected_row = _expected_evolution_row(
                spec,
                data_pk,
                seed,
                actual_pk=expected_pk if auto_id_enabled(spec) else None,
            )
            expected_checksum = stable_checksum(
                [expected_row],
                fields=validation_fields,
                primary_field=primary_name,
            )
            if actual_checksum != expected_checksum:
                raise AssertionError(
                    f"{collection}: evolved field checksum differs from deterministic "
                    f"ground truth for primary key {expected_pk!r}"
                )
        elif actual_checksum != expected_checkpoint_checksums.get(data_pk):
            raise AssertionError(
                f"{collection}: evolved row checkpoint checksum differs for primary "
                f"key {expected_pk!r}"
            )
        sample_checksums.append(
            {
                "data_pk": data_pk,
                "pk": expected_pk,
                "checksum": actual_checksum,
            }
        )

    checkpoint_probes = {
        item["field"]: item for item in (checkpoint_meta or {}).get("search_probes", [])
    }
    search_probes = []
    for field_name, vector_field in indexed_vector_fields(spec):
        preferred_data_pk = checkpoint_probes.get(field_name, {}).get("data_pk")
        search_probes.append(
            _validate_search_probe(
                client,
                collection,
                spec,
                field_name,
                vector_field,
                start_id,
                rows_per_collection,
                seed,
                pk_values,
                preferred_data_pk,
            )
        )

    observation = {
        "schema_name": spec.name,
        "primary_field": primary_name,
        "start_id": start_id,
        "rows_per_collection": rows_per_collection,
        "expected_count": rows_per_collection,
        "min_pk": min_pk,
        "max_pk": max_pk,
        "pk_values": pk_values,
        "seed": seed,
        "validation_fields": validation_fields,
        "sample_data_pks": sample_data_pks,
        "sample_checksums": sample_checksums,
        "search_probes": search_probes,
    }
    return (len(sample_checksums), count, len(search_probes), observation)


def run_schema_evolution(
    client: Any,
    specs: list[SchemaSpec],
    collection_prefix: str,
    rows_per_collection: int,
    batch_size: int,
    start_id: int,
    seed: int,
    function_field_cycle_enabled: bool = True,
    phase: str = "after-upgrade",
    checkpoint: dict[str, Any] | None = None,
    checkpoint_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in {"after-upgrade", "after-rollback"}:
        raise ValueError(f"unsupported schema evolution phase: {phase}")
    if phase == "after-rollback" and not (checkpoint or {}).get("collections"):
        raise ValueError("schema evolution checkpoint has no collections")
    specs_by_name = {spec.name: spec for spec in specs}
    collection_items: list[tuple[str, SchemaSpec, dict[str, Any] | None]]
    if phase == "after-upgrade":
        collection_items = [
            (collection_name(collection_prefix, spec), spec, None) for spec in specs
        ]
    else:
        collection_items = []
        checkpoint_collections = (checkpoint or {}).get("collections", {})
        expected_collections = {
            collection_name(collection_prefix, spec) for spec in specs
        }
        if set(checkpoint_collections) != expected_collections:
            raise ValueError(
                "schema evolution checkpoint collection set differs from the "
                f"schema matrix contract: expected {sorted(expected_collections)}, "
                f"got {sorted(checkpoint_collections)}"
            )
        for collection, checkpoint_meta in checkpoint_collections.items():
            schema_name = str(checkpoint_meta.get("schema_name") or "")
            spec = specs_by_name.get(schema_name)
            if spec is None:
                raise ValueError(
                    f"schema evolution checkpoint references unknown schema {schema_name!r}"
                )
            collection_items.append((collection, spec, checkpoint_meta))

    if checkpoint_output is not None:
        checkpoint_output.clear()
        checkpoint_output.update(
            {
                "version": SCHEMA_EVOLUTION_CHECKPOINT_VERSION,
                "phase": "after-upgrade",
                "collection_prefix": collection_prefix,
                "schema_names": [spec.name for spec in specs],
                "collections": {},
            }
        )
    metrics: dict[str, Any] = {
        "phase": phase,
        "collections_total": len(collection_items),
        "field_add_total": 0,
        "field_add_exists_total": 0,
        "field_add_skipped_total": 0,
        "drop_field_skipped_total": 0,
        "function_cycles_total": 0,
        "function_cycle_skipped_total": 0,
        "written_total": 0,
        "auto_id_inserted_total": 0,
        "upserted_total": 0,
        "nullable_updates_total": 0,
        "queries_total": 0,
        "searches_total": 0,
        "count_checks_total": 0,
        "failed_total": 0,
        "collections": [],
    }
    for collection, spec, checkpoint_meta in collection_items:
        collection_metrics: dict[str, Any] = {
            "schema": spec.name,
            "collection": collection,
        }
        try:
            if not client.has_collection(collection):
                raise RuntimeError(f"{collection} does not exist")
            if phase == "after-upgrade":
                add_status = _add_field(client, collection, EVOLUTION_FIELD)
                collection_metrics["add_field"] = add_status
                metrics["field_add_total"] += 1
                metrics[f"field_add_{add_status}_total"] = (
                    metrics.get(f"field_add_{add_status}_total", 0) + 1
                )

                drop_add_status = _add_field(client, collection, EVOLUTION_DROP_FIELD)
                drop_status = _drop_field(client, collection, EVOLUTION_DROP_FIELD.name)
                collection_metrics["drop_field"] = (
                    drop_status if drop_add_status != "skipped" else "skipped"
                )
                if collection_metrics["drop_field"] == "skipped":
                    metrics["drop_field_skipped_total"] += 1

                cycled = 0
                skipped = 0
                for function in spec.functions:
                    status = (
                        _function_cycle(client, collection, spec, function)
                        if function_field_cycle_enabled
                        else "skipped_disabled"
                    )
                    if status == "cycled":
                        cycled += 1
                    else:
                        skipped += 1
                        collection_metrics.setdefault(
                            "function_cycle_skip_reasons", []
                        ).append(status)
                metrics["function_cycles_total"] += cycled
                metrics["function_cycle_skipped_total"] += skipped
                collection_metrics["function_cycles"] = cycled
                collection_metrics["function_cycle_skipped"] = skipped

                written, nullable_updates, inserted_ids = _write_evolution_rows(
                    client,
                    spec,
                    collection,
                    rows_per_collection,
                    batch_size,
                    start_id,
                    seed,
                )
                if written != rows_per_collection:
                    raise AssertionError(
                        f"{collection}: wrote {written} rows, expected "
                        f"{rows_per_collection}"
                    )
                collection_metrics["write_operation"] = (
                    "insert" if auto_id_enabled(spec) else "upsert"
                )
                collection_metrics["written"] = written
                collection_metrics["inserted_ids"] = len(inserted_ids)
                collection_metrics["nullable_updates"] = nullable_updates
                metrics["written_total"] += written
                if auto_id_enabled(spec):
                    metrics["auto_id_inserted_total"] += written
                else:
                    metrics["upserted_total"] += written
                metrics["nullable_updates_total"] += nullable_updates
            else:
                written = int(checkpoint_meta.get("expected_count", 0))
                inserted_ids = list(checkpoint_meta.get("pk_values") or [])

            _prepare_collection_for_read(
                client,
                collection,
                flush=phase == "after-upgrade",
            )
            effective_start_id = int(
                checkpoint_meta.get("start_id", start_id)
                if checkpoint_meta is not None
                else start_id
            )
            effective_rows = int(
                checkpoint_meta.get("rows_per_collection", rows_per_collection)
                if checkpoint_meta is not None
                else rows_per_collection
            )
            effective_seed = int(
                checkpoint_meta.get("seed", seed)
                if checkpoint_meta is not None
                else seed
            )
            effective_pk_values = (
                list(checkpoint_meta.get("pk_values") or [])
                if checkpoint_meta is not None
                else inserted_ids
            )
            queries, count, searches, observation = _read_validate(
                client,
                spec,
                collection,
                effective_start_id,
                effective_rows,
                effective_seed,
                effective_pk_values,
                checkpoint_meta,
            )
            collection_metrics["query_checks"] = queries
            collection_metrics["count"] = count
            collection_metrics["searches"] = searches
            metrics["queries_total"] += queries
            metrics["count_checks_total"] += 1
            metrics["searches_total"] += searches
            if phase == "after-upgrade" and checkpoint_output is not None:
                checkpoint_output["collections"][collection] = observation
        except Exception as exc:
            metrics["failed_total"] += 1
            collection_metrics["error"] = str(exc)
        metrics["collections"].append(collection_metrics)
    return metrics


def _evolution_checkpoint_path(args) -> Path:
    if args.evolution_checkpoint_file:
        return Path(args.evolution_checkpoint_file)
    return Path(args.checkpoint_dir) / "schema_evolution.json"


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser(
        "Run schema evolution workload against existing Milvus collections"
    )
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "schema_evolution_workload")
    try:
        from milvus_client.common.schema import load_schema_matrix

        checkpoint_path = _evolution_checkpoint_path(args)
        checkpoint = None
        if args.phase == "after-rollback":
            if not checkpoint_path.exists():
                result.status = FAILED
                result.mark_failed(
                    "SCHEMA_EVOLUTION_CHECKPOINT_NOT_FOUND",
                    "schema evolution checkpoint file does not exist",
                    path=str(checkpoint_path),
                )
                result.write(args.output_json)
                return 2
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get(
                "version"
            ) != SCHEMA_EVOLUTION_CHECKPOINT_VERSION or not checkpoint.get(
                "collections"
            ):
                result.status = FAILED
                result.mark_failed(
                    "SCHEMA_EVOLUTION_CHECKPOINT_INVALID",
                    "schema evolution checkpoint is empty or has an unsupported version",
                    path=str(checkpoint_path),
                )
                result.write(args.output_json)
                return 2

        checkpoint_output: dict[str, Any] | None = (
            {} if args.phase == "after-upgrade" else None
        )
        client = create_client(args.uri, args.token, args.db_name)
        metrics = run_schema_evolution(
            client,
            load_schema_matrix(args.schema_matrix),
            args.collection_prefix,
            args.rows_per_collection,
            args.batch_size,
            args.start_id,
            args.seed,
            args.function_field_cycle_enabled,
            args.phase,
            checkpoint,
            checkpoint_output,
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
        if result.status == PASSED:
            if checkpoint_output is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps(checkpoint_output, indent=2, sort_keys=True)
                )
            result.checkpoint = {
                "path": str(checkpoint_path),
                "version": SCHEMA_EVOLUTION_CHECKPOINT_VERSION,
            }
        result.write(args.output_json)
        return 1 if result.status == FAILED else 0
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "SCHEMA_EVOLUTION_FAILED",
            "unexpected schema evolution failure",
            error=str(exc),
        )
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
