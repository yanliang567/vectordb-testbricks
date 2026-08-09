from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Any

from milvus_client.common.data import (
    generate_field_value,
    generate_primary_key_value,
    generate_struct_array_value,
    prepare_struct_vector_query,
    stable_vector_value,
    text_payload_metadata,
)
from milvus_client.common.schema import (
    VECTOR_TYPES,
    FieldSpec,
    IndexSpec,
    SchemaSpec,
    function_output_fields,
    resolve_field,
    struct_array_for_field,
)
from milvus_client.common.validators import ValidationReport, format_filter_value
from milvus_client.common.workload import metric_type_for_field, search_params_for_field


EXTERNAL_VALIDATORS = {"count", "pk_sample", "search_smoke"}
FEATURE_VALIDATORS = {
    "nullable_vector_semantics",
    "struct_array_scalar_round_trip",
    "struct_array_element_search",
    "struct_array_scalar_index_queries",
    "geometry_filter",
    "text_lob_round_trip",
    "text_match_phrase_match",
    "minhash_search",
    "entity_ttl",
    "index_engine_version",
}


def known_validator_names() -> set[str]:
    return EXTERNAL_VALIDATORS | FEATURE_VALIDATORS


def unknown_validators(spec: SchemaSpec) -> list[str]:
    known = known_validator_names()
    return sorted(
        {validator for validator in spec.validators if validator not in known}
    )


def _primary_field(spec: SchemaSpec) -> FieldSpec:
    primary = [field for field in spec.fields if field.primary]
    if len(primary) != 1:
        raise ValueError(f"{spec.name}: expected exactly one primary field")
    return primary[0]


def _data_pk_range(meta: dict[str, Any]) -> tuple[int, int]:
    return (
        int(meta.get("data_min_pk", meta["min_pk"])),
        int(meta.get("data_max_pk", meta["max_pk"])),
    )


def _actual_primary_value(spec: SchemaSpec, meta: dict[str, Any], data_pk: int) -> Any:
    primary = _primary_field(spec)
    pk_values = meta.get("pk_values") or []
    if primary.auto_id and pk_values:
        data_min_pk, _ = _data_pk_range(meta)
        offset = data_pk - data_min_pk
        if 0 <= offset < len(pk_values):
            return pk_values[offset]
        return pk_values[0]
    return generate_primary_key_value(primary, data_pk)


def _query_by_data_pk(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    data_pk: int,
    output_fields: list[str],
) -> list[dict[str, Any]]:
    primary = _primary_field(spec)
    actual_pk = _actual_primary_value(spec, meta, data_pk)
    return client.query(
        collection_name=collection,
        filter=f"{primary.name} == {format_filter_value(actual_pk)}",
        output_fields=list(dict.fromkeys([primary.name, *output_fields])),
        limit=1,
    )


def _normalize(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    return value


def _hit_value(hit: Any, key: str) -> Any:
    if isinstance(hit, dict):
        if key in hit:
            return hit[key]
        entity = hit.get("entity")
        if isinstance(entity, dict) and key in entity:
            return entity[key]
    if hasattr(hit, key):
        return getattr(hit, key)
    if hasattr(hit, "get"):
        try:
            return hit.get(key)
        except Exception:
            return None
    return None


def _hit_pk(hit: Any, primary_name: str) -> Any:
    for key in ("id", "pk", primary_name):
        value = _hit_value(hit, key)
        if value is not None:
            return value
    return None


def _search_hits(response: Any) -> list[Any]:
    if not isinstance(response, list) or len(response) != 1:
        return []
    return list(response[0] or [])


def _record_pass(report: ValidationReport, collection: str, validator: str) -> None:
    key = f"{collection}.{validator}.passed"
    report.metrics[key] = int(report.metrics.get(key, 0)) + 1


def validate_nullable_vector_semantics(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    primary = _primary_field(spec)
    data_min_pk, data_max_pk = _data_pk_range(meta)
    nullable_vectors = [
        field for field in spec.fields if field.dtype in VECTOR_TYPES and field.nullable
    ]
    if not nullable_vectors:
        report.fail(
            "FEATURE_VALIDATION_TARGET_MISSING",
            "nullable vector validator found no nullable vector fields",
            collection=collection,
            schema=spec.name,
        )
        return
    for field in nullable_vectors:
        null_pk = next(
            (pk for pk in range(data_min_pk, data_max_pk + 1) if pk % 10 == 0),
            None,
        )
        value_pk = next(
            (pk for pk in range(data_min_pk, data_max_pk + 1) if pk % 10 != 0),
            None,
        )
        if null_pk is None or value_pk is None:
            report.fail(
                "FEATURE_VALIDATION_FAILED",
                "nullable vector checkpoint range lacks null or non-null probe rows",
                collection=collection,
                field=field.name,
            )
            continue
        null_rows = _query_by_data_pk(
            client, collection, spec, meta, null_pk, [field.name]
        )
        value_rows = _query_by_data_pk(
            client, collection, spec, meta, value_pk, [field.name]
        )
        if not null_rows or null_rows[0].get(field.name) is not None:
            report.fail(
                "NULLABLE_VECTOR_MISMATCH",
                "nullable vector null state changed",
                collection=collection,
                field=field.name,
                data_pk=null_pk,
                actual=null_rows,
            )
            continue
        if not value_rows or value_rows[0].get(field.name) is None:
            report.fail(
                "NULLABLE_VECTOR_MISMATCH",
                "nullable vector non-null state changed",
                collection=collection,
                field=field.name,
                data_pk=value_pk,
                actual=value_rows,
            )
            continue
        query_vector = stable_vector_value(field, value_pk, seed)
        actual_pk = _actual_primary_value(spec, meta, value_pk)
        response = client.search(
            collection_name=collection,
            data=[query_vector],
            anns_field=field.name,
            filter=f"{primary.name} == {format_filter_value(actual_pk)}",
            limit=5,
            search_params={
                "metric_type": metric_type_for_field(spec, field.name),
                "params": search_params_for_field(spec, field.name),
            },
        )
        if actual_pk not in [
            _hit_pk(hit, primary.name) for hit in _search_hits(response)
        ]:
            report.fail(
                "NULLABLE_VECTOR_SEARCH_FAILED",
                "non-null vector self-search did not return the expected row",
                collection=collection,
                field=field.name,
                expected_pk=actual_pk,
            )
            continue
        _record_pass(report, collection, "nullable_vector_semantics")


def _struct_scalar_projection(spec: SchemaSpec, struct_name: str, value: Any) -> Any:
    struct_array = next(item for item in spec.struct_arrays if item.name == struct_name)
    scalar_names = {
        field.name for field in struct_array.fields if field.dtype not in VECTOR_TYPES
    }
    if value is None:
        return None
    return [
        {name: _normalize(element.get(name)) for name in scalar_names}
        for element in value
    ]


def validate_struct_array_scalar_round_trip(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    data_min_pk, data_max_pk = _data_pk_range(meta)
    candidates = [
        data_min_pk,
        data_min_pk + (data_max_pk - data_min_pk) // 2,
        data_max_pk,
    ]
    struct_arrays = [
        struct_array
        for struct_array in spec.struct_arrays
        if any(field.dtype not in VECTOR_TYPES for field in struct_array.fields)
    ]
    if not struct_arrays:
        report.fail(
            "FEATURE_VALIDATION_TARGET_MISSING",
            "StructArray scalar round-trip validator found no scalar sub-fields",
            collection=collection,
            schema=spec.name,
        )
        return
    for struct_array in struct_arrays:
        for data_pk in dict.fromkeys(candidates):
            expected = generate_struct_array_value(struct_array, data_pk, seed)
            rows = _query_by_data_pk(
                client, collection, spec, meta, data_pk, [struct_array.name]
            )
            actual = rows[0].get(struct_array.name) if rows else None
            expected_projection = _struct_scalar_projection(
                spec, struct_array.name, expected
            )
            actual_projection = _struct_scalar_projection(
                spec, struct_array.name, actual
            )
            if actual_projection != expected_projection:
                report.fail(
                    "STRUCT_ARRAY_ROUND_TRIP_MISMATCH",
                    "StructArray scalar sub-fields changed across lifecycle phase",
                    collection=collection,
                    field=struct_array.name,
                    data_pk=data_pk,
                    expected=expected_projection,
                    actual=actual_projection,
                )
                continue
            _record_pass(report, collection, "struct_array_scalar_round_trip")


def _struct_vector_indexes(spec: SchemaSpec) -> list[tuple[IndexSpec, FieldSpec]]:
    values = []
    for index in spec.indexes:
        field = resolve_field(spec, index.field)
        if field is None or field.dtype not in VECTOR_TYPES:
            continue
        if struct_array_for_field(spec, index.field) is not None:
            values.append((index, field))
    return values


def validate_struct_array_element_search(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    primary = _primary_field(spec)
    data_min_pk, data_max_pk = _data_pk_range(meta)
    vector_indexes = _struct_vector_indexes(spec)
    if not vector_indexes:
        report.fail(
            "FEATURE_VALIDATION_TARGET_MISSING",
            "StructArray vector validator found no indexed vector sub-fields",
            collection=collection,
            schema=spec.name,
        )
        return
    for index, field in vector_indexes:
        struct_array = struct_array_for_field(spec, index.field)
        if struct_array is None:
            continue
        probe = None
        for data_pk in range(data_min_pk, data_max_pk + 1):
            generated = generate_struct_array_value(struct_array, data_pk, seed)
            if not generated:
                continue
            for offset, element in enumerate(generated):
                if element.get(field.name) is not None:
                    probe = (data_pk, offset, element[field.name])
                    break
            if probe is not None:
                break
        if probe is None:
            report.fail(
                "STRUCT_ARRAY_SEARCH_FAILED",
                "could not build a deterministic StructArray vector probe",
                collection=collection,
                field=index.field,
            )
            continue
        data_pk, expected_offset, query_vector = probe
        metric_type = metric_type_for_field(spec, index.field)
        query_vector, expected_offset = prepare_struct_vector_query(
            metric_type, query_vector, expected_offset
        )
        expected_pk = _actual_primary_value(spec, meta, data_pk)
        response = client.search(
            collection_name=collection,
            data=[query_vector],
            anns_field=index.field,
            filter=f"{primary.name} == {format_filter_value(expected_pk)}",
            limit=5,
            search_params={
                "metric_type": metric_type,
                "params": search_params_for_field(spec, index.field),
            },
        )
        matching = []
        for hit in _search_hits(response):
            raw_offset = _hit_value(hit, "offset")
            try:
                actual_offset = int(raw_offset)
            except (TypeError, ValueError):
                actual_offset = -1
            if _hit_pk(hit, primary.name) == expected_pk and (
                expected_offset is None or actual_offset == expected_offset
            ):
                matching.append(hit)
        if not matching:
            report.fail(
                "STRUCT_ARRAY_SEARCH_FAILED",
                "StructArray vector search did not return the expected row or element",
                collection=collection,
                field=index.field,
                expected_pk=expected_pk,
                expected_offset=expected_offset,
            )
            continue
        _record_pass(report, collection, "struct_array_element_search")


def _struct_scalar_filter(
    spec: SchemaSpec, index: IndexSpec, field: FieldSpec, data_pk: int, seed: int
) -> str | None:
    struct_array = struct_array_for_field(spec, index.field)
    if struct_array is None:
        return None
    generated = generate_struct_array_value(struct_array, data_pk, seed)
    if not generated:
        return None
    value = generated[0].get(field.name)
    operator = (
        ">="
        if index.index_type == "STL_SORT"
        and field.dtype in {"INT8", "INT16", "INT32", "INT64", "FLOAT", "DOUBLE"}
        else "=="
    )
    return (
        f"MATCH_ANY({struct_array.name}, $[{field.name}] {operator} "
        f"{format_filter_value(value)})"
    )


def validate_struct_array_scalar_index_queries(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    primary = _primary_field(spec)
    data_min_pk, data_max_pk = _data_pk_range(meta)
    executed = 0
    scalar_indexes = [
        index
        for index in spec.indexes
        if (field := resolve_field(spec, index.field)) is not None
        and field.dtype not in VECTOR_TYPES
        and struct_array_for_field(spec, index.field) is not None
    ]
    if not scalar_indexes:
        report.fail(
            "FEATURE_VALIDATION_TARGET_MISSING",
            "StructArray scalar index validator found no indexed scalar sub-fields",
            collection=collection,
            schema=spec.name,
        )
    for index in scalar_indexes:
        field = resolve_field(spec, index.field)
        if field is None:
            continue
        probe = None
        for data_pk in range(data_min_pk, data_max_pk + 1):
            filter_expr = _struct_scalar_filter(spec, index, field, data_pk, seed)
            if filter_expr:
                probe = (data_pk, filter_expr)
                break
        if probe is None:
            report.fail(
                "STRUCT_ARRAY_SCALAR_INDEX_FAILED",
                "could not build StructArray scalar index query",
                collection=collection,
                field=index.field,
            )
            continue
        data_pk, scalar_filter = probe
        expected_pk = _actual_primary_value(spec, meta, data_pk)
        rows = client.query(
            collection_name=collection,
            filter=(
                f"({scalar_filter}) && "
                f"{primary.name} == {format_filter_value(expected_pk)}"
            ),
            output_fields=[primary.name],
            limit=1,
        )
        if expected_pk not in [row.get(primary.name) for row in rows]:
            report.fail(
                "STRUCT_ARRAY_SCALAR_INDEX_FAILED",
                "StructArray scalar index query missed the deterministic row",
                collection=collection,
                field=index.field,
                filter=scalar_filter,
                expected_pk=expected_pk,
            )
            continue
        executed += 1
        _record_pass(report, collection, "struct_array_scalar_index_queries")
    report.metrics[f"{collection}.struct_array_scalar_index_queries.total"] = executed
    expected_minimum = int(
        spec.validator_params.get("min_struct_scalar_index_queries", 1)
    )
    if executed < expected_minimum:
        report.fail(
            "STRUCT_ARRAY_SCALAR_INDEX_COVERAGE_INCOMPLETE",
            "fewer StructArray scalar index queries executed than required",
            collection=collection,
            schema=spec.name,
            expected_minimum=expected_minimum,
            actual=executed,
        )


def validate_geometry_filter(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    del seed
    primary = _primary_field(spec)
    geometry_fields = [field for field in spec.fields if field.dtype == "GEOMETRY"]
    if not geometry_fields:
        report.fail(
            "FEATURE_VALIDATION_TARGET_MISSING",
            "Geometry validator found no Geometry fields",
            collection=collection,
            schema=spec.name,
        )
        return
    data_pk, _ = _data_pk_range(meta)
    expected_pk = _actual_primary_value(spec, meta, data_pk)
    for field in geometry_fields:
        value = generate_field_value(field, data_pk, 0)
        escaped = str(value).replace("'", "\\'")
        filters = [
            f"ST_EQUALS({field.name}, '{escaped}')",
            f"ST_DWITHIN({field.name}, '{escaped}', 0)",
        ]
        for filter_expr in filters:
            rows = client.query(
                collection_name=collection,
                filter=(
                    f"({filter_expr}) && "
                    f"{primary.name} == {format_filter_value(expected_pk)}"
                ),
                output_fields=[primary.name],
                limit=1,
            )
            if expected_pk not in [row.get(primary.name) for row in rows]:
                report.fail(
                    "GEOMETRY_FILTER_FAILED",
                    "Geometry index predicate missed the deterministic row",
                    collection=collection,
                    field=field.name,
                    filter=filter_expr,
                    expected_pk=expected_pk,
                )
                continue
            _record_pass(report, collection, "geometry_filter")


def _boundary_pks(data_min_pk: int, data_max_pk: int) -> list[int]:
    values = []
    for slot in range(7):
        candidate = data_min_pk + ((slot - data_min_pk) % 1000)
        if candidate <= data_max_pk:
            values.append(candidate)
    return values


def validate_text_lob_round_trip(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    data_min_pk, data_max_pk = _data_pk_range(meta)
    text_fields = [
        field for field in spec.fields if field.value_profile == "text_lob_boundary"
    ]
    if not text_fields:
        report.fail(
            "FEATURE_VALIDATION_TARGET_MISSING",
            "TEXT LOB validator found no boundary-profile TEXT fields",
            collection=collection,
            schema=spec.name,
        )
        return
    for field in text_fields:
        pks = _boundary_pks(data_min_pk, data_max_pk)
        if len(pks) < 7:
            report.fail(
                "TEXT_LOB_RANGE_TOO_SMALL",
                "checkpoint range does not include every TEXT LOB boundary slot",
                collection=collection,
                field=field.name,
                boundary_pks=pks,
            )
            continue
        for data_pk in pks:
            rows = _query_by_data_pk(
                client, collection, spec, meta, data_pk, [field.name]
            )
            actual = rows[0].get(field.name) if rows else None
            expected = generate_field_value(field, data_pk, seed)
            actual_meta = text_payload_metadata(actual)
            expected_meta = text_payload_metadata(expected)
            report.metrics[f"{collection}.{field.name}.pk_{data_pk}.payload"] = (
                actual_meta
            )
            if actual_meta != expected_meta:
                report.fail(
                    "TEXT_LOB_MISMATCH",
                    "TEXT payload metadata or hash changed across lifecycle phase",
                    collection=collection,
                    field=field.name,
                    data_pk=data_pk,
                    expected=expected_meta,
                    actual=actual_meta,
                )
                continue
            _record_pass(report, collection, "text_lob_round_trip")


def validate_text_match_phrase_match(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    del meta, seed
    primary = _primary_field(spec)
    text_fields = [field for field in spec.fields if field.dtype == "TEXT"]
    bm25_outputs = [
        output
        for output in function_output_fields(spec)
        if any(
            index.field == output and index.metric_type == "BM25"
            for index in spec.indexes
        )
    ]
    if not text_fields and not bm25_outputs:
        report.fail(
            "FEATURE_VALIDATION_TARGET_MISSING",
            "TEXT validator found no TEXT fields or BM25 outputs",
            collection=collection,
            schema=spec.name,
        )
        return
    for field in text_fields:
        filters = [
            f"TEXT_MATCH({field.name}, 'milvus')",
            f"PHRASE_MATCH({field.name}, 'milvus upgrade', 1)",
        ]
        for filter_expr in filters:
            rows = client.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=[primary.name],
                limit=10,
            )
            if not rows:
                report.fail(
                    "TEXT_FILTER_FAILED",
                    "TEXT_MATCH or PHRASE_MATCH returned no rows",
                    collection=collection,
                    field=field.name,
                    filter=filter_expr,
                )
                continue
            _record_pass(report, collection, "text_match_phrase_match")
    for output in bm25_outputs:
        response = client.search(
            collection_name=collection,
            data=["milvus upgrade rollback"],
            anns_field=output,
            limit=10,
            search_params={
                "metric_type": "BM25",
                "params": search_params_for_field(spec, output),
            },
        )
        if not _search_hits(response):
            report.fail(
                "TEXT_BM25_FAILED",
                "BM25 search over TEXT returned no rows",
                collection=collection,
                field=output,
            )
            continue
        _record_pass(report, collection, "text_match_phrase_match")


def validate_minhash_search(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    metric_prefix = f"{collection}.minhash_search"
    report.metrics[f"{metric_prefix}.coverage_mode"] = (
        "exact_self_search_with_observational_near_duplicate"
    )
    report.metrics[f"{metric_prefix}.exact_self_search_enforced"] = True
    report.metrics[f"{metric_prefix}.near_duplicate_gate_enforced"] = False
    report.metrics[f"{metric_prefix}.ranking_gate_mode"] = (
        "conditional_when_both_observational_hits_returned"
    )
    primary = _primary_field(spec)
    data_min_pk, data_max_pk = _data_pk_range(meta)
    start = data_min_pk + ((0 - data_min_pk) % 3)
    if start + 2 > data_max_pk:
        report.fail(
            "MINHASH_RANGE_TOO_SMALL",
            "checkpoint range lacks exact, near-duplicate, and unrelated MinHash rows",
            collection=collection,
        )
        return
    data_pks = [start, start + 1, start + 2]
    actual_pks = [_actual_primary_value(spec, meta, pk) for pk in data_pks]
    document = next(
        field for field in spec.fields if field.value_profile == "minhash_documents"
    )
    output = next(
        output
        for output in function_output_fields(spec)
        if resolve_field(spec, output).dtype == "BINARY_VECTOR"
    )
    response = client.search(
        collection_name=collection,
        data=[generate_field_value(document, start, seed)],
        anns_field=output,
        filter=(
            f"{primary.name} in ["
            + ", ".join(format_filter_value(pk) for pk in actual_pks)
            + "]"
        ),
        output_fields=[primary.name, document.name],
        limit=3,
        search_params={
            "metric_type": "MHJACCARD",
            "params": search_params_for_field(spec, output),
        },
    )
    hits = _search_hits(response)
    hit_pks = [_hit_pk(hit, primary.name) for hit in hits]
    if actual_pks[0] not in hit_pks:
        report.fail(
            "MINHASH_SEARCH_FAILED",
            "MinHash search missed the exact document",
            collection=collection,
            expected_exact=actual_pks[0],
            actual_pks=hit_pks,
        )
        return
    report.metrics[f"{metric_prefix}.near_duplicate_returned"] = int(
        actual_pks[1] in hit_pks
    )
    report.metrics[f"{metric_prefix}.unrelated_returned"] = int(
        actual_pks[2] in hit_pks
    )
    rank = {pk: hit_pks.index(pk) for pk in actual_pks if pk in hit_pks}
    if (
        actual_pks[1] in rank
        and actual_pks[2] in rank
        and rank[actual_pks[1]] > rank[actual_pks[2]]
    ):
        report.fail(
            "MINHASH_SEARCH_FAILED",
            "near-duplicate document ranked below the unrelated document",
            collection=collection,
            related_pk=actual_pks[1],
            unrelated_pk=actual_pks[2],
            actual_pks=hit_pks,
        )
        return
    _record_pass(report, collection, "minhash_search")


def validate_entity_ttl(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
) -> None:
    primary = _primary_field(spec)
    ttl_field_name = str(spec.properties.get("ttl_field") or "")
    ttl_field = resolve_field(spec, ttl_field_name)
    if ttl_field is None or ttl_field.dtype != "TIMESTAMPTZ":
        report.fail(
            "ENTITY_TTL_INVALID",
            "entity TTL validator requires a TIMESTAMPTZ ttl_field collection property",
            collection=collection,
            ttl_field=ttl_field_name,
        )
        return
    _, data_max_pk = _data_pk_range(meta)
    temp_pks = [
        data_max_pk + 10_000_001,
        data_max_pk + 10_000_002,
        data_max_pk + 10_000_003,
    ]
    function_outputs = function_output_fields(spec)
    rows = []
    ttl_values = ["2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z", None]
    for pk, ttl_value in zip(temp_pks, ttl_values):
        row = {}
        for field in spec.fields:
            if field.name in function_outputs or (field.primary and field.auto_id):
                continue
            if field.primary:
                row[field.name] = generate_primary_key_value(field, pk)
            elif field.name == ttl_field.name:
                row[field.name] = ttl_value
            else:
                row[field.name] = generate_field_value(field, pk, seed)
        rows.append(row)
    client.insert(collection_name=collection, data=rows)
    try:
        client.flush(collection_name=collection)
    except TypeError:
        client.flush(collection)
    expected_visible = {
        generate_primary_key_value(primary, temp_pks[1]),
        generate_primary_key_value(primary, temp_pks[2]),
    }
    expired_pk = generate_primary_key_value(primary, temp_pks[0])
    deadline = monotonic() + 60
    visible: set[Any] = set()
    filter_expr = (
        f"{primary.name} in ["
        + ", ".join(
            format_filter_value(generate_primary_key_value(primary, pk))
            for pk in temp_pks
        )
        + "]"
    )
    while monotonic() < deadline:
        queried = client.query(
            collection_name=collection,
            filter=filter_expr,
            output_fields=[primary.name, ttl_field.name],
            limit=3,
        )
        visible = {row.get(primary.name) for row in queried}
        if expired_pk not in visible and expected_visible <= visible:
            break
        sleep(2)
    if expired_pk in visible or not expected_visible <= visible:
        report.fail(
            "ENTITY_TTL_FAILED",
            "expired/future/null TTL visibility did not match the contract",
            collection=collection,
            expired_pk=expired_pk,
            expected_visible=sorted(expected_visible, key=str),
            actual_visible=sorted(visible, key=str),
            checked_at=datetime.now(timezone.utc).isoformat(),
        )
    else:
        _record_pass(report, collection, "entity_ttl")
    try:
        client.delete(collection_name=collection, filter=filter_expr)
    except Exception:
        pass


def _nested_config(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def validate_index_engine_version(
    collection: str,
    spec: SchemaSpec,
    runtime_config: dict[str, Any] | None,
    report: ValidationReport,
) -> None:
    runtime_config = runtime_config or {}
    expected_vec = spec.validator_params.get("target_vec_index_version")
    expected_scalar = spec.validator_params.get("target_scalar_index_version")
    actual_vec = _nested_config(runtime_config, "dataCoord", "targetVecIndexVersion")
    actual_scalar = _nested_config(
        runtime_config, "dataCoord", "targetScalarIndexVersion"
    )
    pairs = [
        ("targetVecIndexVersion", expected_vec, actual_vec),
        ("targetScalarIndexVersion", expected_scalar, actual_scalar),
    ]
    for name, expected, actual in pairs:
        if expected is None:
            report.fail(
                "INDEX_ENGINE_VERSION_INVALID",
                "schema validator_params omit an expected index engine version",
                collection=collection,
                config=name,
            )
            continue
        try:
            matches = int(actual) == int(expected)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            report.fail(
                "INDEX_ENGINE_VERSION_MISMATCH",
                "runtime Milvus target index engine configuration differs from the promoted matrix",
                collection=collection,
                config=name,
                expected=expected,
                actual=actual,
            )
            continue
        _record_pass(report, collection, "index_engine_version")


def run_feature_validator(
    validator: str,
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
    runtime_config: dict[str, Any] | None = None,
) -> None:
    handlers = {
        "nullable_vector_semantics": validate_nullable_vector_semantics,
        "struct_array_scalar_round_trip": validate_struct_array_scalar_round_trip,
        "struct_array_element_search": validate_struct_array_element_search,
        "struct_array_scalar_index_queries": validate_struct_array_scalar_index_queries,
        "geometry_filter": validate_geometry_filter,
        "text_lob_round_trip": validate_text_lob_round_trip,
        "text_match_phrase_match": validate_text_match_phrase_match,
        "minhash_search": validate_minhash_search,
        "entity_ttl": validate_entity_ttl,
    }
    if validator == "index_engine_version":
        validate_index_engine_version(collection, spec, runtime_config, report)
        return
    handler = handlers.get(validator)
    if handler is None:
        report.fail(
            "UNKNOWN_SCHEMA_VALIDATOR",
            "schema matrix declares an unsupported feature validator",
            collection=collection,
            schema=spec.name,
            validator=validator,
        )
        return
    handler(client, collection, spec, meta, seed, report)
