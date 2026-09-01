from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json
import math
import sys

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.client import create_client, get_server_version
from milvus_client.common.data import (
    generate_field_value,
    generate_primary_key_value,
    generate_struct_array_value,
    prepare_struct_vector_query,
)
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.pressure_maintenance import record_maintenance_window
from milvus_client.common.schema import (
    FieldSpec,
    IndexSpec,
    SchemaSpec,
    VECTOR_TYPES,
    auto_id_enabled,
    build_index_params,
    function_output_fields,
    load_schema_matrix,
    resolve_field,
    struct_array_for_field,
)
from milvus_client.common.validators import (
    ValidationReport,
    format_filter_value,
    pk_range_filter,
    validate_collection_count,
    validate_pk_samples,
)
from milvus_client.common.workload import (
    approximate_recall_index,
    assert_search_result,
    metric_type_for_field,
    search_params_for_field,
)
from milvus_client.common.version import (
    diskann_max_sim_cached_distance_bug,
    server_version_for_feature_detection,
    version_at_least,
)


INDEX_SEARCH_FAILED = "INDEX_SEARCH_FAILED"
INDEX_SCALAR_QUERY_FAILED = "INDEX_SCALAR_QUERY_FAILED"
INDEX_REBUILD_FAILED = "INDEX_REBUILD_FAILED"
INDEX_METADATA_MISMATCH = "INDEX_METADATA_MISMATCH"
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


def _indexed_fields(spec: SchemaSpec) -> list[str]:
    return list(dict.fromkeys(index.field for index in spec.indexes))


def _indexed_vector_indexes(spec: SchemaSpec) -> list[tuple[IndexSpec, FieldSpec]]:
    return [
        (index, field)
        for index in spec.indexes
        if (field := resolve_field(spec, index.field)) is not None
        and field.dtype in VECTOR_TYPES
    ]


def _indexed_vector_fields(spec: SchemaSpec) -> list[FieldSpec]:
    return [field for _, field in _indexed_vector_indexes(spec)]


def indexed_scalar_indexes(spec: SchemaSpec) -> list[tuple[IndexSpec, FieldSpec]]:
    return [
        (index, field)
        for index in spec.indexes
        if (field := resolve_field(spec, index.field)) is not None
        and field.dtype not in VECTOR_TYPES
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


def _release_collection(client: Any, collection: str, timeout_sec: int) -> None:
    release = getattr(client, "release_collection", None)
    if release is None:
        raise RuntimeError("Milvus client does not expose release_collection")
    try:
        _call_with_optional_timeout(
            release,
            collection_name=collection,
            timeout_sec=timeout_sec,
        )
    except TypeError:
        _call_with_optional_timeout(release, collection, timeout_sec=timeout_sec)


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
        raise RuntimeError("Milvus client does not expose list_indexes")
    try:
        names = list_indexes(collection_name=collection, field_name=field_name)
    except TypeError:
        names = list_indexes(collection, field_name)
    return list(names or [])


def _describe_index(
    client: Any,
    collection: str,
    field_name: str,
    index_name: str,
) -> dict[str, Any]:
    describe_index = getattr(client, "describe_index", None)
    if describe_index is None:
        raise RuntimeError("Milvus client does not expose describe_index")
    try:
        payload = describe_index(collection_name=collection, index_name=index_name)
    except TypeError:
        try:
            payload = describe_index(collection, index_name)
        except TypeError:
            payload = describe_index(collection_name=collection, field_name=field_name)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        payload = {}
    index_param = payload.get("index_param") or payload.get("indexParam") or {}
    raw_params = payload.get("params") or payload.get("index_params") or {}
    params = dict(raw_params) if isinstance(raw_params, dict) else {}
    if not params and isinstance(index_param, dict):
        index_params = index_param.get("params") or {}
        if isinstance(index_params, dict):
            params.update(index_params)
    for key in (
        "json_path",
        "json_cast_type",
        "faiss_index_name",
        "inverted_index_algo",
        "mh_lsh_band",
        "with_raw_data",
        "sq_type",
        "refine",
        "refine_type",
        "resolved_index_type",
        "min_gram",
        "max_gram",
    ):
        if payload.get(key) is not None:
            params[key] = payload[key]
        if isinstance(index_param, dict) and index_param.get(key) is not None:
            params[key] = index_param[key]
    metadata = {
        "index_name": str(
            payload.get("index_name")
            or payload.get("indexName")
            or payload.get("index")
            or index_name
        ),
        "field_name": str(
            payload.get("field_name")
            or payload.get("fieldName")
            or payload.get("field")
            or field_name
        ),
        "index_type": (
            payload.get("index_type")
            or payload.get("indexType")
            or (
                index_param.get("index_type") if isinstance(index_param, dict) else None
            )
        ),
        "metric_type": (
            payload.get("metric_type")
            or payload.get("metricType")
            or (
                index_param.get("metric_type")
                if isinstance(index_param, dict)
                else None
            )
        ),
        "params": params or {},
    }
    if (
        not metadata["index_name"]
        or not metadata["field_name"]
        or not metadata["index_type"]
    ):
        raise RuntimeError(
            f"incomplete index metadata for {collection}.{field_name}/{index_name}: {payload}"
        )
    return metadata


def _actual_index_metadata(
    client: Any,
    collection: str,
    spec: SchemaSpec,
) -> list[dict[str, Any]]:
    indexes = []
    for field_name in _indexed_fields(spec):
        for index_name in _index_names_for_field(client, collection, field_name):
            indexes.append(_describe_index(client, collection, field_name, index_name))
    return sorted(
        indexes,
        key=lambda item: (
            str(item.get("field_name")),
            str(item.get("index_name")),
            str(item.get("index_type")),
        ),
    )


def _index_identity(index: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "index_name": index.get("index_name"),
        "field_name": index.get("field_name"),
        "index_type": index.get("index_type"),
        "metric_type": index.get("metric_type"),
    }
    params = index.get("params") or {}
    compatibility_params = {
        key: params[key]
        for key in (
            "json_path",
            "json_cast_type",
            "faiss_index_name",
            "inverted_index_algo",
            "mh_lsh_band",
            "with_raw_data",
            "sq_type",
            "refine",
            "refine_type",
            "resolved_index_type",
            "min_gram",
            "max_gram",
        )
        if key in params
    }
    if compatibility_params:
        identity["compatibility_params"] = compatibility_params
    return identity


def _validate_resolved_index_types(
    collection: str,
    spec: SchemaSpec,
    actual_indexes: list[dict[str, Any]],
    report: ValidationReport,
) -> None:
    actual_by_field: dict[str, tuple[str | None, str]] = {}
    for actual_index in actual_indexes:
        params = actual_index.get("params") or {}
        actual_type = actual_index.get("index_type")
        resolved = params.get("resolved_index_type")
        source = "params.resolved_index_type"
        if resolved is None:
            param_type = params.get("index_type")
            if param_type is not None and str(param_type) != str(actual_type):
                resolved = param_type
                source = "params.index_type"
            elif actual_type is not None and str(actual_type) != "AUTOINDEX":
                resolved = actual_type
                source = "top_level.index_type"
            else:
                source = "public_sdk_unavailable"
        actual_by_field[str(actual_index.get("field_name"))] = (
            str(resolved) if resolved is not None else None,
            source,
        )
    for index in spec.indexes:
        expected = index.expected_resolved_index_type
        if not expected:
            continue
        actual, source = actual_by_field.get(
            index.field, (None, "index_metadata_unavailable")
        )
        metric_prefix = f"{collection}.{index.field}.resolved_index_type"
        report.metrics[f"{metric_prefix}.expected"] = expected
        report.metrics[f"{metric_prefix}.observed"] = actual or "unavailable"
        report.metrics[f"{metric_prefix}.source"] = source
        if actual is None:
            report.metrics["resolved_index_types_unobservable_total"] = (
                int(report.metrics.get("resolved_index_types_unobservable_total", 0))
                + 1
            )
            if (
                source == "public_sdk_unavailable"
                and str(index.index_type) == "AUTOINDEX"
            ):
                report.metrics[f"{metric_prefix}.validation"] = (
                    "not_observable_via_public_sdk"
                )
                continue
            report.fail(
                INDEX_METADATA_MISMATCH,
                "resolved index type is required but unavailable",
                collection=collection,
                field=index.field,
                expected_resolved_index_type=expected,
                actual_index_type=None,
                resolved_index_type_source=source,
            )
            continue
        if actual != expected:
            report.fail(
                INDEX_METADATA_MISMATCH,
                "resolved index type differs from schema matrix expectation",
                collection=collection,
                field=index.field,
                expected_resolved_index_type=expected,
                actual_index_type=actual,
                resolved_index_type_source=source,
            )


def _metadata_value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str):
            return actual.lower() == str(expected).lower()
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _validate_index_metadata_matches_spec(
    collection: str,
    spec: SchemaSpec,
    actual_indexes: list[dict[str, Any]],
    report: ValidationReport,
) -> None:
    actual_by_field = {str(index.get("field_name")): index for index in actual_indexes}
    compatibility_keys = {
        "json_path",
        "json_cast_type",
        "faiss_index_name",
        "inverted_index_algo",
        "mh_lsh_band",
        "with_raw_data",
        "sq_type",
        "refine",
        "refine_type",
        "min_gram",
        "max_gram",
    }
    for expected in spec.indexes:
        actual = actual_by_field.get(expected.field)
        if actual is None:
            continue
        actual_type = actual.get("index_type")
        accepted_types = {expected.index_type}
        if expected.expected_resolved_index_type:
            accepted_types.add(expected.expected_resolved_index_type)
        if str(actual_type) not in accepted_types:
            report.fail(
                INDEX_METADATA_MISMATCH,
                "actual index type differs from the schema matrix",
                collection=collection,
                field=expected.field,
                expected_index_types=sorted(accepted_types),
                actual_index_type=actual_type,
            )
        actual_metric = actual.get("metric_type")
        if (
            expected.metric_type is not None
            and str(actual_metric) != expected.metric_type
        ):
            report.fail(
                INDEX_METADATA_MISMATCH,
                "actual index metric differs from the schema matrix",
                collection=collection,
                field=expected.field,
                expected_metric_type=expected.metric_type,
                actual_metric_type=actual_metric,
            )
        if expected.index_name and actual.get("index_name") != expected.index_name:
            report.fail(
                INDEX_METADATA_MISMATCH,
                "actual index name differs from the schema matrix",
                collection=collection,
                field=expected.field,
                expected_index_name=expected.index_name,
                actual_index_name=actual.get("index_name"),
            )
        actual_params = actual.get("params") or {}
        for key in compatibility_keys & set(expected.params):
            expected_value = expected.params[key]
            actual_value = actual_params.get(key)
            if not _metadata_value_matches(expected_value, actual_value):
                report.fail(
                    INDEX_METADATA_MISMATCH,
                    "actual index parameter differs from the schema matrix",
                    collection=collection,
                    field=expected.field,
                    parameter=key,
                    expected_value=expected_value,
                    actual_value=actual_value,
                )


def _validate_index_metadata_matches_checkpoint(
    collection: str,
    expected_indexes: list[dict[str, Any]],
    actual_indexes: list[dict[str, Any]],
    report: ValidationReport,
) -> None:
    expected = sorted(
        [_index_identity(index) for index in expected_indexes],
        key=lambda item: (
            str(item.get("field_name")),
            str(item.get("index_name")),
            str(item.get("index_type")),
        ),
    )
    actual = sorted(
        [_index_identity(index) for index in actual_indexes],
        key=lambda item: (
            str(item.get("field_name")),
            str(item.get("index_name")),
            str(item.get("index_type")),
        ),
    )
    if actual != expected:
        report.fail(
            INDEX_METADATA_MISMATCH,
            "actual index metadata differs from after-upgrade checkpoint",
            collection=collection,
            expected=expected,
            actual=actual,
        )


def _validate_expected_index_fields_present(
    collection: str,
    expected_fields: list[str],
    actual_indexes: list[dict[str, Any]],
    report: ValidationReport,
) -> None:
    actual_fields = {str(index.get("field_name")) for index in actual_indexes}
    missing = [field for field in expected_fields if field not in actual_fields]
    if missing:
        report.fail(
            INDEX_METADATA_MISMATCH,
            "expected indexed fields are missing from actual index metadata",
            collection=collection,
            missing_fields=missing,
            actual_indexes=[_index_identity(index) for index in actual_indexes],
        )


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


def _expected_primary_value(
    spec: SchemaSpec,
    meta: dict[str, Any],
    data_pk_number: int,
) -> Any:
    primary = _primary_field(spec)
    if auto_id_enabled(spec):
        pk_values = meta.get("pk_values") or meta.get("pk_samples") or []
        if pk_values:
            data_min_pk = int(meta.get("data_min_pk", 0))
            offset = data_pk_number - data_min_pk
            if 0 <= offset < len(pk_values):
                return pk_values[offset]
            return pk_values[0]
        return meta.get("min_pk")
    if primary is not None:
        return generate_primary_key_value(primary, data_pk_number)
    return data_pk_number


def _data_pk_range(meta: dict[str, Any]) -> tuple[int, int]:
    min_pk = int(meta.get("data_min_pk", meta["min_pk"]))
    max_pk = int(meta.get("data_max_pk", meta["max_pk"]))
    return min_pk, max_pk


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


def _hit_primary_key(hit: Any, primary_name: str) -> Any:
    for key in ("id", "pk", primary_name):
        value = _hit_value(hit, key)
        if value is not None:
            return value
    entity = _hit_value(hit, "entity")
    if isinstance(entity, dict):
        return entity.get(primary_name)
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


def _hit_offset(hit: Any) -> int | None:
    value = _hit_value(hit, "offset")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_vector_search_hit(
    response: Any,
    collection: str,
    field_name: str,
    primary_name: str,
    expected_pk: Any,
    expected_offset: int | None,
    metric_type: str,
    report: ValidationReport,
    index_type: str = "",
    lossy_index: bool = False,
    diskann_max_sim_bug: bool = False,
) -> None:
    assert_search_result(response, collection, field_name)
    hits = response[0]
    actual_hits = [
        {
            "pk": _hit_primary_key(hit, primary_name),
            "offset": _hit_offset(hit),
            "distance": _hit_distance(hit),
        }
        for hit in hits
    ]
    expected_hit = None
    hit_pks = []
    for hit in hits:
        hit_pk = _hit_primary_key(hit, primary_name)
        hit_pks.append(hit_pk)
        hit_offset = _hit_offset(hit)
        if hit_pk == expected_pk and (
            expected_offset is None or hit_offset == expected_offset
        ):
            expected_hit = hit
            break
    if expected_hit is None:
        report.fail(
            INDEX_SEARCH_FAILED,
            "indexed vector search did not return expected primary key",
            collection=collection,
            field=field_name,
            expected_pk=expected_pk,
            expected_offset=expected_offset,
            actual_pks=hit_pks,
            actual_offsets=[_hit_offset(hit) for hit in hits],
            actual_hits=actual_hits,
        )
        return

    distance = _hit_distance(expected_hit)
    if distance is None:
        report.fail(
            INDEX_SEARCH_FAILED,
            "indexed vector self-search did not expose a distance or score",
            collection=collection,
            field=field_name,
            metric_type=metric_type,
            index_type=index_type,
            expected_pk=expected_pk,
            expected_offset=expected_offset,
            actual_hits=actual_hits,
        )
        return
    if not math.isfinite(distance):
        report.fail(
            INDEX_SEARCH_FAILED,
            "indexed vector self-search returned a non-finite distance or score",
            collection=collection,
            field=field_name,
            metric_type=metric_type,
            index_type=index_type,
            expected_pk=expected_pk,
            distance=distance,
            actual_hits=actual_hits,
        )
        return
    metric = metric_type.upper().removeprefix("MAX_SIM_")
    max_distance = 0.5 if lossy_index and metric == "L2" else 1e-3
    min_score = 0.5 if lossy_index and metric in {"COSINE", "IP"} else 0.9
    if metric in {"L2", "HAMMING", "JACCARD"} and distance < 0:
        report.fail(
            INDEX_SEARCH_FAILED,
            "indexed vector self-search distance is lower than the metric minimum",
            collection=collection,
            field=field_name,
            metric_type=metric_type,
            index_type=index_type,
            expected_pk=expected_pk,
            distance=distance,
            min_distance=0.0,
            actual_hits=actual_hits,
        )
        return
    if metric in {"COSINE", "MHJACCARD"} and distance > 1.001:
        report.fail(
            INDEX_SEARCH_FAILED,
            "indexed vector self-search score is higher than the metric maximum",
            collection=collection,
            field=field_name,
            metric_type=metric_type,
            index_type=index_type,
            expected_pk=expected_pk,
            distance=distance,
            max_score=1.001,
            actual_hits=actual_hits,
        )
        return
    if metric in {"L2", "HAMMING", "JACCARD"} and distance > max_distance:
        report.fail(
            INDEX_SEARCH_FAILED,
            "indexed vector self-search distance is higher than expected",
            collection=collection,
            field=field_name,
            metric_type=metric_type,
            index_type=index_type,
            expected_pk=expected_pk,
            distance=distance,
            max_distance=max_distance,
            actual_hits=actual_hits,
        )
    if metric in {"COSINE", "IP"} and distance < min_score:
        if (
            diskann_max_sim_bug
            and index_type.upper() == "DISKANN"
            and metric_type.upper().startswith("MAX_SIM_")
            and distance < 0
        ):
            report.metrics["diskann_max_sim_negative_score_known"] = True
        else:
            report.fail(
                INDEX_SEARCH_FAILED,
                "indexed vector self-search score is lower than expected",
                collection=collection,
                field=field_name,
                metric_type=metric_type,
                index_type=index_type,
                expected_pk=expected_pk,
                distance=distance,
                min_score=min_score,
                actual_hits=actual_hits,
            )


def _vector_index_probe(
    spec: SchemaSpec,
    meta: dict[str, Any],
    index: IndexSpec,
    vector_field: FieldSpec,
    seed: int,
) -> tuple[int, Any, Any, int | None] | None:
    function_outputs = function_output_fields(spec)
    data_min_pk, data_max_pk = _data_pk_range(meta)
    for data_pk_number in range(data_min_pk, data_max_pk + 1):
        expected_pk = _expected_primary_value(spec, meta, data_pk_number)
        if vector_field.name in function_outputs:
            function = next(
                item
                for item in spec.functions
                if vector_field.name in item.output_fields
            )
            input_field = resolve_field(spec, function.input_fields[0])
            if input_field is None:
                return None
            query = generate_field_value(input_field, data_pk_number, seed)
            if query is None or query == "":
                continue
            return (
                data_pk_number,
                expected_pk,
                query,
                None,
            )
        struct_array = struct_array_for_field(spec, index.field)
        if struct_array is not None:
            value = generate_struct_array_value(struct_array, data_pk_number, seed)
            if value is None:
                continue
            for offset, element in enumerate(value):
                vector = element.get(vector_field.name)
                if vector is not None:
                    query, expected_offset = prepare_struct_vector_query(
                        index.metric_type or "COSINE", vector, offset
                    )
                    return data_pk_number, expected_pk, query, expected_offset
            continue
        value = generate_field_value(vector_field, data_pk_number, seed)
        if value is not None:
            return data_pk_number, expected_pk, value, None
    return None


def _validate_index_searches(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
    diskann_max_sim_bug: bool = False,
) -> int:
    searches = 0
    primary = _primary_field(spec)
    primary_name = meta.get("primary_field") or (
        primary.name if primary is not None else "id"
    )
    for index, vector_field in _indexed_vector_indexes(spec):
        metric_type = metric_type_for_field(spec, index.field)
        probe = _vector_index_probe(spec, meta, index, vector_field, seed)
        if probe is None:
            report.fail(
                INDEX_SEARCH_FAILED,
                "could not build deterministic vector index probe",
                collection=collection,
                field=index.field,
            )
            continue
        data_pk_number, expected_pk, query_vector, expected_offset = probe
        filter_expr = f"{primary_name} == {format_filter_value(expected_pk)}"
        try:
            response = client.search(
                collection_name=collection,
                data=[query_vector],
                anns_field=index.field,
                filter=filter_expr,
                limit=5,
                search_params={
                    "metric_type": metric_type,
                    "params": search_params_for_field(spec, index.field),
                },
            )
            _validate_vector_search_hit(
                response,
                collection,
                index.field,
                primary_name,
                expected_pk,
                expected_offset,
                metric_type,
                report,
                index_type=index.index_type,
                lossy_index=approximate_recall_index(spec, index.field),
                diskann_max_sim_bug=diskann_max_sim_bug,
            )
            searches += 1
        except Exception as exc:
            report.fail(
                INDEX_SEARCH_FAILED,
                "indexed vector search failed",
                collection=collection,
                field=index.field,
                metric_type=metric_type,
                data_pk=data_pk_number,
                expected_pk=expected_pk,
                filter=filter_expr,
                error=str(exc),
            )
    return searches


def _json_path_keys(json_path: str, field_name: str) -> list[str]:
    node = ast.parse(json_path, mode="eval").body
    keys = []
    while isinstance(node, ast.Subscript):
        key_node = node.slice
        if not isinstance(key_node, ast.Constant) or not isinstance(
            key_node.value, str
        ):
            raise ValueError(f"unsupported JSON path component: {json_path}")
        keys.append(key_node.value)
        node = node.value
    if not isinstance(node, ast.Name) or node.id != field_name:
        raise ValueError(
            f"JSON path {json_path!r} does not start with field {field_name!r}"
        )
    return list(reversed(keys))


def _json_path_value(value: Any, keys: list[str]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _format_timestamptz_filter_value(value: Any) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"ISO '{escaped}'"


def scalar_index_filter_for_value(
    spec: SchemaSpec,
    index: IndexSpec,
    field: FieldSpec,
    value: Any,
) -> str | None:
    struct_array = struct_array_for_field(spec, index.field)
    if struct_array is not None:
        if not value:
            return None
        nested_value = value[0].get(field.name)
        operator = (
            ">="
            if index.index_type == "STL_SORT"
            and field.dtype in {"INT8", "INT16", "INT32", "INT64", "FLOAT", "DOUBLE"}
            else "=="
        )
        return (
            f"MATCH_ANY({struct_array.name}, $[{field.name}] {operator} "
            f"{format_filter_value(nested_value)})"
        )
    if field.dtype == "JSON":
        json_path = str(index.params.get("json_path") or f"{field.name}['bucket']")
        path_value = _json_path_value(value, _json_path_keys(json_path, field.name))
        if path_value is not None:
            if index.index_type == "NGRAM" and isinstance(path_value, str):
                return f"{json_path} LIKE {format_filter_value('%' + path_value + '%')}"
            return f"{json_path} == {format_filter_value(path_value)}"
        return None
    if field.dtype == "ARRAY":
        if isinstance(value, list) and value:
            return f"ARRAY_CONTAINS({field.name}, {format_filter_value(value[0])})"
        return None
    if field.dtype == "GEOMETRY":
        if isinstance(value, str) and value:
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            return f"ST_EQUALS({field.name}, '{escaped}')"
        return None
    if value is None:
        return f"{field.name} is null"
    if field.dtype == "TIMESTAMPTZ":
        return f"{field.name} == {_format_timestamptz_filter_value(value)}"
    if index.index_type == "NGRAM" and field.dtype in {"VARCHAR", "STRING", "TEXT"}:
        return f"{field.name} LIKE {format_filter_value('%' + str(value) + '%')}"
    return f"{field.name} == {format_filter_value(value)}"


def _scalar_index_filter(
    spec: SchemaSpec, index: IndexSpec, field: FieldSpec, pk: int, seed: int
) -> str | None:
    struct_array = struct_array_for_field(spec, index.field)
    value = (
        generate_struct_array_value(struct_array, pk, seed)
        if struct_array is not None
        else generate_field_value(field, pk, seed)
    )
    return scalar_index_filter_for_value(spec, index, field, value)


def _scalar_index_probe(
    spec: SchemaSpec,
    meta: dict[str, Any],
    index: IndexSpec,
    field: FieldSpec,
    seed: int,
) -> tuple[int, Any, str] | None:
    data_min_pk, data_max_pk = _data_pk_range(meta)
    null_fallback: tuple[int, Any, str] | None = None
    for data_pk_number in range(data_min_pk, data_max_pk + 1):
        filter_expr = _scalar_index_filter(spec, index, field, data_pk_number, seed)
        if not filter_expr:
            continue
        expected_pk = _expected_primary_value(spec, meta, data_pk_number)
        probe = (data_pk_number, expected_pk, filter_expr)
        if not filter_expr.endswith(" is null"):
            return probe
        if null_fallback is None:
            null_fallback = probe
    return null_fallback


def validate_scalar_index_queries(
    client: Any,
    collection: str,
    spec: SchemaSpec,
    meta: dict[str, Any],
    seed: int,
    report: ValidationReport,
    probe_overrides: dict[str, tuple[int, Any, str]] | None = None,
    server_version: str | None = None,
) -> int:
    primary = _primary_field(spec)
    primary_name = meta.get("primary_field") or (
        primary.name if primary is not None else "id"
    )
    queries = 0
    scalar_indexes = indexed_scalar_indexes(spec)
    nested_scalar_indexes = [
        (index, field)
        for index, field in scalar_indexes
        if struct_array_for_field(spec, index.field) is not None
    ]
    nested_query_supported = True
    if nested_scalar_indexes:
        actual_server_version = server_version or get_server_version(client)
        try:
            nested_query_supported = version_at_least(actual_server_version, "3.0.0")
        except ValueError:
            report.fail(
                INDEX_SCALAR_QUERY_FAILED,
                "cannot determine whether the runtime supports StructArray scalar filters",
                collection=collection,
                server_version=actual_server_version,
            )
            nested_query_supported = False
        report.metrics[f"{collection}.struct_array_scalar_index_queries.supported"] = (
            nested_query_supported
        )
        if not nested_query_supported:
            report.metrics[
                f"{collection}.struct_array_scalar_index_queries.skipped_unsupported_total"
            ] = len(nested_scalar_indexes)
    for index, field in scalar_indexes:
        if (
            struct_array_for_field(spec, index.field) is not None
            and not nested_query_supported
        ):
            continue
        probe = (probe_overrides or {}).get(index.field) or _scalar_index_probe(
            spec, meta, index, field, seed
        )
        if probe is None:
            report.fail(
                INDEX_SCALAR_QUERY_FAILED,
                "could not build deterministic scalar index probe",
                collection=collection,
                field=index.field,
            )
            continue
        data_pk_number, expected_pk, scalar_filter_expr = probe
        filter_expr = (
            f"({scalar_filter_expr}) && "
            f"{primary_name} == {format_filter_value(expected_pk)}"
        )
        try:
            scalar_rows = client.query(
                collection_name=collection,
                filter=scalar_filter_expr,
                output_fields=[primary_name],
                limit=1,
            )
            if not scalar_rows:
                report.fail(
                    INDEX_SCALAR_QUERY_FAILED,
                    "indexed scalar filter query returned no matches",
                    collection=collection,
                    field=index.field,
                    filter=scalar_filter_expr,
                    data_pk=data_pk_number,
                    expected_pk=expected_pk,
                )
            rows = client.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=[primary_name],
                limit=1,
            )
            actual_pks = [row.get(primary_name) for row in rows]
            if expected_pk not in actual_pks:
                report.fail(
                    INDEX_SCALAR_QUERY_FAILED,
                    "indexed scalar filter query did not return expected primary key",
                    collection=collection,
                    field=index.field,
                    filter=filter_expr,
                    scalar_filter=scalar_filter_expr,
                    data_pk=data_pk_number,
                    expected_pk=expected_pk,
                    actual_pks=actual_pks,
                )
            queries += 1
        except Exception as exc:
            report.fail(
                INDEX_SCALAR_QUERY_FAILED,
                "indexed scalar filter query failed",
                collection=collection,
                field=index.field,
                filter=filter_expr,
                error=str(exc),
            )
    return queries


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
        actual_server_version = get_server_version(client)
        server_version = server_version_for_feature_detection(
            actual_server_version,
            args.server_version_hint,
            expected_image=args.expected_server_image,
            release_gate_eligible=args.release_gate_eligible,
        )
        result.capabilities = {
            "server_version": actual_server_version,
            "effective_server_version": server_version,
        }
        diskann_max_sim_bug = diskann_max_sim_cached_distance_bug(
            args.expected_server_image
        )
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
            "actual_indexes_total": 0,
            "searches_total": 0,
            "scalar_index_queries_total": 0,
            "reload_cycles_total": 0,
            "reload_searches_total": 0,
            "reload_scalar_index_queries_total": 0,
        }

        for collection, meta in _collection_items_for_phase(
            seed_checkpoint,
            index_checkpoint,
            args.phase,
        ).items():
            print(
                "index compatibility "
                f"phase={args.phase} collection={collection} "
                f"rebuild_index={args.rebuild_index}",
                flush=True,
            )
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
            collection_metric_prefix = f"{collection}."
            collection_metrics = {
                f"{collection_metric_prefix}actual_indexes_total": 0,
                f"{collection_metric_prefix}vector_searches_total": 0,
                f"{collection_metric_prefix}scalar_index_queries_total": 0,
                f"{collection_metric_prefix}reload_cycles_total": 0,
                f"{collection_metric_prefix}reload_vector_searches_total": 0,
                f"{collection_metric_prefix}reload_scalar_index_queries_total": 0,
                f"{collection_metric_prefix}declared_autoindexes_total": sum(
                    index.index_type == "AUTOINDEX" for index in spec.indexes
                ),
            }
            metrics.update(collection_metrics)
            if indexed_fields:
                metrics["collections_with_index"] += 1
            try:
                release_status = "not_requested"
                if args.rebuild_index:
                    _flush_collection(client, collection, args.timeout_sec)
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
                actual_indexes = _actual_index_metadata(client, collection, spec)
                metrics["actual_indexes_total"] += len(actual_indexes)
                metrics[f"{collection_metric_prefix}actual_indexes_total"] = len(
                    actual_indexes
                )
                _validate_expected_index_fields_present(
                    collection,
                    indexed_fields,
                    actual_indexes,
                    report,
                )
                _validate_index_metadata_matches_spec(
                    collection,
                    spec,
                    actual_indexes,
                    report,
                )
                _validate_resolved_index_types(
                    collection,
                    spec,
                    actual_indexes,
                    report,
                )
                if args.phase == "after-rollback":
                    _validate_index_metadata_matches_checkpoint(
                        collection,
                        index_checkpoint.get("collections", {})
                        .get(collection, {})
                        .get("actual_indexes", []),
                        actual_indexes,
                        report,
                    )
                _validate_query_serviceability(client, collection, spec, meta, report)
                vector_searches = _validate_index_searches(
                    client,
                    collection,
                    spec,
                    meta,
                    args.seed,
                    report,
                    diskann_max_sim_bug=diskann_max_sim_bug,
                )
                metrics["searches_total"] += vector_searches
                metrics[f"{collection_metric_prefix}vector_searches_total"] = (
                    vector_searches
                )
                scalar_index_queries = validate_scalar_index_queries(
                    client,
                    collection,
                    spec,
                    meta,
                    args.seed,
                    report,
                    server_version=server_version,
                )
                metrics["scalar_index_queries_total"] += scalar_index_queries
                metrics[f"{collection_metric_prefix}scalar_index_queries_total"] = (
                    scalar_index_queries
                )
                maintenance_windows = report.metrics.setdefault(
                    "maintenance_windows", []
                )
                with record_maintenance_window(
                    maintenance_windows,
                    label=f"index-compatibility-reload-{args.phase}",
                    source="validate_index_compatibility",
                    collection=collection,
                ):
                    _release_collection(client, collection, args.timeout_sec)
                    _load_collection(client, collection, args.timeout_sec)
                metrics["reload_cycles_total"] += 1
                metrics[f"{collection_metric_prefix}reload_cycles_total"] = 1
                _validate_query_serviceability(client, collection, spec, meta, report)
                reload_vector_searches = _validate_index_searches(
                    client,
                    collection,
                    spec,
                    meta,
                    args.seed,
                    report,
                    diskann_max_sim_bug=diskann_max_sim_bug,
                )
                metrics["reload_searches_total"] += reload_vector_searches
                metrics[f"{collection_metric_prefix}reload_vector_searches_total"] = (
                    reload_vector_searches
                )
                reload_scalar_index_queries = validate_scalar_index_queries(
                    client,
                    collection,
                    spec,
                    meta,
                    args.seed,
                    report,
                    server_version=server_version,
                )
                metrics["reload_scalar_index_queries_total"] += (
                    reload_scalar_index_queries
                )
                metrics[
                    f"{collection_metric_prefix}reload_scalar_index_queries_total"
                ] = reload_scalar_index_queries
                output_checkpoint["collections"][collection] = {
                    "schema_name": schema_name,
                    "actual_indexes": actual_indexes,
                    "indexed_fields": [
                        index.get("field_name") for index in actual_indexes
                    ],
                    "indexed_vector_fields": [
                        index.field for index, _ in _indexed_vector_indexes(spec)
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

        checkpoint_written = args.phase == "after-upgrade" and report.passed
        if checkpoint_written:
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
        if checkpoint_written or args.phase == "after-rollback":
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
