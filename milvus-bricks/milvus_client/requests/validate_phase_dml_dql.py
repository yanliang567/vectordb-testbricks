from __future__ import annotations

from pathlib import Path
import json
import math
import sys
from time import monotonic, sleep
from typing import Any

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.client import create_client, get_server_version
from milvus_client.common.data import (
    apply_deterministic_update,
    generate_field_value,
    generate_primary_key_value,
    generate_rows,
    generate_struct_array_value,
    indexed_vector_fields,
    prepare_struct_vector_query,
    update_projection_field,
)
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.pressure_maintenance import record_maintenance_window
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
    function_input_query_value,
    index_type_for_field,
    metric_type_for_field,
    search_params_for_field,
)
from milvus_client.common.version import (
    diskann_max_sim_cached_distance_bug,
    server_version_for_feature_detection,
)
from milvus_client.requests.validate_index_compatibility import (
    indexed_scalar_indexes,
    scalar_index_filter_for_value,
    validate_scalar_index_queries,
)


PHASE_DML_FAILED = "PHASE_DML_FAILED"
PHASE_DQL_FAILED = "PHASE_DQL_FAILED"
PHASE_NEW_COLLECTION_FAILED = "PHASE_NEW_COLLECTION_FAILED"
PHASE_UPSERT_NOT_APPLIED = "PHASE_UPSERT_NOT_APPLIED"
PHASE_CHECKPOINT_RELOAD_FAILED = "PHASE_CHECKPOINT_RELOAD_FAILED"
PHASE_COLLECTION_RELOAD_FAILED = "PHASE_COLLECTION_RELOAD_FAILED"
PHASE_CHECKPOINT_INVALID = "PHASE_CHECKPOINT_INVALID"
PHASE_CHECKPOINT_TARGET_ONLY_COLLECTION_PRESENT = (
    "PHASE_CHECKPOINT_TARGET_ONLY_COLLECTION_PRESENT"
)
CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
PHASE_CHECKPOINT_NOT_FOUND = "PHASE_CHECKPOINT_NOT_FOUND"
DEFAULT_RELOAD_TIMEOUT_SEC = 120.0
_EXPECTED_PK_UNSET = object()


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
    parser.add_argument("--phase-checkpoint-file", default="")
    parser.add_argument("--validate-phase-checkpoint", type=parse_bool, default=False)
    parser.add_argument(
        "--phase-checkpoint-new-collections-contract",
        choices=("none", "target_only", "round_trip"),
        default="none",
    )
    parser.add_argument("--visibility-timeout-sec", type=int, default=120)
    parser.add_argument("--visibility-interval-sec", type=float, default=2.0)
    parser.add_argument(
        "--reload-timeout-sec", type=float, default=DEFAULT_RELOAD_TIMEOUT_SEC
    )
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


def _phase_checkpoint_path(args) -> Path:
    if args.phase_checkpoint_file:
        return Path(args.phase_checkpoint_file)
    return Path(args.checkpoint_dir) / "phase_dml_dql_after_upgrade.json"


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
    responses: list[tuple[Any, list[int]]] = []
    if spec.partitions:
        partition_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for offset, row in enumerate(rows):
            partition = _partition_for_id(spec.partitions, start_id + offset)
            partition_rows.setdefault(partition or "", []).append((offset, row))
        for partition, indexed_batch in partition_rows.items():
            positions = [offset for offset, _ in indexed_batch]
            batch = [row for _, row in indexed_batch]
            responses.append(
                (
                    client.insert(
                        collection_name=target_collection,
                        data=batch,
                        partition_name=partition or None,
                    ),
                    positions,
                )
            )
    else:
        responses.append(
            (
                client.insert(collection_name=target_collection, data=rows),
                list(range(len(rows))),
            )
        )

    if auto_id_enabled(spec):
        ids_by_position: list[Any] = [None] * len(rows)
        for response, positions in responses:
            ids = _extract_insert_ids(response)
            if len(ids) != len(positions):
                raise RuntimeError(
                    f"{target_collection}: auto-id insert returned {len(ids)} "
                    f"primary keys for {len(positions)} rows"
                )
            for position, pk in zip(positions, ids):
                ids_by_position[position] = pk
        if any(pk is None for pk in ids_by_position):
            raise RuntimeError(
                f"{target_collection}: auto-id insert response contains an empty primary key"
            )
        return ids_by_position

    ids: list[Any] = []
    for response, _ in responses:
        ids.extend(_extract_insert_ids(response))
    return ids


def _upsert_rows(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    rows: list[dict[str, Any]],
    start_id: int,
) -> None:
    if spec.partitions:
        partition_rows: dict[str, list[dict[str, Any]]] = {}
        for offset, row in enumerate(rows):
            partition = _partition_for_id(spec.partitions, start_id + offset)
            partition_rows.setdefault(partition or "", []).append(row)
        for partition, batch in partition_rows.items():
            client.upsert(
                collection_name=target_collection,
                data=batch,
                partition_name=partition or None,
            )
        return
    client.upsert(collection_name=target_collection, data=rows)


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


def _wait_for_validation(
    validator: Any,
    timeout_sec: int,
    interval_sec: float,
) -> tuple[ValidationReport, int]:
    deadline = monotonic() + max(0, timeout_sec)
    attempts = 0
    while True:
        attempts += 1
        current = ValidationReport()
        validator(current)
        if current.passed or monotonic() >= deadline:
            return current, attempts
        sleep(max(0.0, interval_sec))


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
    if not pk_values:
        return
    for start in range(0, len(pk_values), 100):
        batch = pk_values[start : start + 100]
        try:
            rows = _query_rows_by_pk_values(
                client,
                target_collection,
                primary_name,
                batch,
                [primary_name],
            )
        except Exception as exc:
            report.fail(
                PHASE_DQL_FAILED,
                "deleted primary key batch query failed",
                collection=target_collection,
                pk_values=batch,
                error=str(exc),
            )
            continue
        for row in rows:
            pk = row.get(primary_name)
            report.fail(
                PHASE_DQL_FAILED,
                "deleted primary key is still queryable",
                collection=target_collection,
                pk=pk,
            )


def _query_rows_by_pk_values(
    client: Any,
    target_collection: str,
    primary_name: str,
    pk_values: list[Any],
    output_fields: list[str],
) -> list[dict[str, Any]]:
    if not pk_values:
        return []
    values = ", ".join(format_filter_value(value) for value in pk_values)
    return client.query(
        collection_name=target_collection,
        filter=f"{primary_name} in [{values}]",
        output_fields=output_fields,
        limit=len(pk_values),
    )


def _validate_pk_values_present_strict(
    client: Any,
    target_collection: str,
    primary_name: str,
    pk_values: list[Any],
    report: ValidationReport,
) -> None:
    if not pk_values:
        return
    found = set()
    for start in range(0, len(pk_values), 100):
        batch = pk_values[start : start + 100]
        try:
            rows = _query_rows_by_pk_values(
                client,
                target_collection,
                primary_name,
                batch,
                [primary_name],
            )
        except Exception as exc:
            report.fail(
                PHASE_DQL_FAILED,
                "primary key checkpoint query failed",
                collection=target_collection,
                pk_values=batch,
                error=str(exc),
            )
            continue
        found.update(row.get(primary_name) for row in rows)
    for pk in pk_values:
        if pk not in found:
            report.fail(
                "MISSING_PK",
                "phase checkpoint primary key is missing",
                collection=target_collection,
                pk=pk,
            )


def _hit_value(hit: Any, key: str) -> Any:
    if isinstance(hit, dict):
        return hit.get(key)
    return getattr(hit, key, None)


def _hit_primary_key(hit: Any, primary_name: str) -> Any:
    value = _hit_value(hit, "id")
    if value is not None:
        return value
    value = _hit_value(hit, primary_name)
    if value is not None:
        return value
    entity = _hit_value(hit, "entity")
    if isinstance(entity, dict):
        return entity.get(primary_name)
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


def _phase_search_query(
    spec: SchemaSpec,
    field_name: str,
    vector_field: FieldSpec,
    data_pk: int,
    seed: int,
    apply_update: bool = False,
) -> tuple[Any, int | None] | None:
    function_outputs = function_output_fields(spec)
    if vector_field.name in function_outputs:
        query = function_input_query_value(spec, vector_field.name, data_pk, seed)
        if apply_update:
            function = next(
                item
                for item in spec.functions
                if vector_field.name in item.output_fields
            )
            input_field = resolve_field(spec, function.input_fields[0])
            updated_row = generate_rows(spec, start_id=data_pk, count=1, seed=seed)[0]
            apply_deterministic_update(spec, updated_row, data_pk)
            query = updated_row.get(input_field.name)
        if query is None or query == "":
            return None
        return query, None
    struct_array = struct_array_for_field(spec, field_name)
    updated_row = None
    if apply_update:
        updated_row = generate_rows(spec, start_id=data_pk, count=1, seed=seed)[0]
        apply_deterministic_update(spec, updated_row, data_pk)
    if struct_array is not None:
        value = (
            updated_row.get(struct_array.name)
            if updated_row is not None
            else generate_struct_array_value(struct_array, data_pk, seed)
        )
        if value is None:
            return None
        for offset, element in enumerate(value):
            vector = element.get(vector_field.name)
            if vector is not None:
                return prepare_struct_vector_query(
                    metric_type_for_field(spec, field_name), vector, offset
                )
        return None
    query = (
        updated_row.get(vector_field.name)
        if updated_row is not None
        else generate_field_value(vector_field, data_pk, seed)
    )
    if query is None:
        return None
    return query, None


def _select_phase_search_probe_pk(
    spec: SchemaSpec,
    start_id: int,
    rows: int,
    seed: int,
    apply_update: bool = False,
) -> int:
    if rows <= 0:
        return start_id
    fields = indexed_vector_fields(spec)
    for data_pk in range(start_id + rows - 1, start_id - 1, -1):
        if all(
            _phase_search_query(
                spec,
                field_name,
                vector_field,
                data_pk,
                seed,
                apply_update=apply_update,
            )
            is not None
            for field_name, vector_field in fields
        ):
            return data_pk
    return start_id + rows - 1


def _validate_phase_search_hit(
    result: Any,
    target_collection: str,
    field_name: str,
    primary_name: str,
    expected_pk: Any,
    expected_offset: int | None,
    metric_type: str,
    index_type: str,
    report: ValidationReport,
    lossy_index: bool = False,
    diskann_max_sim_bug: bool = False,
) -> None:
    assert_search_result(result, target_collection, field_name)
    hits = result[0]
    matched_hit = None
    for hit in hits:
        if _hit_primary_key(hit, primary_name) == expected_pk and (
            expected_offset is None or _hit_offset(hit) == expected_offset
        ):
            matched_hit = hit
            break
    if matched_hit is None:
        report.fail(
            PHASE_DQL_FAILED,
            "phase vector search did not return the newly written primary key",
            collection=target_collection,
            field=field_name,
            expected_pk=expected_pk,
            expected_offset=expected_offset,
            actual_pks=[_hit_primary_key(hit, primary_name) for hit in hits],
            actual_offsets=[_hit_offset(hit) for hit in hits],
        )
        return

    distance = _hit_distance(matched_hit)
    if distance is None:
        report.fail(
            PHASE_DQL_FAILED,
            "phase vector self-search did not expose a distance or score",
            collection=target_collection,
            field=field_name,
            expected_pk=expected_pk,
            metric_type=metric_type,
        )
        return
    if not math.isfinite(distance):
        report.fail(
            PHASE_DQL_FAILED,
            "phase vector self-search returned a non-finite distance or score",
            collection=target_collection,
            field=field_name,
            expected_pk=expected_pk,
            metric_type=metric_type,
            index_type=index_type,
            distance=distance,
        )
        return
    metric = metric_type.upper().removeprefix("MAX_SIM_")
    max_distance = 0.5 if lossy_index and metric == "L2" else 1e-3
    min_score = 0.5 if lossy_index and metric in {"COSINE", "IP"} else 0.9
    if metric in {"L2", "HAMMING", "JACCARD"} and distance < 0:
        report.fail(
            PHASE_DQL_FAILED,
            "phase vector self-search distance is lower than the metric minimum",
            collection=target_collection,
            field=field_name,
            expected_pk=expected_pk,
            metric_type=metric_type,
            index_type=index_type,
            distance=distance,
            min_distance=0.0,
        )
        return
    if metric in {"COSINE", "MHJACCARD"} and distance > 1.001:
        report.fail(
            PHASE_DQL_FAILED,
            "phase vector self-search score is higher than the metric maximum",
            collection=target_collection,
            field=field_name,
            expected_pk=expected_pk,
            metric_type=metric_type,
            index_type=index_type,
            distance=distance,
            max_score=1.001,
        )
        return
    if metric in {"L2", "HAMMING", "JACCARD"} and distance > max_distance:
        report.fail(
            PHASE_DQL_FAILED,
            "phase vector self-search distance is higher than expected",
            collection=target_collection,
            field=field_name,
            expected_pk=expected_pk,
            metric_type=metric_type,
            index_type=index_type,
            distance=distance,
            max_distance=max_distance,
        )
    if metric in {"COSINE", "IP", "MHJACCARD"} and distance < min_score:
        if (
            diskann_max_sim_bug
            and index_type.upper() == "DISKANN"
            and metric_type.upper().startswith("MAX_SIM_")
            and distance < 0
        ):
            report.metrics["diskann_max_sim_negative_score_known"] = True
        else:
            report.fail(
                PHASE_DQL_FAILED,
                "phase vector self-search score is lower than expected",
                collection=target_collection,
                field=field_name,
                expected_pk=expected_pk,
                metric_type=metric_type,
                index_type=index_type,
                distance=distance,
                min_score=min_score,
            )


def _run_searches(
    client: Any,
    spec: SchemaSpec,
    target_collection: str,
    seed: int,
    pk: int,
    report: ValidationReport,
    expected_pk: Any = _EXPECTED_PK_UNSET,
    apply_update: bool = False,
    diskann_max_sim_bug: bool = False,
) -> int:
    searches = 0
    primary = _primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    if expected_pk is _EXPECTED_PK_UNSET:
        expected_pk = (
            generate_primary_key_value(primary, pk) if primary is not None else pk
        )
    for field_name, vector_field in indexed_vector_fields(spec):
        metric_type = metric_type_for_field(spec, field_name)
        query_probe = _phase_search_query(
            spec,
            field_name,
            vector_field,
            pk,
            seed,
            apply_update=apply_update,
        )
        if query_probe is None:
            report.fail(
                PHASE_DQL_FAILED,
                "could not build deterministic phase vector search probe",
                collection=target_collection,
                field=field_name,
                data_pk=pk,
                expected_pk=expected_pk,
            )
            continue
        query_vector, expected_offset = query_probe
        filter_expr = f"{primary_name} == {format_filter_value(expected_pk)}"
        try:
            result = client.search(
                collection_name=target_collection,
                data=[query_vector],
                anns_field=field_name,
                filter=filter_expr,
                limit=5,
                search_params={
                    "metric_type": metric_type,
                    "params": search_params_for_field(spec, field_name),
                },
            )
            _validate_phase_search_hit(
                result,
                target_collection,
                field_name,
                primary_name,
                expected_pk,
                expected_offset,
                metric_type,
                index_type_for_field(spec, field_name),
                report,
                lossy_index=approximate_recall_index(spec, field_name),
                diskann_max_sim_bug=diskann_max_sim_bug,
            )
            searches += 1
        except Exception as exc:
            report.fail(
                PHASE_DQL_FAILED,
                "phase vector search failed",
                collection=target_collection,
                field=field_name,
                data_pk=pk,
                expected_pk=expected_pk,
                expected_offset=expected_offset,
                filter=filter_expr,
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
    validation_field = update_projection_field(spec)
    if validation_field is None:
        report.fail(
            PHASE_UPSERT_NOT_APPLIED,
            "schema has no queryable field for upsert visibility validation",
            collection=target_collection,
            schema=spec.name,
        )
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
            output_fields=[primary_name, validation_field],
            limit=len(sample_values),
        )
    except Exception as exc:
        report.fail(
            PHASE_DQL_FAILED,
            "upserted value query failed",
            collection=target_collection,
            field=validation_field,
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
                field=validation_field,
            )
            continue
        actual = _normalize_for_compare(row.get(validation_field))
        expected_rows = generate_rows(
            spec, start_id=start_id + offset, count=1, seed=seed + 101
        )
        apply_deterministic_update(spec, expected_rows[0], start_id + offset)
        expected = _normalize_for_compare(expected_rows[0].get(validation_field))
        if actual != expected:
            report.fail(
                PHASE_UPSERT_NOT_APPLIED,
                "upserted field value does not match expected updated value",
                collection=target_collection,
                pk=pk_value,
                field=validation_field,
                expected=expected,
                actual=actual,
            )


def _upsert_sample_payload(
    spec: SchemaSpec,
    primary: FieldSpec,
    start_id: int,
    sample_offsets: list[int],
    seed: int,
) -> dict[str, Any]:
    validation_field = update_projection_field(spec)
    if validation_field is None:
        return {"field": None, "samples": []}
    samples = []
    for offset in sample_offsets:
        expected_rows = generate_rows(
            spec,
            start_id=start_id + offset,
            count=1,
            seed=seed + 101,
        )
        apply_deterministic_update(spec, expected_rows[0], start_id + offset)
        pk_value = generate_primary_key_value(primary, start_id + offset)
        samples.append(
            {
                "pk": pk_value,
                "expected": _normalize_for_compare(
                    expected_rows[0].get(validation_field)
                ),
            }
        )
    return {"field": validation_field, "samples": samples}


def _validate_upsert_samples(
    client: Any,
    target_collection: str,
    primary_name: str,
    checkpoint: dict[str, Any],
    report: ValidationReport,
) -> None:
    validation_field = checkpoint.get("field")
    samples = checkpoint.get("samples") or []
    if not validation_field or not samples:
        return
    pk_values = [sample["pk"] for sample in samples]
    try:
        rows = _query_rows_by_pk_values(
            client,
            target_collection,
            primary_name,
            pk_values,
            [primary_name, validation_field],
        )
    except Exception as exc:
        report.fail(
            PHASE_DQL_FAILED,
            "phase checkpoint upsert sample query failed",
            collection=target_collection,
            field=validation_field,
            error=str(exc),
        )
        return
    rows_by_pk = {row.get(primary_name): row for row in rows}
    for sample in samples:
        row = rows_by_pk.get(sample["pk"])
        if not row:
            report.fail(
                PHASE_UPSERT_NOT_APPLIED,
                "phase checkpoint upserted primary key is missing",
                collection=target_collection,
                pk=sample["pk"],
                field=validation_field,
            )
            continue
        actual = _normalize_for_compare(row.get(validation_field))
        if actual != sample["expected"]:
            report.fail(
                PHASE_UPSERT_NOT_APPLIED,
                "phase checkpoint upserted field value does not match",
                collection=target_collection,
                pk=sample["pk"],
                field=validation_field,
                expected=sample["expected"],
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
    visibility_timeout_sec: int,
    visibility_interval_sec: float,
    report: ValidationReport,
    *,
    reload_timeout_sec: float = DEFAULT_RELOAD_TIMEOUT_SEC,
    reload_maintenance_label: str = "phase-dml-dql-reload",
    server_version: str | None = None,
    diskann_max_sim_bug: bool = False,
) -> dict[str, Any]:
    primary = _primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    metrics: dict[str, Any] = {
        "collection": target_collection,
        "schema_name": spec.name,
        "primary_field": primary_name,
        "start_id": start_id,
        "rows": rows,
        "inserted": 0,
        "upserted": 0,
        "deleted": 0,
        "inserted_values": [],
        "remaining_count": 0,
        "deleted_values": [],
        "remaining_values": [],
        "remaining_min_pk": None,
        "remaining_max_pk": None,
        "upsert_samples": {"field": None, "samples": []},
        "searches": 0,
        "scalar_index_queries": 0,
        "reload_attempted": False,
        "reload_operations_succeeded": False,
        "reload_succeeded": False,
        "reload_vector_searches": 0,
        "reload_scalar_index_queries": 0,
        "search_probe_data_pk": None,
        "search_probe_pk": None,
        "search_probe_seed": None,
        "upsert_skipped_auto_id": False,
        "visibility_attempts": 0,
    }
    inserted_ids: list[Any] = []

    try:
        for start in range(start_id, start_id + rows, batch_size):
            count = min(batch_size, start_id + rows - start)
            batch = generate_rows(spec, start_id=start, count=count, seed=seed)
            ids = _insert_rows(client, spec, target_collection, batch, start)
            inserted_ids.extend(ids)
            metrics["inserted"] += len(ids) if auto_id_enabled(spec) else len(batch)

        if auto_id_enabled(spec):
            if len(inserted_ids) != rows:
                raise RuntimeError(
                    f"{target_collection}: auto-id insert returned {len(inserted_ids)} "
                    f"primary keys for {rows} rows"
                )
            if len(set(inserted_ids)) != len(inserted_ids):
                raise RuntimeError(
                    f"{target_collection}: auto-id insert returned duplicate primary keys"
                )
            metrics["inserted_values"] = list(inserted_ids)

        if auto_id_enabled(spec):
            metrics["upsert_skipped_auto_id"] = True
        else:
            for start in range(start_id, start_id + rows, batch_size):
                count = min(batch_size, start_id + rows - start)
                batch = generate_rows(
                    spec, start_id=start, count=count, seed=seed + 101
                )
                for offset, row in enumerate(batch):
                    apply_deterministic_update(spec, row, start + offset)
                _upsert_rows(client, spec, target_collection, batch, start)
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
        metrics["deleted_values"] = list(deleted_values)
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
        remaining_values = inserted_ids[metrics["deleted"] :]
        metrics["remaining_count"] = max(0, metrics["inserted"] - metrics["deleted"])
        metrics["remaining_values"] = list(remaining_values)
    else:
        remaining_start_id = start_id + metrics["deleted"]
        min_pk = generate_primary_key_value(primary, remaining_start_id)
        max_pk = generate_primary_key_value(primary, start_id + rows - 1)
        metrics["remaining_count"] = rows - metrics["deleted"]
        metrics["remaining_min_pk"] = min_pk
        metrics["remaining_max_pk"] = max_pk
        sample_values = [
            generate_primary_key_value(primary, remaining_start_id),
            generate_primary_key_value(primary, start_id + rows - 1),
        ]
        metrics["remaining_values"] = sample_values
        metrics["upsert_samples"] = _upsert_sample_payload(
            spec,
            primary,
            start_id,
            [metrics["deleted"], rows - 1],
            seed,
        )

    validation_failures_before = len(report.failures)

    def validate_visibility(current: ValidationReport) -> None:
        if auto_id_enabled(spec):
            _validate_pk_values_present_strict(
                client,
                target_collection,
                primary_name,
                metrics["remaining_values"],
                current,
            )
        else:
            validate_collection_count(
                client,
                target_collection,
                rows - metrics["deleted"],
                current,
                filter_expr=pk_range_filter(
                    primary_name,
                    metrics["remaining_min_pk"],
                    metrics["remaining_max_pk"],
                ),
                metric_suffix="phase_existing_dml_count",
            )
            validate_pk_samples(
                client,
                target_collection,
                primary_name,
                metrics["remaining_values"],
                current,
            )
            _validate_upserted_values(
                client,
                spec,
                target_collection,
                primary,
                start_id,
                [metrics["deleted"], rows - 1],
                seed,
                current,
            )
        _validate_deleted_pk_values(
            client,
            target_collection,
            primary_name,
            deleted_values,
            current,
        )

    visibility_report, visibility_attempts = _wait_for_validation(
        validate_visibility,
        visibility_timeout_sec,
        visibility_interval_sec,
    )
    metrics["visibility_attempts"] = visibility_attempts
    report.metrics.update(visibility_report.metrics)
    if not visibility_report.passed:
        report.passed = False
        report.failures.extend(visibility_report.failures)
    if rows <= 0:
        return metrics
    search_probe_seed = seed if auto_id_enabled(spec) else seed + 101
    search_probe_data_pk = _select_phase_search_probe_pk(
        spec,
        start_id,
        rows,
        search_probe_seed,
        apply_update=not auto_id_enabled(spec),
    )
    if auto_id_enabled(spec):
        search_probe_pk = inserted_ids[search_probe_data_pk - start_id]
    else:
        search_probe_pk = generate_primary_key_value(primary, search_probe_data_pk)
    metrics["search_probe_data_pk"] = search_probe_data_pk
    metrics["search_probe_pk"] = search_probe_pk
    metrics["search_probe_seed"] = search_probe_seed
    metrics["searches"] = _run_searches(
        client,
        spec,
        target_collection,
        search_probe_seed,
        search_probe_data_pk,
        report,
        expected_pk=search_probe_pk,
        apply_update=not auto_id_enabled(spec),
        diskann_max_sim_bug=diskann_max_sim_bug,
    )
    metrics["scalar_index_queries"] = _validate_phase_checkpoint_scalar_indexes(
        client,
        spec,
        metrics,
        search_probe_seed,
        report,
        existing=True,
        server_version=server_version,
    )
    if len(report.failures) > validation_failures_before:
        return metrics
    metrics["reload_attempted"] = True
    if not _reload_phase_collection(
        client,
        target_collection,
        report,
        timeout_sec=reload_timeout_sec,
        maintenance_label=reload_maintenance_label,
    ):
        return metrics
    metrics["reload_operations_succeeded"] = True
    reload_failures_before = len(report.failures)
    metrics["reload_vector_searches"] = _validate_existing_phase_checkpoint_collection(
        client,
        spec,
        metrics,
        report,
        seed,
        metric_prefix="phase_reload",
        diskann_max_sim_bug=diskann_max_sim_bug,
    )
    metrics["reload_scalar_index_queries"] = _validate_phase_checkpoint_scalar_indexes(
        client,
        spec,
        metrics,
        search_probe_seed,
        report,
        existing=True,
        server_version=server_version,
    )
    metrics["reload_succeeded"] = len(report.failures) == reload_failures_before
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
    *,
    reload_timeout_sec: float = DEFAULT_RELOAD_TIMEOUT_SEC,
    reload_maintenance_label: str = "phase-dml-dql-reload",
    server_version: str | None = None,
    diskann_max_sim_bug: bool = False,
) -> dict[str, Any]:
    primary = _primary_field(spec)
    primary_name = primary.name if primary is not None else "id"
    metrics: dict[str, Any] = {
        "collection": target_collection,
        "schema_name": spec.name,
        "primary_field": primary_name,
        "start_id": start_id,
        "rows": rows,
        "inserted": 0,
        "inserted_values": [],
        "sample_values": [],
        "min_pk": None,
        "max_pk": None,
        "searches": 0,
        "scalar_index_queries": 0,
        "reload_attempted": False,
        "reload_operations_succeeded": False,
        "reload_succeeded": False,
        "reload_vector_searches": 0,
        "reload_scalar_index_queries": 0,
        "search_probe_data_pk": None,
        "search_probe_pk": None,
        "search_probe_seed": None,
    }
    inserted_ids: list[Any] = []
    try:
        _create_new_collection(client, spec, target_collection, drop_if_exists)
        for start in range(start_id, start_id + rows, batch_size):
            count = min(batch_size, start_id + rows - start)
            batch = generate_rows(spec, start_id=start, count=count, seed=seed)
            ids = _insert_rows(client, spec, target_collection, batch, start)
            inserted_ids.extend(ids)
            metrics["inserted"] += len(ids) if auto_id_enabled(spec) else len(batch)
        if auto_id_enabled(spec):
            if len(inserted_ids) != rows:
                raise RuntimeError(
                    f"{target_collection}: auto-id insert returned {len(inserted_ids)} "
                    f"primary keys for {rows} rows"
                )
            if len(set(inserted_ids)) != len(inserted_ids):
                raise RuntimeError(
                    f"{target_collection}: auto-id insert returned duplicate primary keys"
                )
            metrics["inserted_values"] = list(inserted_ids)
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

    validation_failures_before = len(report.failures)
    if auto_id_enabled(spec):
        validate_collection_count(
            client,
            target_collection,
            rows,
            report,
            metric_suffix="phase_new_collection_count",
        )
        _validate_pk_values_present_strict(
            client, target_collection, primary_name, inserted_ids, report
        )
        metrics["sample_values"] = list(inserted_ids[:3])
    else:
        min_pk = generate_primary_key_value(primary, start_id)
        max_pk = generate_primary_key_value(primary, start_id + rows - 1)
        metrics["min_pk"] = min_pk
        metrics["max_pk"] = max_pk
        validate_collection_count(
            client,
            target_collection,
            rows,
            report,
            filter_expr=pk_range_filter(primary_name, min_pk, max_pk),
            metric_suffix="phase_new_collection_count",
        )
        sample_values = [
            generate_primary_key_value(primary, start_id),
            generate_primary_key_value(primary, start_id + rows - 1),
        ]
        metrics["sample_values"] = sample_values
        validate_pk_samples(
            client, target_collection, primary_name, sample_values, report
        )
    if rows <= 0:
        return metrics
    search_probe_seed = seed
    search_probe_data_pk = _select_phase_search_probe_pk(
        spec, start_id, rows, search_probe_seed
    )
    if auto_id_enabled(spec):
        search_probe_pk = inserted_ids[search_probe_data_pk - start_id]
    else:
        search_probe_pk = generate_primary_key_value(primary, search_probe_data_pk)
    metrics["search_probe_data_pk"] = search_probe_data_pk
    metrics["search_probe_pk"] = search_probe_pk
    metrics["search_probe_seed"] = search_probe_seed
    metrics["searches"] = _run_searches(
        client,
        spec,
        target_collection,
        search_probe_seed,
        search_probe_data_pk,
        report,
        expected_pk=search_probe_pk,
        diskann_max_sim_bug=diskann_max_sim_bug,
    )
    metrics["scalar_index_queries"] = _validate_phase_checkpoint_scalar_indexes(
        client,
        spec,
        metrics,
        search_probe_seed,
        report,
        existing=False,
        server_version=server_version,
    )
    if len(report.failures) > validation_failures_before:
        return metrics
    metrics["reload_attempted"] = True
    if not _reload_phase_collection(
        client,
        target_collection,
        report,
        timeout_sec=reload_timeout_sec,
        maintenance_label=reload_maintenance_label,
    ):
        return metrics
    metrics["reload_operations_succeeded"] = True
    reload_failures_before = len(report.failures)
    metrics["reload_vector_searches"] = _validate_new_phase_checkpoint_collection(
        client,
        spec,
        metrics,
        report,
        seed,
        metric_prefix="phase_reload",
        diskann_max_sim_bug=diskann_max_sim_bug,
    )
    metrics["reload_scalar_index_queries"] = _validate_phase_checkpoint_scalar_indexes(
        client,
        spec,
        metrics,
        search_probe_seed,
        report,
        existing=False,
        server_version=server_version,
    )
    metrics["reload_succeeded"] = len(report.failures) == reload_failures_before
    return metrics


def _write_after_upgrade_phase_checkpoint(
    path: Path,
    args,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "version": 2,
        "phase": args.phase,
        "existing_start_id": args.existing_start_id,
        "new_start_id": args.new_start_id,
        "existing_dml_rows": args.existing_dml_rows,
        "existing_delete_rows": args.existing_delete_rows,
        "new_collection_rows": args.new_collection_rows,
        "existing_collections": {
            item["collection"]: item for item in metrics["existing_collections"]
        },
        "new_collections": {
            item["collection"]: item for item in metrics["new_collections"]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _validate_existing_phase_checkpoint_collection(
    client: Any,
    spec: SchemaSpec,
    checkpoint: dict[str, Any],
    report: ValidationReport,
    seed: int,
    *,
    metric_prefix: str = "phase_checkpoint",
    diskann_max_sim_bug: bool = False,
) -> int:
    collection = checkpoint["collection"]
    primary_name = checkpoint["primary_field"]
    searches = 0
    if checkpoint.get("remaining_min_pk") is not None:
        validate_collection_count(
            client,
            collection,
            int(checkpoint["remaining_count"]),
            report,
            filter_expr=pk_range_filter(
                primary_name,
                checkpoint["remaining_min_pk"],
                checkpoint["remaining_max_pk"],
            ),
            metric_suffix=f"{metric_prefix}_existing_count",
        )
    _validate_pk_values_present_strict(
        client,
        collection,
        primary_name,
        checkpoint.get("remaining_values") or [],
        report,
    )
    _validate_deleted_pk_values(
        client,
        collection,
        primary_name,
        checkpoint.get("deleted_values") or [],
        report,
    )
    _validate_upsert_samples(
        client,
        collection,
        primary_name,
        checkpoint.get("upsert_samples") or {},
        report,
    )
    if checkpoint.get("rows", 0) > 0:
        search_probe_data_pk = int(
            checkpoint.get("search_probe_data_pk")
            or int(checkpoint["start_id"]) + int(checkpoint["rows"]) - 1
        )
        search_probe_pk = checkpoint.get("search_probe_pk", _EXPECTED_PK_UNSET)
        search_probe_seed = int(
            checkpoint.get(
                "search_probe_seed",
                seed if auto_id_enabled(spec) else seed + 101,
            )
        )
        if search_probe_pk is _EXPECTED_PK_UNSET and auto_id_enabled(spec):
            report.fail(
                PHASE_DQL_FAILED,
                "auto-id phase checkpoint lacks the actual search probe primary key",
                collection=collection,
                data_pk=search_probe_data_pk,
            )
            return searches
        searches += _run_searches(
            client,
            spec,
            collection,
            search_probe_seed,
            search_probe_data_pk,
            report,
            expected_pk=search_probe_pk,
            apply_update=not auto_id_enabled(spec),
            diskann_max_sim_bug=diskann_max_sim_bug,
        )
    return searches


def _validate_new_phase_checkpoint_collection(
    client: Any,
    spec: SchemaSpec,
    checkpoint: dict[str, Any],
    report: ValidationReport,
    seed: int,
    *,
    metric_prefix: str = "phase_checkpoint",
    diskann_max_sim_bug: bool = False,
) -> int:
    collection = checkpoint["collection"]
    primary_name = checkpoint["primary_field"]
    searches = 0
    if checkpoint.get("min_pk") is not None:
        validate_collection_count(
            client,
            collection,
            int(checkpoint["inserted"]),
            report,
            filter_expr=pk_range_filter(
                primary_name, checkpoint["min_pk"], checkpoint["max_pk"]
            ),
            metric_suffix=f"{metric_prefix}_new_collection_count",
        )
    else:
        validate_collection_count(
            client,
            collection,
            int(checkpoint["inserted"]),
            report,
            metric_suffix=f"{metric_prefix}_new_collection_count",
        )
    _validate_pk_values_present_strict(
        client,
        collection,
        primary_name,
        checkpoint.get("inserted_values") or checkpoint.get("sample_values") or [],
        report,
    )
    if checkpoint.get("rows", 0) > 0:
        search_probe_data_pk = int(
            checkpoint.get("search_probe_data_pk")
            or int(checkpoint["start_id"]) + int(checkpoint["rows"]) - 1
        )
        search_probe_pk = checkpoint.get("search_probe_pk", _EXPECTED_PK_UNSET)
        search_probe_seed = int(checkpoint.get("search_probe_seed", seed))
        if search_probe_pk is _EXPECTED_PK_UNSET and auto_id_enabled(spec):
            report.fail(
                PHASE_DQL_FAILED,
                "auto-id phase checkpoint lacks the actual search probe primary key",
                collection=collection,
                data_pk=search_probe_data_pk,
            )
            return searches
        searches += _run_searches(
            client,
            spec,
            collection,
            search_probe_seed,
            search_probe_data_pk,
            report,
            expected_pk=search_probe_pk,
            diskann_max_sim_bug=diskann_max_sim_bug,
        )
    return searches


def _reload_collection_strict(
    client: Any,
    collection: str,
    report: ValidationReport,
    *,
    failure_type: str,
    context: str,
    timeout_sec: float,
    maintenance_label: str,
) -> bool:
    if timeout_sec <= 0:
        report.fail(
            failure_type,
            "collection reload timeout must be positive",
            collection=collection,
            operation="configuration",
            timeout_sec=timeout_sec,
            context=context,
        )
        return False
    methods = []
    for operation in ("release_collection", "load_collection"):
        method = getattr(client, operation, None)
        if method is None:
            report.fail(
                failure_type,
                "Milvus client does not expose a required collection reload operation",
                collection=collection,
                operation=operation,
                context=context,
            )
            return False
        methods.append((operation, method))
    maintenance_windows = report.metrics.setdefault("maintenance_windows", [])
    with record_maintenance_window(
        maintenance_windows,
        label=maintenance_label,
        source="validate_phase_dml_dql",
        collection=collection,
    ):
        for operation, method in methods:
            try:
                method(collection_name=collection, timeout=timeout_sec)
            except TypeError:
                try:
                    method(collection, timeout=timeout_sec)
                except Exception as exc:
                    report.fail(
                        failure_type,
                        f"{context} collection reload failed",
                        collection=collection,
                        operation=operation,
                        error=str(exc),
                        context=context,
                        timeout_sec=timeout_sec,
                    )
                    return False
            except Exception as exc:
                report.fail(
                    failure_type,
                    f"{context} collection reload failed",
                    collection=collection,
                    operation=operation,
                    error=str(exc),
                    context=context,
                    timeout_sec=timeout_sec,
                )
                return False
    return True


def _reload_phase_collection(
    client: Any,
    collection: str,
    report: ValidationReport,
    *,
    timeout_sec: float = DEFAULT_RELOAD_TIMEOUT_SEC,
    maintenance_label: str = "phase-dml-dql-reload",
) -> bool:
    return _reload_collection_strict(
        client,
        collection,
        report,
        failure_type=PHASE_COLLECTION_RELOAD_FAILED,
        context="phase DML/DQL",
        timeout_sec=timeout_sec,
        maintenance_label=maintenance_label,
    )


def _reload_phase_checkpoint_collection(
    client: Any,
    collection: str,
    report: ValidationReport,
    *,
    timeout_sec: float = DEFAULT_RELOAD_TIMEOUT_SEC,
) -> bool:
    return _reload_collection_strict(
        client,
        collection,
        report,
        failure_type=PHASE_CHECKPOINT_RELOAD_FAILED,
        context="phase checkpoint",
        timeout_sec=timeout_sec,
        maintenance_label="phase-checkpoint-reload-after-rollback",
    )


def _phase_checkpoint_index_meta(
    spec: SchemaSpec,
    checkpoint: dict[str, Any],
    *,
    existing: bool,
) -> dict[str, Any] | None:
    rows = int(checkpoint.get("rows", 0))
    if rows <= 0:
        return None
    start_id = int(checkpoint["start_id"])
    deleted = int(checkpoint.get("deleted", 0)) if existing else 0
    data_min_pk = start_id + deleted
    data_max_pk = start_id + rows - 1
    if data_min_pk > data_max_pk:
        return None
    meta: dict[str, Any] = {
        "primary_field": checkpoint["primary_field"],
        "min_pk": data_min_pk,
        "max_pk": data_max_pk,
        "data_min_pk": data_min_pk,
        "data_max_pk": data_max_pk,
    }
    if auto_id_enabled(spec):
        pk_values = (
            checkpoint.get("remaining_values")
            if existing
            else checkpoint.get("inserted_values")
        ) or []
        meta["pk_values"] = list(pk_values)
    return meta


def _validate_phase_checkpoint_scalar_indexes(
    client: Any,
    spec: SchemaSpec,
    checkpoint: dict[str, Any],
    seed: int,
    report: ValidationReport,
    *,
    existing: bool,
    server_version: str | None = None,
) -> int:
    meta = _phase_checkpoint_index_meta(spec, checkpoint, existing=existing)
    if meta is None:
        return 0
    return validate_scalar_index_queries(
        client,
        checkpoint["collection"],
        spec,
        meta,
        seed,
        report,
        probe_overrides=_phase_upsert_scalar_probe_overrides(spec, checkpoint),
        server_version=server_version,
    )


def _phase_upsert_scalar_probe_overrides(
    spec: SchemaSpec,
    checkpoint: dict[str, Any],
) -> dict[str, tuple[int, Any, str]]:
    upsert_samples = checkpoint.get("upsert_samples") or {}
    updated_field = upsert_samples.get("field")
    samples = upsert_samples.get("samples") or []
    if not updated_field or not samples:
        return {}
    sample = samples[0]
    data_pk = int(checkpoint["start_id"]) + int(checkpoint.get("deleted", 0))
    overrides = {}
    for index, field in indexed_scalar_indexes(spec):
        if index.field != updated_field and not index.field.startswith(
            f"{updated_field}["
        ):
            continue
        filter_expr = scalar_index_filter_for_value(
            spec,
            index,
            field,
            sample.get("expected"),
        )
        if filter_expr:
            overrides[index.field] = (data_pk, sample["pk"], filter_expr)
    return overrides


def _validate_phase_checkpoint_entry_payload(
    collection_checkpoint: dict[str, Any],
    spec: SchemaSpec,
    report: ValidationReport,
    *,
    group_name: str,
    collection: str,
    expected_start_id: int,
    expected_rows: int,
    expected_delete_rows: int,
    expected_seed: int,
) -> None:
    primary = _primary_field(spec)
    expected_primary_field = primary.name if primary is not None else "id"
    if collection_checkpoint.get("primary_field") != expected_primary_field:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint primary field does not match the schema",
            group=group_name,
            collection=collection,
            expected=expected_primary_field,
            actual=collection_checkpoint.get("primary_field"),
        )
    expected_values = {
        "start_id": expected_start_id,
        "rows": expected_rows,
        "inserted": expected_rows,
    }
    if group_name == "existing_collections":
        expected_values.update(
            {
                "deleted": expected_delete_rows,
                "remaining_count": expected_rows - expected_delete_rows,
            }
        )
    for field_name, expected_value in expected_values.items():
        actual_value = collection_checkpoint.get(field_name)
        if type(actual_value) is not int or actual_value != expected_value:
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "phase checkpoint collection payload does not match run parameters",
                group=group_name,
                collection=collection,
                field=field_name,
                expected=expected_value,
                actual=actual_value,
            )

    probe_min_pk = expected_start_id
    if group_name == "existing_collections":
        probe_min_pk += expected_delete_rows
    probe_max_pk = expected_start_id + expected_rows - 1
    probe_data_pk = collection_checkpoint.get("search_probe_data_pk")
    if (
        type(probe_data_pk) is not int
        or probe_data_pk < probe_min_pk
        or probe_data_pk > probe_max_pk
    ):
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint search probe is outside the persisted data range",
            group=group_name,
            collection=collection,
            probe_data_pk=probe_data_pk,
            probe_min_pk=probe_min_pk,
            probe_max_pk=probe_max_pk,
        )
    if collection_checkpoint.get("search_probe_pk") is None:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint search probe lacks the actual primary key",
            group=group_name,
            collection=collection,
        )
    if type(collection_checkpoint.get("search_probe_seed")) is not int:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint search probe lacks a deterministic seed",
            group=group_name,
            collection=collection,
            actual=collection_checkpoint.get("search_probe_seed"),
        )

    expected_searches = len(indexed_vector_fields(spec))
    actual_searches = collection_checkpoint.get("searches")
    if type(actual_searches) is not int or actual_searches != expected_searches:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint vector search coverage is incomplete",
            group=group_name,
            collection=collection,
            expected_searches=expected_searches,
            actual_searches=actual_searches,
        )
    scalar_queries = collection_checkpoint.get("scalar_index_queries")
    if type(scalar_queries) is not int or scalar_queries < 0:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint scalar index query count is invalid",
            group=group_name,
            collection=collection,
            actual=scalar_queries,
        )

    expected_probe_seed = (
        expected_seed
        if group_name == "existing_collections" and auto_id_enabled(spec)
        else expected_seed + 101
        if group_name == "existing_collections"
        else expected_seed + 17
    )
    if collection_checkpoint.get("search_probe_seed") != expected_probe_seed:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint search probe seed does not match the workflow",
            group=group_name,
            collection=collection,
            expected=expected_probe_seed,
            actual=collection_checkpoint.get("search_probe_seed"),
        )

    primary = _primary_field(spec)
    if primary is None:
        return
    inserted_values = collection_checkpoint.get("inserted_values")
    sample_values = collection_checkpoint.get("sample_values")
    if auto_id_enabled(spec):
        if (
            not isinstance(inserted_values, list)
            or len(inserted_values) != expected_rows
            or len(set(inserted_values)) != expected_rows
        ):
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "auto-id phase checkpoint lacks the complete unique returned IDs",
                group=group_name,
                collection=collection,
                expected_count=expected_rows,
                actual_count=(
                    len(inserted_values) if isinstance(inserted_values, list) else None
                ),
            )
            return
        expected_samples = inserted_values[:3]
        if group_name == "existing_collections":
            expected_auto_id_values = {
                "deleted_values": inserted_values[:expected_delete_rows],
                "remaining_values": inserted_values[expected_delete_rows:],
                "remaining_min_pk": None,
                "remaining_max_pk": None,
                "upserted": 0,
                "upsert_skipped_auto_id": True,
                "upsert_samples": {"field": None, "samples": []},
            }
            for field_name, expected_value in expected_auto_id_values.items():
                if collection_checkpoint.get(field_name) != expected_value:
                    report.fail(
                        PHASE_CHECKPOINT_INVALID,
                        "auto-id phase checkpoint oracle is incomplete",
                        group=group_name,
                        collection=collection,
                        field=field_name,
                        expected=expected_value,
                        actual=collection_checkpoint.get(field_name),
                    )
        elif sample_values != expected_samples:
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "auto-id new collection sample IDs do not match returned IDs",
                group=group_name,
                collection=collection,
                expected=expected_samples,
                actual=sample_values,
            )
        return

    expected_min_pk = generate_primary_key_value(
        primary,
        expected_start_id
        + (expected_delete_rows if group_name == "existing_collections" else 0),
    )
    expected_max_pk = generate_primary_key_value(
        primary,
        expected_start_id + expected_rows - 1,
    )
    if group_name == "existing_collections":
        expected_explicit_values = {
            "inserted_values": [],
            "remaining_min_pk": expected_min_pk,
            "remaining_max_pk": expected_max_pk,
            "remaining_values": [expected_min_pk, expected_max_pk],
            "deleted_values": [
                generate_primary_key_value(primary, expected_start_id + offset)
                for offset in range(expected_delete_rows)
            ],
            "upserted": expected_rows,
            "upsert_skipped_auto_id": False,
            "upsert_samples": _upsert_sample_payload(
                spec,
                primary,
                expected_start_id,
                [expected_delete_rows, expected_rows - 1],
                expected_seed,
            ),
        }
    else:
        expected_explicit_values = {
            "inserted_values": [],
            "min_pk": expected_min_pk,
            "max_pk": expected_max_pk,
            "sample_values": [expected_min_pk, expected_max_pk],
        }
    for field_name, expected_value in expected_explicit_values.items():
        if collection_checkpoint.get(field_name) != expected_value:
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "explicit-PK phase checkpoint oracle is incomplete",
                group=group_name,
                collection=collection,
                field=field_name,
                expected=expected_value,
                actual=collection_checkpoint.get(field_name),
            )


def _validate_phase_checkpoint_contract(
    checkpoint: Any,
    specs: dict[str, SchemaSpec],
    report: ValidationReport,
    *,
    expected_existing_collections: dict[str, str],
    expected_new_collection_prefix: str,
    expected_existing_dml_rows: int,
    expected_existing_delete_rows: int,
    expected_new_collection_rows: int,
    expected_seed: int,
) -> bool:
    failures_before = len(report.failures)
    if not isinstance(checkpoint, dict):
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint root must be an object",
            actual_type=type(checkpoint).__name__,
        )
        return False
    if type(checkpoint.get("version")) is not int or checkpoint.get("version") != 2:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint version is unsupported",
            expected_version=2,
            actual_version=checkpoint.get("version"),
        )
    if checkpoint.get("phase") != "after-upgrade":
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint was not produced after upgrade",
            expected_phase="after-upgrade",
            actual_phase=checkpoint.get("phase"),
        )

    expected_run_parameters = {
        "existing_dml_rows": expected_existing_dml_rows,
        "existing_delete_rows": expected_existing_delete_rows,
        "new_collection_rows": expected_new_collection_rows,
    }
    if expected_existing_dml_rows <= 0 or expected_new_collection_rows <= 0:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint validation requires positive phase row counts",
            expected_existing_dml_rows=expected_existing_dml_rows,
            expected_new_collection_rows=expected_new_collection_rows,
        )
    if not 0 <= expected_existing_delete_rows < expected_existing_dml_rows:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint delete count must leave searchable rows",
            expected_existing_dml_rows=expected_existing_dml_rows,
            expected_existing_delete_rows=expected_existing_delete_rows,
        )
    for field_name, expected_value in expected_run_parameters.items():
        actual_value = checkpoint.get(field_name)
        if type(actual_value) is not int or actual_value != expected_value:
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "phase checkpoint run parameters do not match rollback expectations",
                field=field_name,
                expected=expected_value,
                actual=actual_value,
            )
    start_ids = {}
    for field_name in ("existing_start_id", "new_start_id"):
        value = checkpoint.get(field_name)
        if type(value) is not int or value < 0:
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "phase checkpoint start id is invalid",
                field=field_name,
                actual=value,
            )
        else:
            start_ids[field_name] = value

    if not expected_existing_collections:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "seed checkpoint does not declare existing collections",
        )
    if not expected_new_collection_prefix:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "expected target-written collection prefix is empty",
        )
    expected_new_collections = {
        collection_name(expected_new_collection_prefix, spec): schema_name
        for schema_name, spec in specs.items()
    }
    expected_groups = {
        "existing_collections": expected_existing_collections,
        "new_collections": expected_new_collections,
    }

    expected_schemas = set(specs)
    observed_group_collections: dict[str, set[str]] = {}
    for group_name in ("existing_collections", "new_collections"):
        group = checkpoint.get(group_name)
        if not isinstance(group, dict):
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "phase checkpoint collection group must be an object",
                group=group_name,
                actual_type=type(group).__name__,
            )
            continue

        schema_names: list[str] = []
        observed_collections: dict[str, str] = {}
        for collection_key, collection_checkpoint in group.items():
            if not isinstance(collection_checkpoint, dict):
                report.fail(
                    PHASE_CHECKPOINT_INVALID,
                    "phase checkpoint collection entry must be an object",
                    group=group_name,
                    collection_key=collection_key,
                    actual_type=type(collection_checkpoint).__name__,
                )
                continue
            collection = collection_checkpoint.get("collection")
            schema_name = collection_checkpoint.get("schema_name")
            if not isinstance(collection, str) or not collection:
                report.fail(
                    PHASE_CHECKPOINT_INVALID,
                    "phase checkpoint collection entry lacks a collection name",
                    group=group_name,
                    collection_key=collection_key,
                )
            elif collection != collection_key:
                report.fail(
                    PHASE_CHECKPOINT_INVALID,
                    "phase checkpoint collection key does not match its payload",
                    group=group_name,
                    collection_key=collection_key,
                    collection=collection,
                )
            if not isinstance(schema_name, str) or not schema_name:
                report.fail(
                    PHASE_CHECKPOINT_INVALID,
                    "phase checkpoint collection entry lacks a schema name",
                    group=group_name,
                    collection_key=collection_key,
                )
                continue
            schema_names.append(schema_name)
            observed_collections[collection_key] = schema_name

            expected_schema_name = expected_groups[group_name].get(collection_key)
            start_id_field = (
                "existing_start_id"
                if group_name == "existing_collections"
                else "new_start_id"
            )
            if (
                expected_schema_name == schema_name
                and schema_name in specs
                and start_id_field in start_ids
            ):
                _validate_phase_checkpoint_entry_payload(
                    collection_checkpoint,
                    specs[schema_name],
                    report,
                    group_name=group_name,
                    collection=collection_key,
                    expected_start_id=start_ids[start_id_field],
                    expected_rows=(
                        expected_existing_dml_rows
                        if group_name == "existing_collections"
                        else expected_new_collection_rows
                    ),
                    expected_delete_rows=(
                        expected_existing_delete_rows
                        if group_name == "existing_collections"
                        else 0
                    ),
                    expected_seed=expected_seed,
                )

        observed_schemas = set(schema_names)
        duplicate_schemas = sorted(
            schema_name
            for schema_name in observed_schemas
            if schema_names.count(schema_name) > 1
        )
        missing_schemas = sorted(expected_schemas - observed_schemas)
        unexpected_schemas = sorted(observed_schemas - expected_schemas)
        if missing_schemas or unexpected_schemas or duplicate_schemas:
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "phase checkpoint schema coverage is incomplete or ambiguous",
                group=group_name,
                missing_schemas=missing_schemas,
                unexpected_schemas=unexpected_schemas,
                duplicate_schemas=duplicate_schemas,
            )
        expected_collections = expected_groups[group_name]
        missing_collections = sorted(
            set(expected_collections) - set(observed_collections)
        )
        unexpected_collections = sorted(
            set(observed_collections) - set(expected_collections)
        )
        schema_mismatches = {
            collection: {
                "expected": expected_collections[collection],
                "actual": observed_collections[collection],
            }
            for collection in sorted(
                set(expected_collections) & set(observed_collections)
            )
            if expected_collections[collection] != observed_collections[collection]
        }
        if missing_collections or unexpected_collections or schema_mismatches:
            report.fail(
                PHASE_CHECKPOINT_INVALID,
                "phase checkpoint collection identities do not match the workflow",
                group=group_name,
                missing_collections=missing_collections,
                unexpected_collections=unexpected_collections,
                schema_mismatches=schema_mismatches,
            )
        observed_group_collections[group_name] = set(observed_collections)

    overlapping_collections = sorted(
        observed_group_collections.get("existing_collections", set())
        & observed_group_collections.get("new_collections", set())
    )
    if overlapping_collections:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "existing and target-written checkpoint collections overlap",
            overlapping_collections=overlapping_collections,
        )
    return len(report.failures) == failures_before


def _validate_phase_checkpoint_before_rollback(
    client: Any,
    specs: dict[str, SchemaSpec],
    path: Path,
    seed: int,
    report: ValidationReport,
    *,
    expected_existing_collections: dict[str, str],
    expected_new_collection_prefix: str,
    expected_existing_dml_rows: int,
    expected_existing_delete_rows: int,
    expected_new_collection_rows: int,
    new_collections_contract: str = "none",
    reload_timeout_sec: float = DEFAULT_RELOAD_TIMEOUT_SEC,
    server_version: str | None = None,
    diskann_max_sim_bug: bool = False,
) -> dict[str, Any]:
    metrics = {
        "phase_checkpoint_validated": False,
        "phase_checkpoint_new_collections_contract": new_collections_contract,
        "phase_checkpoint_existing_collections_total": 0,
        "phase_checkpoint_new_collections_total": 0,
        "phase_checkpoint_target_only_collections_absent_total": 0,
        "phase_checkpoint_target_only_collections_present_total": 0,
        "phase_checkpoint_searches_total": 0,
        "phase_checkpoint_scalar_index_queries_total": 0,
        "phase_checkpoint_reload_collections_total": 0,
        "phase_checkpoint_reload_failures_total": 0,
    }
    if not path.exists():
        report.fail(
            PHASE_CHECKPOINT_NOT_FOUND,
            "after-upgrade phase checkpoint file does not exist",
            path=str(path),
        )
        return metrics
    try:
        checkpoint = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report.fail(
            PHASE_CHECKPOINT_INVALID,
            "phase checkpoint cannot be read as JSON",
            path=str(path),
            error=str(exc),
        )
        return metrics
    if not _validate_phase_checkpoint_contract(
        checkpoint,
        specs,
        report,
        expected_existing_collections=expected_existing_collections,
        expected_new_collection_prefix=expected_new_collection_prefix,
        expected_existing_dml_rows=expected_existing_dml_rows,
        expected_existing_delete_rows=expected_existing_delete_rows,
        expected_new_collection_rows=expected_new_collection_rows,
        expected_seed=seed,
    ):
        return metrics
    for collection_checkpoint in checkpoint.get("existing_collections", {}).values():
        spec = specs.get(collection_checkpoint["schema_name"])
        if spec is None:
            report.fail(
                PHASE_DQL_FAILED,
                "schema from phase checkpoint is not present in schema matrix",
                collection=collection_checkpoint.get("collection"),
                schema=collection_checkpoint.get("schema_name"),
            )
            continue
        metrics["phase_checkpoint_existing_collections_total"] += 1
        if not _reload_phase_checkpoint_collection(
            client,
            collection_checkpoint["collection"],
            report,
            timeout_sec=reload_timeout_sec,
        ):
            metrics["phase_checkpoint_reload_failures_total"] += 1
            continue
        metrics["phase_checkpoint_reload_collections_total"] += 1
        metrics["phase_checkpoint_searches_total"] += (
            _validate_existing_phase_checkpoint_collection(
                client,
                spec,
                collection_checkpoint,
                report,
                seed,
                diskann_max_sim_bug=diskann_max_sim_bug,
            )
        )
        scalar_queries = _validate_phase_checkpoint_scalar_indexes(
            client,
            spec,
            collection_checkpoint,
            seed,
            report,
            existing=True,
            server_version=server_version,
        )
        metrics["phase_checkpoint_scalar_index_queries_total"] += scalar_queries
        metrics[
            f"phase_checkpoint.{collection_checkpoint['collection']}.scalar_index_queries_total"
        ] = scalar_queries
    for collection_checkpoint in checkpoint.get("new_collections", {}).values():
        spec = specs.get(collection_checkpoint["schema_name"])
        if spec is None:
            report.fail(
                PHASE_DQL_FAILED,
                "schema from phase checkpoint is not present in schema matrix",
                collection=collection_checkpoint.get("collection"),
                schema=collection_checkpoint.get("schema_name"),
            )
            continue
        metrics["phase_checkpoint_new_collections_total"] += 1
        if new_collections_contract == "target_only":
            collection = collection_checkpoint["collection"]
            try:
                present = client.has_collection(collection_name=collection)
            except Exception as exc:
                report.fail(
                    PHASE_DQL_FAILED,
                    "failed to verify target-only phase checkpoint collection absence",
                    collection=collection,
                    schema=collection_checkpoint["schema_name"],
                    contract=new_collections_contract,
                    error=str(exc),
                )
                continue
            if present:
                metrics["phase_checkpoint_target_only_collections_present_total"] += 1
                report.fail(
                    PHASE_CHECKPOINT_TARGET_ONLY_COLLECTION_PRESENT,
                    "target-only phase checkpoint collection must be absent after rollback",
                    collection=collection,
                    schema=collection_checkpoint["schema_name"],
                    contract=new_collections_contract,
                )
            else:
                metrics["phase_checkpoint_target_only_collections_absent_total"] += 1
            continue
        if not _reload_phase_checkpoint_collection(
            client,
            collection_checkpoint["collection"],
            report,
            timeout_sec=reload_timeout_sec,
        ):
            metrics["phase_checkpoint_reload_failures_total"] += 1
            continue
        metrics["phase_checkpoint_reload_collections_total"] += 1
        metrics["phase_checkpoint_searches_total"] += (
            _validate_new_phase_checkpoint_collection(
                client,
                spec,
                collection_checkpoint,
                report,
                seed + 17,
                diskann_max_sim_bug=diskann_max_sim_bug,
            )
        )
        scalar_queries = _validate_phase_checkpoint_scalar_indexes(
            client,
            spec,
            collection_checkpoint,
            seed + 17,
            report,
            existing=False,
            server_version=server_version,
        )
        metrics["phase_checkpoint_scalar_index_queries_total"] += scalar_queries
        metrics[
            f"phase_checkpoint.{collection_checkpoint['collection']}.scalar_index_queries_total"
        ] = scalar_queries
    metrics["phase_checkpoint_validated"] = report.passed
    return metrics


def _accumulate_phase_reload_metrics(
    metrics: dict[str, Any],
    collection_metrics: dict[str, Any],
) -> None:
    if collection_metrics.get("reload_attempted"):
        metrics["phase_reload_attempted_collections_total"] += 1
    if collection_metrics.get("reload_succeeded"):
        metrics["phase_reload_collections_total"] += 1
    elif collection_metrics.get("reload_attempted"):
        metrics["phase_reload_failures_total"] += 1
    metrics["phase_reload_vector_searches_total"] += int(
        collection_metrics.get("reload_vector_searches", 0)
    )
    metrics["phase_reload_scalar_index_queries_total"] += int(
        collection_metrics.get("reload_scalar_index_queries", 0)
    )


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
        metrics: dict[str, Any] = {
            "phase": args.phase,
            "reload_timeout_sec": args.reload_timeout_sec,
            "existing_collections_total": 0,
            "new_collections_total": 0,
            "existing_inserted_total": 0,
            "existing_upserted_total": 0,
            "existing_deleted_total": 0,
            "existing_upsert_skipped_auto_id_total": 0,
            "carried_collections_total": 0,
            "carried_collections_skipped_target_only_total": 0,
            "carried_inserted_total": 0,
            "carried_upserted_total": 0,
            "carried_deleted_total": 0,
            "new_collection_inserted_total": 0,
            "searches_total": 0,
            "scalar_index_queries_total": 0,
            "phase_reload_attempted_collections_total": 0,
            "phase_reload_collections_total": 0,
            "phase_reload_failures_total": 0,
            "phase_reload_vector_searches_total": 0,
            "phase_reload_scalar_index_queries_total": 0,
            "existing_collections": [],
            "carried_collections": [],
            "new_collections": [],
            "phase_checkpoint_path": str(_phase_checkpoint_path(args)),
            "phase_checkpoint_validated": False,
            "phase_checkpoint_new_collections_contract": (
                args.phase_checkpoint_new_collections_contract
            ),
            "phase_checkpoint_existing_collections_total": 0,
            "phase_checkpoint_new_collections_total": 0,
            "phase_checkpoint_target_only_collections_absent_total": 0,
            "phase_checkpoint_target_only_collections_present_total": 0,
            "phase_checkpoint_searches_total": 0,
            "phase_checkpoint_scalar_index_queries_total": 0,
            "phase_checkpoint_reload_collections_total": 0,
            "phase_checkpoint_reload_failures_total": 0,
        }

        if args.phase == "after-rollback" and args.validate_phase_checkpoint:
            metrics.update(
                _validate_phase_checkpoint_before_rollback(
                    client,
                    specs,
                    _phase_checkpoint_path(args),
                    args.seed,
                    report,
                    expected_existing_collections={
                        collection: meta["schema_name"]
                        for collection, meta in checkpoint.get(
                            "collections", {}
                        ).items()
                    },
                    expected_new_collection_prefix=args.carried_collection_prefix,
                    expected_existing_dml_rows=args.existing_dml_rows,
                    expected_existing_delete_rows=args.existing_delete_rows,
                    expected_new_collection_rows=args.new_collection_rows,
                    new_collections_contract=(
                        args.phase_checkpoint_new_collections_contract
                    ),
                    reload_timeout_sec=args.reload_timeout_sec,
                    server_version=server_version,
                    diskann_max_sim_bug=diskann_max_sim_bug,
                )
            )
            if not report.passed:
                result.status = FAILED
                result.failures = report.failures
                result.metrics = {**report.metrics, **metrics}
                result.write(args.output_json)
                return 1

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
                args.visibility_timeout_sec,
                args.visibility_interval_sec,
                report,
                reload_timeout_sec=args.reload_timeout_sec,
                reload_maintenance_label=f"phase-dml-dql-reload-{args.phase}",
                server_version=server_version,
                diskann_max_sim_bug=diskann_max_sim_bug,
            )
            metrics["existing_collections"].append(existing_metrics)
            metrics["existing_inserted_total"] += existing_metrics["inserted"]
            metrics["existing_upserted_total"] += existing_metrics["upserted"]
            metrics["existing_deleted_total"] += existing_metrics["deleted"]
            metrics["searches_total"] += existing_metrics["searches"]
            metrics["scalar_index_queries_total"] += existing_metrics[
                "scalar_index_queries"
            ]
            _accumulate_phase_reload_metrics(metrics, existing_metrics)
            if existing_metrics["upsert_skipped_auto_id"]:
                metrics["existing_upsert_skipped_auto_id_total"] += 1

        if (
            args.carried_collection_prefix
            and args.phase_checkpoint_new_collections_contract != "target_only"
        ):
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
                    args.visibility_timeout_sec,
                    args.visibility_interval_sec,
                    report,
                    reload_timeout_sec=args.reload_timeout_sec,
                    reload_maintenance_label=f"phase-dml-dql-reload-{args.phase}",
                    server_version=server_version,
                    diskann_max_sim_bug=diskann_max_sim_bug,
                )
                metrics["carried_collections"].append(carried_metrics)
                metrics["carried_inserted_total"] += carried_metrics["inserted"]
                metrics["carried_upserted_total"] += carried_metrics["upserted"]
                metrics["carried_deleted_total"] += carried_metrics["deleted"]
                metrics["searches_total"] += carried_metrics["searches"]
                metrics["scalar_index_queries_total"] += carried_metrics[
                    "scalar_index_queries"
                ]
                _accumulate_phase_reload_metrics(metrics, carried_metrics)
        elif args.carried_collection_prefix:
            metrics["carried_collections_skipped_target_only_total"] = len(specs)

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
                reload_timeout_sec=args.reload_timeout_sec,
                reload_maintenance_label=f"phase-dml-dql-reload-{args.phase}",
                server_version=server_version,
                diskann_max_sim_bug=diskann_max_sim_bug,
            )
            metrics["new_collections"].append(new_metrics)
            metrics["new_collection_inserted_total"] += new_metrics["inserted"]
            metrics["searches_total"] += new_metrics["searches"]
            metrics["scalar_index_queries_total"] += new_metrics["scalar_index_queries"]
            _accumulate_phase_reload_metrics(metrics, new_metrics)

        if args.phase == "after-upgrade" and report.passed:
            _write_after_upgrade_phase_checkpoint(
                _phase_checkpoint_path(args),
                args,
                metrics,
            )

        result.status = PASSED if report.passed else FAILED
        result.failures = report.failures
        result.metrics = {**report.metrics, **metrics}
        if args.phase == "after-upgrade" and report.passed:
            result.checkpoint = {
                "path": str(_phase_checkpoint_path(args)),
                "version": 2,
            }
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
