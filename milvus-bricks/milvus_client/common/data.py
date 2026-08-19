from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from random import Random
from struct import pack, unpack
from typing import Any

from milvus_client.common.schema import (
    VECTOR_TYPES,
    FieldSpec,
    SchemaSpec,
    StructArraySpec,
    function_output_fields,
    resolve_field,
)


def canonical_float32(value: float) -> float:
    return unpack("!f", pack("!f", value))[0]


def stable_float_vector(seed: int, pk: int, dim: int) -> list[float]:
    rng = Random(seed + pk)
    values = [rng.random() for _ in range(dim)]
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        normalized = values
    else:
        normalized = [value / norm for value in values]
    return [canonical_float32(value) for value in normalized]


def stable_int8_vector(seed: int, pk: int, dim: int) -> list[int]:
    rng = Random(seed + pk)
    return [rng.randint(-128, 127) for _ in range(dim)]


def stable_float16_vector(seed: int, pk: int, dim: int):
    import numpy as np

    return np.asarray(stable_float_vector(seed, pk, dim), dtype=np.float16)


def stable_bfloat16_vector(seed: int, pk: int, dim: int) -> bytes:
    import numpy as np

    values = np.asarray(stable_float_vector(seed, pk, dim), dtype=np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def stable_int8_vector_array(seed: int, pk: int, dim: int):
    import numpy as np

    return np.asarray(stable_int8_vector(seed, pk, dim), dtype=np.int8)


def stable_binary_vector(seed: int, pk: int, dim: int) -> bytes:
    rng = Random(seed + pk)
    byte_count = max(1, dim // 8)
    return bytes(rng.getrandbits(8) for _ in range(byte_count))


def stable_sparse_vector(seed: int, pk: int, dim: int = 1024) -> dict[int, float]:
    rng = Random(seed + pk)
    return {rng.randint(0, dim - 1): rng.random() for _ in range(16)}


def stable_vector_value(field: FieldSpec, pk: int, seed: int) -> Any:
    if field.dtype == "FLOAT_VECTOR":
        return stable_float_vector(seed, pk, field.dim or 128)
    if field.dtype == "FLOAT16_VECTOR":
        return stable_float16_vector(seed, pk, field.dim or 128)
    if field.dtype == "BFLOAT16_VECTOR":
        return stable_bfloat16_vector(seed, pk, field.dim or 128)
    if field.dtype == "INT8_VECTOR":
        return stable_int8_vector_array(seed, pk, field.dim or 128)
    if field.dtype == "BINARY_VECTOR":
        return stable_binary_vector(seed, pk, field.dim or 128)
    if field.dtype == "SPARSE_FLOAT_VECTOR":
        return stable_sparse_vector(seed, pk)
    raise ValueError(f"Unsupported generated vector dtype: {field.dtype}")


def prepare_struct_vector_query(
    metric_type: str, vector: Any, offset: int
) -> tuple[Any, int | None]:
    if metric_type.upper().startswith("MAX_SIM_"):
        from pymilvus.client.embedding_list import EmbeddingList

        query = EmbeddingList()
        query.add(vector)
        return query, None
    return vector, offset


def _normalize_for_checksum(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_checksum(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], bytes):
            return {"__bytes__": value[0].hex()}
        return [_normalize_for_checksum(item) for item in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_normalize_for_checksum(item) for item in value]
    return value


def stable_checksum(
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
    primary_field: str = "id",
) -> str:
    digest = sha256()
    selected_rows = []
    for row in rows:
        if fields is None:
            selected = dict(row)
        else:
            selected = {field: row.get(field) for field in fields}
        sort_value = _normalize_for_checksum(row.get(primary_field))
        selected_rows.append(
            (sort_value is None, sort_value, _normalize_for_checksum(selected))
        )
    selected_rows.sort(key=lambda item: (item[0], item[1]))
    for _, _, selected in selected_rows:
        digest.update(
            json.dumps(
                selected, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        )
    return digest.hexdigest()


def checksum_fields_for_spec(spec: SchemaSpec) -> list[str]:
    if spec.checksum_fields:
        return list(spec.checksum_fields)
    function_outputs = function_output_fields(spec)
    scalar_fields = [
        field.name
        for field in spec.fields
        if field.dtype not in VECTOR_TYPES
        and not field.auto_id
        and field.name not in function_outputs
    ]
    return [*scalar_fields, *(struct_array.name for struct_array in spec.struct_arrays)]


def update_projection_field(spec: SchemaSpec) -> str | None:
    function_outputs = function_output_fields(spec)
    for field in spec.fields:
        if (
            field.primary
            or field.is_partition_key
            or field.name in function_outputs
            or field.dtype in VECTOR_TYPES
        ):
            continue
        return field.name
    if spec.struct_arrays:
        return spec.struct_arrays[0].name
    for field in spec.fields:
        if not field.primary and field.name not in function_outputs:
            return field.name
    return None


def _deterministic_update_value(field: FieldSpec, pk: int, current: Any) -> Any:
    if field.dtype in {"INT64", "INT32", "INT16", "INT8"}:
        return pk % 97 + 1
    if field.dtype == "FLOAT":
        return canonical_float32(float(pk % 997) + 0.25)
    if field.dtype == "DOUBLE":
        return float(pk % 997) + 0.25
    if field.dtype == "BOOL":
        return not bool(current)
    if field.dtype in {"VARCHAR", "STRING", "TEXT"}:
        value = f"phase_upsert_{pk}_milvus_upgrade_rollback"
        return value[: field.max_length] if field.max_length else value
    if field.dtype == "JSON":
        return {"phase_upsert_pk": pk, "active": True}
    if field.dtype == "ARRAY":
        if field.element_type in {"INT64", "INT32", "INT16", "INT8"}:
            return [pk % 97, (pk + 1) % 97]
        if field.element_type == "FLOAT":
            return [canonical_float32(float(pk % 97) + 0.25)]
        if field.element_type == "DOUBLE":
            return [float(pk % 97) + 0.25]
        if field.element_type == "BOOL":
            return [True, False]
        return [f"phase_upsert_{pk}"]
    if field.dtype == "GEOMETRY":
        return f"POINT ({-100.0 + (pk % 10):g} {20.0 + (pk % 10):g})"
    if field.dtype == "TIMESTAMPTZ":
        return (
            (datetime(2200, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=pk))
            .isoformat()
            .replace("+00:00", "Z")
        )
    if field.dtype in VECTOR_TYPES:
        return stable_vector_value(field, pk, 909)
    return current


def apply_deterministic_update(
    spec: SchemaSpec, row: dict[str, Any], pk: int
) -> str | None:
    projection = update_projection_field(spec)
    if projection is None:
        return None
    top_level = next((field for field in spec.fields if field.name == projection), None)
    if top_level is not None:
        row[projection] = _deterministic_update_value(
            top_level, pk, row.get(projection)
        )
        return projection
    struct_array = next(
        (item for item in spec.struct_arrays if item.name == projection), None
    )
    values = row.get(projection)
    if struct_array is None or not values:
        return projection
    field = next(
        (item for item in struct_array.fields if item.dtype not in VECTOR_TYPES),
        struct_array.fields[0],
    )
    values[0][field.name] = _deterministic_update_value(
        field, pk * 1000, values[0].get(field.name)
    )
    return projection


def generate_primary_key_value(field: FieldSpec, pk: int) -> Any:
    if field.dtype in {"VARCHAR", "STRING"}:
        return f"pk_{pk:020d}"
    return pk


def _text_lob_boundary_value(pk: int) -> str:
    boundary_slot = pk % 1000
    if boundary_slot == 1:
        return ""
    if boundary_slot == 2:
        return "Milvus Unicode compatibility: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4"
    if boundary_slot == 3:
        return "a" * (64 * 1024 - 1)
    if boundary_slot == 4:
        return "b" * (64 * 1024)
    if boundary_slot == 5:
        return "c" * (64 * 1024 + 1)
    if boundary_slot == 6:
        return "d" * (1024 * 1024)
    return f"text lob document {pk} milvus upgrade rollback token_{pk % 16}"


def generate_field_value(field: FieldSpec, pk: int, seed: int) -> Any:
    if field.primary:
        return generate_primary_key_value(field, pk)
    if field.nullable and pk % 10 == 0:
        return None
    if field.dtype == "INT64":
        if field.name == "category" or field.is_partition_key:
            return pk % 1024
        return pk
    if field.dtype in {"INT32", "INT16", "INT8"}:
        return pk % 127
    if field.dtype == "FLOAT":
        return canonical_float32(float(pk % 1000) / 10.0)
    if field.dtype == "DOUBLE":
        return float(pk % 1000) / 10.0
    if field.dtype == "BOOL":
        return pk % 2 == 0
    if field.dtype in {"VARCHAR", "STRING", "TEXT"}:
        if field.value_profile == "text_lob_boundary":
            return _text_lob_boundary_value(pk)
        if field.value_profile == "minhash_documents":
            variants = (
                "the quick brown fox jumps over the lazy dog",
                "the quick brown fox jumps over a lazy dog",
                "distributed vector databases provide scalable similarity search",
            )
            return variants[pk % len(variants)]
        if field.name in {"text", "document"}:
            return (
                f"document {pk} milvus compatibility upgrade rollback token_{pk % 16}"
            )
        if field.is_partition_key:
            return f"tenant_{pk % 16}"
        return f"{field.name}_{pk}"
    if field.dtype == "JSON":
        if field.name == "json_bool":
            return {"active": pk % 2 == 0, "pk": pk}
        if field.name == "json_double":
            return {"score": float(pk % 1000) / 10.0, "pk": pk}
        if field.name == "json_varchar":
            return {"label": f"label_{pk % 8}", "pk": pk}
        if field.name == "json_auto":
            return {
                "active": pk % 2 == 0,
                "score": float(pk % 1000) / 10.0,
                "label": f"label_{pk % 8}",
                "bucket": pk % 16,
                "pk": pk,
            }
        if field.name == "json_nested":
            return {
                "pk": pk,
                "bucket": pk % 16,
                "nested": {
                    "score": float(pk % 1000) / 10.0,
                    "active": pk % 2 == 0,
                },
                "labels": [f"label_{pk % 8}", f"label_{(pk + 1) % 8}"],
            }
        return {"pk": pk, "bucket": pk % 16, "checksum": f"json_{pk}"}
    if field.dtype == "ARRAY":
        if field.element_type in {"INT64", "INT32", "INT16", "INT8"}:
            return [pk % 8, (pk + 1) % 8]
        if field.element_type == "FLOAT":
            return [
                canonical_float32(float(pk % 8)),
                canonical_float32(float((pk + 1) % 8)),
            ]
        if field.element_type == "DOUBLE":
            return [float(pk % 8), float((pk + 1) % 8)]
        if field.element_type == "BOOL":
            return [pk % 2 == 0, (pk + 1) % 2 == 0]
        return [f"tag_{pk % 8}", f"tag_{(pk + 1) % 8}"]
    if field.dtype in VECTOR_TYPES:
        return stable_vector_value(field, pk, seed)
    if field.dtype == "GEOMETRY":
        lon = -122.0 + (pk % 100) * 0.001
        lat = 37.0 + (pk % 100) * 0.001
        return f"POINT ({lon:g} {lat:g})"
    if field.dtype == "TIMESTAMPTZ":
        base = (
            datetime(2100, 1, 1, tzinfo=timezone.utc)
            if field.value_profile == "future_timestamptz"
            else datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        timestamp = base + timedelta(seconds=pk)
        return timestamp.isoformat().replace("+00:00", "Z")
    raise ValueError(f"Unsupported generated dtype: {field.dtype}")


def generate_struct_field_value(
    field: FieldSpec,
    pk: int,
    offset: int,
    seed: int,
) -> Any:
    nested_pk = pk * 1000 + offset
    if field.nullable and (pk + offset) % 10 == 0:
        return None
    if field.dtype in VECTOR_TYPES:
        return stable_vector_value(field, nested_pk, seed)
    if field.dtype == "FLOAT":
        return canonical_float32(float((pk % 1000) * 10 + offset) / 10.0)
    if field.dtype == "DOUBLE":
        return float((pk % 1000) * 10 + offset) / 10.0
    if field.dtype in {"INT64", "INT32", "INT16", "INT8"}:
        return pk * 10 + offset
    if field.dtype == "BOOL":
        return (pk + offset) % 2 == 0
    if field.dtype in {"VARCHAR", "STRING"}:
        if "category" in field.name:
            return f"category_{(pk + offset) % 8}"
        if "tag" in field.name:
            return f"tag_{(pk + offset) % 4}"
        return f"{field.name}_{pk}_{offset}"
    raise ValueError(f"Unsupported generated StructArray dtype: {field.dtype}")


def generate_struct_array_value(
    struct_array: StructArraySpec,
    pk: int,
    seed: int,
) -> list[dict[str, Any]] | None:
    if struct_array.nullable and pk % 10 == 0:
        return None
    length = min(struct_array.max_capacity, 1 + pk % 4)
    return [
        {
            field.name: generate_struct_field_value(field, pk, offset, seed)
            for field in struct_array.fields
        }
        for offset in range(length)
    ]


def generate_rows(
    spec: SchemaSpec, start_id: int, count: int, seed: int
) -> list[dict[str, Any]]:
    rows = []
    primary_fields = [field for field in spec.fields if field.primary]
    if len(primary_fields) != 1:
        raise ValueError(f"{spec.name}: expected exactly one primary field")
    function_outputs = function_output_fields(spec)
    for offset in range(count):
        pk = start_id + offset
        row = {}
        for field in spec.fields:
            if (field.primary and field.auto_id) or field.name in function_outputs:
                continue
            row[field.name] = generate_field_value(field, pk, seed)
        for struct_array in spec.struct_arrays:
            row[struct_array.name] = generate_struct_array_value(struct_array, pk, seed)
        if spec.enable_dynamic_field:
            row.update(generate_dynamic_fields(pk))
        rows.append(row)
    return rows


def generate_dynamic_fields(pk: int) -> dict[str, Any]:
    return {
        "dyn_bucket": pk % 32,
        "dyn_text": f"dynamic_{pk % 17}",
        "dyn_json": {"pk_mod": pk % 11, "active": pk % 2 == 0},
    }


def first_vector_field(spec: SchemaSpec) -> FieldSpec | None:
    for field in spec.fields:
        if field.dtype in VECTOR_TYPES:
            return field
    return None


def vector_fields(spec: SchemaSpec) -> list[FieldSpec]:
    return [field for field in spec.fields if field.dtype in VECTOR_TYPES]


def indexed_vector_fields(spec: SchemaSpec) -> list[tuple[str, FieldSpec]]:
    fields = []
    for index in spec.indexes:
        field = resolve_field(spec, index.field)
        if field is not None and field.dtype in VECTOR_TYPES:
            fields.append((index.field, field))
    return fields


def text_payload_metadata(value: str | None) -> dict[str, Any]:
    if value is None:
        return {"state": "null", "bytes": 0, "chars": 0, "sha256": None}
    encoded = value.encode("utf-8")
    return {
        "state": "empty" if value == "" else "value",
        "bytes": len(encoded),
        "chars": len(value),
        "prefix": value[:32],
        "suffix": value[-32:] if value else "",
        "sha256": sha256(encoded).hexdigest(),
    }
