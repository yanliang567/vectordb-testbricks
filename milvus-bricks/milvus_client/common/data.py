from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from random import Random
from typing import Any

from milvus_client.common.schema import FieldSpec, SchemaSpec, VECTOR_TYPES, function_output_fields


def stable_float_vector(seed: int, pk: int, dim: int) -> list[float]:
    rng = Random(seed + pk)
    values = [rng.random() for _ in range(dim)]
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        return values
    return [value / norm for value in values]


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


def _normalize_for_checksum(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(key): _normalize_for_checksum(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
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
        selected_rows.append((sort_value is None, sort_value, _normalize_for_checksum(selected)))
    selected_rows.sort(key=lambda item: (item[0], item[1]))
    for _, _, selected in selected_rows:
        digest.update(json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str).encode())
    return digest.hexdigest()


def checksum_fields_for_spec(spec: SchemaSpec) -> list[str]:
    function_outputs = function_output_fields(spec)
    return [
        field.name
        for field in spec.fields
        if field.dtype not in VECTOR_TYPES and not field.auto_id and field.name not in function_outputs
    ]


def generate_primary_key_value(field: FieldSpec, pk: int) -> Any:
    if field.dtype in {"VARCHAR", "STRING"}:
        return f"pk_{pk:020d}"
    return pk


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
    if field.dtype in {"FLOAT", "DOUBLE"}:
        return float(pk % 1000) / 10.0
    if field.dtype == "BOOL":
        return pk % 2 == 0
    if field.dtype in {"VARCHAR", "STRING", "TEXT"}:
        if field.name in {"text", "document"}:
            return f"document {pk} milvus compatibility upgrade rollback token_{pk % 16}"
        if field.is_partition_key:
            return f"tenant_{pk % 16}"
        return f"{field.name}_{pk}"
    if field.dtype == "JSON":
        return {"pk": pk, "bucket": pk % 16, "checksum": f"json_{pk}"}
    if field.dtype == "ARRAY":
        if field.element_type in {"INT64", "INT32", "INT16", "INT8"}:
            return [pk % 8, (pk + 1) % 8]
        if field.element_type in {"FLOAT", "DOUBLE"}:
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
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=pk)
        return timestamp.isoformat().replace("+00:00", "Z")
    raise ValueError(f"Unsupported generated dtype: {field.dtype}")


def generate_rows(spec: SchemaSpec, start_id: int, count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    primary_fields = [field for field in spec.fields if field.primary]
    if len(primary_fields) != 1:
        raise ValueError(f"{spec.name}: expected exactly one primary field")
    primary = primary_fields[0]
    function_outputs = function_output_fields(spec)
    for offset in range(count):
        pk = start_id + offset
        row = {}
        for field in spec.fields:
            if (field.primary and field.auto_id) or field.name in function_outputs:
                continue
            row[field.name] = generate_field_value(field, pk if field is not primary else pk, seed)
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
