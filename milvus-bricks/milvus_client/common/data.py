from __future__ import annotations

from hashlib import sha256
import json
from random import Random
from typing import Any

from milvus_client.common.schema import FieldSpec, SchemaSpec, VECTOR_TYPES


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


def stable_binary_vector(seed: int, pk: int, dim: int) -> bytes:
    rng = Random(seed + pk)
    byte_count = max(1, dim // 8)
    return bytes(rng.getrandbits(8) for _ in range(byte_count))


def stable_sparse_vector(seed: int, pk: int, dim: int = 1024) -> dict[int, float]:
    rng = Random(seed + pk)
    return {rng.randint(0, dim - 1): rng.random() for _ in range(16)}


def _normalize_for_checksum(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {str(key): _normalize_for_checksum(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
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
        selected_rows.append(_normalize_for_checksum(selected))
    selected_rows.sort(key=lambda row: (row.get(primary_field) is None, row.get(primary_field)))
    for row in selected_rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode())
    return digest.hexdigest()


def checksum_fields_for_spec(spec: SchemaSpec) -> list[str]:
    return [field.name for field in spec.fields if field.dtype not in VECTOR_TYPES and not field.auto_id]


def generate_field_value(field: FieldSpec, pk: int, seed: int) -> Any:
    if field.primary:
        return pk
    if field.nullable and pk % 10 == 0:
        return None
    if field.dtype == "INT64":
        return pk % 1024 if field.name == "category" else pk
    if field.dtype in {"INT32", "INT16", "INT8"}:
        return pk % 127
    if field.dtype in {"FLOAT", "DOUBLE"}:
        return float(pk % 1000) / 10.0
    if field.dtype == "BOOL":
        return pk % 2 == 0
    if field.dtype in {"VARCHAR", "STRING", "TEXT"}:
        return f"{field.name}_{pk}"
    if field.dtype == "JSON":
        return {"pk": pk, "bucket": pk % 16, "checksum": f"json_{pk}"}
    if field.dtype == "ARRAY":
        return [f"tag_{pk % 8}", f"tag_{(pk + 1) % 8}"]
    if field.dtype == "FLOAT_VECTOR":
        return stable_float_vector(seed, pk, field.dim or 128)
    if field.dtype in {"FLOAT16_VECTOR", "BFLOAT16_VECTOR"}:
        return stable_float_vector(seed, pk, field.dim or 128)
    if field.dtype == "INT8_VECTOR":
        return stable_int8_vector(seed, pk, field.dim or 128)
    if field.dtype == "BINARY_VECTOR":
        return stable_binary_vector(seed, pk, field.dim or 128)
    if field.dtype == "SPARSE_FLOAT_VECTOR":
        return stable_sparse_vector(seed, pk)
    if field.dtype == "GEOMETRY":
        lon = -122.0 + (pk % 100) * 0.001
        lat = 37.0 + (pk % 100) * 0.001
        return f"POINT ({lon} {lat})"
    if field.dtype == "TIMESTAMPTZ":
        return 1_700_000_000_000 + pk
    raise ValueError(f"Unsupported generated dtype: {field.dtype}")


def generate_rows(spec: SchemaSpec, start_id: int, count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    primary_fields = [field for field in spec.fields if field.primary]
    if len(primary_fields) != 1:
        raise ValueError(f"{spec.name}: expected exactly one primary field")
    primary = primary_fields[0]
    for offset in range(count):
        pk = start_id + offset
        row = {}
        for field in spec.fields:
            if field.primary and field.auto_id:
                continue
            row[field.name] = generate_field_value(field, pk if field is not primary else pk, seed)
        rows.append(row)
    return rows


def first_vector_field(spec: SchemaSpec) -> FieldSpec | None:
    for field in spec.fields:
        if field.dtype in VECTOR_TYPES:
            return field
    return None
