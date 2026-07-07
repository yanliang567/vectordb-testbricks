from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    primary: bool = False
    auto_id: bool = False
    nullable: bool = False
    dim: int | None = None
    max_length: int | None = None
    element_type: str | None = None
    max_capacity: int | None = None


@dataclass(frozen=True)
class IndexSpec:
    field: str
    index_type: str
    metric_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    version: str
    fields: list[FieldSpec]
    indexes: list[IndexSpec] = field(default_factory=list)
    feature_tags: list[str] = field(default_factory=list)
    compat_mode: str = "rollback_safe"
    required_capabilities: list[str] = field(default_factory=list)
    validators: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    domain: str
    priority: str
    compat_mode: str
    required_capabilities: list[str] = field(default_factory=list)
    bricks: list[str] = field(default_factory=list)


VECTOR_TYPES = {
    "FLOAT_VECTOR",
    "FLOAT16_VECTOR",
    "BFLOAT16_VECTOR",
    "BINARY_VECTOR",
    "SPARSE_FLOAT_VECTOR",
    "INT8_VECTOR",
}
COMPAT_MODES = {"rollback_safe", "upgrade_only", "forward_only"}


def _as_field_spec(payload: dict[str, Any]) -> FieldSpec:
    return FieldSpec(
        name=payload["name"],
        dtype=payload["dtype"],
        primary=bool(payload.get("primary", False)),
        auto_id=bool(payload.get("auto_id", False)),
        nullable=bool(payload.get("nullable", False)),
        dim=payload.get("dim"),
        max_length=payload.get("max_length"),
        element_type=payload.get("element_type"),
        max_capacity=payload.get("max_capacity"),
    )


def _as_index_spec(payload: dict[str, Any]) -> IndexSpec:
    return IndexSpec(
        field=payload["field"],
        index_type=payload["index_type"],
        metric_type=payload.get("metric_type"),
        params=payload.get("params", {}),
    )


def load_schema_matrix(path: str | Path) -> list[SchemaSpec]:
    matrix_path = Path(path)
    payload = yaml.safe_load(matrix_path.read_text()) or {}
    version = str(payload.get("version", "unknown"))
    specs = []
    for item in payload.get("schemas", []):
        specs.append(
            SchemaSpec(
                name=item["name"],
                version=version,
                fields=[_as_field_spec(field) for field in item.get("fields", [])],
                indexes=[_as_index_spec(index) for index in item.get("indexes", [])],
                feature_tags=list(item.get("feature_tags", [])),
                compat_mode=item.get("compat_mode", "rollback_safe"),
                required_capabilities=list(item.get("required_capabilities", [])),
                validators=list(item.get("validators", [])),
            )
        )
    return specs


def load_feature_inventory(path: str | Path) -> dict[str, FeatureSpec]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    features = {}
    for item in payload.get("features", []):
        spec = FeatureSpec(
            id=item["id"],
            domain=item.get("domain", "unknown"),
            priority=item.get("priority", "P2"),
            compat_mode=item.get("compat_mode", "forward_only"),
            required_capabilities=list(item.get("required_capabilities", [])),
            bricks=list(item.get("bricks", [])),
        )
        features[spec.id] = spec
    return features


def validate_schema_matrix(
    specs: list[SchemaSpec],
    features: dict[str, FeatureSpec] | None = None,
    capabilities: set[str] | None = None,
) -> list[str]:
    errors = []
    names = set()
    for spec in specs:
        if spec.name in names:
            errors.append(f"duplicate schema name: {spec.name}")
        names.add(spec.name)

        if spec.compat_mode not in COMPAT_MODES:
            errors.append(f"{spec.name}: invalid compat_mode {spec.compat_mode}")

        primary_fields = [field for field in spec.fields if field.primary]
        if len(primary_fields) != 1:
            errors.append(f"{spec.name}: expected exactly one primary field")

        field_names = {field.name for field in spec.fields}
        for field_spec in spec.fields:
            if field_spec.dtype in VECTOR_TYPES and field_spec.dtype != "SPARSE_FLOAT_VECTOR" and not field_spec.dim:
                errors.append(f"{spec.name}.{field_spec.name}: vector field requires dim")
        for index in spec.indexes:
            if index.field not in field_names:
                errors.append(f"{spec.name}: index references unknown field {index.field}")

        if features is not None:
            for tag in spec.feature_tags:
                if tag not in features:
                    errors.append(f"{spec.name}: unknown feature tag {tag}")
        if capabilities is not None:
            for capability in spec.required_capabilities:
                if capability not in capabilities:
                    errors.append(f"{spec.name}: unknown capability {capability}")
    return errors


def dtype_to_milvus(dtype: str):
    from pymilvus import DataType

    if not hasattr(DataType, dtype):
        raise ValueError(f"Unsupported DataType: {dtype}")
    return getattr(DataType, dtype)


def build_milvus_schema(spec: SchemaSpec):
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=any(field.auto_id for field in spec.fields))
    for field_spec in spec.fields:
        kwargs: dict[str, Any] = {
            "field_name": field_spec.name,
            "datatype": dtype_to_milvus(field_spec.dtype),
            "is_primary": field_spec.primary,
            "nullable": field_spec.nullable,
        }
        if field_spec.dim is not None:
            kwargs["dim"] = field_spec.dim
        if field_spec.max_length is not None:
            kwargs["max_length"] = field_spec.max_length
        if field_spec.max_capacity is not None:
            kwargs["max_capacity"] = field_spec.max_capacity
        if field_spec.element_type is not None:
            kwargs["element_type"] = getattr(DataType, field_spec.element_type)
        schema.add_field(**kwargs)
    return schema


def build_index_params(spec: SchemaSpec):
    from pymilvus import MilvusClient

    index_params = MilvusClient.prepare_index_params()
    for index in spec.indexes:
        params = dict(index.params)
        if index.metric_type:
            params["metric_type"] = index.metric_type
        index_params.add_index(
            field_name=index.field,
            index_type=index.index_type,
            metric_type=index.metric_type,
            params=params or None,
        )
    return index_params


def collection_name(prefix: str, spec: SchemaSpec) -> str:
    return f"{prefix}_{spec.name}"

