from pathlib import Path

import pytest

from milvus_client.common.capability import load_capability_catalog
from milvus_client.common.schema import (
    FieldSpec,
    SchemaSpec,
    StructArraySpec,
    build_milvus_schema,
    create_collection_kwargs,
    load_feature_inventory,
    load_schema_matrix,
    resolve_field,
    rollback_incompatible_specs,
    validate_schema_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def test_schema_matrix_manifests_are_valid():
    features = load_feature_inventory(ROOT / "manifests" / "feature_inventory.yaml")
    capabilities = load_capability_catalog(
        ROOT / "manifests" / "capability_catalog.yaml"
    )

    for name in [
        "schema_matrix_2_6.yaml",
        "schema_matrix_3_0.yaml",
        "schema_matrix_3_0_storage_v3.yaml",
        "schema_matrix_3_0_index_v10_v4.yaml",
        "schema_matrix_json_shredding.yaml",
    ]:
        specs = load_schema_matrix(ROOT / "manifests" / name)
        errors = validate_schema_matrix(specs, features, set(capabilities))
        assert errors == []


@pytest.mark.parametrize("version", [None, "unknown"])
def test_load_schema_matrix_requires_parseable_version(tmp_path, version):
    matrix = tmp_path / "invalid-version.yaml"
    version_line = "" if version is None else f'version: "{version}"\n'
    matrix.write_text(
        f"""\
{version_line}schemas:
  - name: future_feature
    compat_mode: forward_only
    fields:
      - {{name: id, dtype: INT64, primary: true}}
"""
    )

    with pytest.raises(ValueError, match="version"):
        load_schema_matrix(matrix)


def test_rollback_compatibility_fails_closed_for_unknown_schema_version():
    spec = SchemaSpec(
        name="future_feature",
        version="unknown",
        compat_mode="forward_only",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
    )

    assert rollback_incompatible_specs([spec], "2.6.18") == [spec]


def test_schema_matrix_2_6_covers_expanded_rollback_safe_shapes():
    specs = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")

    assert [spec.name for spec in specs] == [
        "scalar_dynamic_partition_key",
        "vector_autoid_bm25",
        "explicit_partitions_nullable",
        "struct_array_element_rollback_safe",
        "nullable_vectors_all",
        "geometry_rtree_rollback_safe",
        "legacy_index_rollback_safe",
    ]
    assert any(spec.enable_dynamic_field for spec in specs)
    assert any(
        any(field.is_partition_key for field in spec.fields) and spec.num_partitions
        for spec in specs
    )
    assert any(spec.partitions == ["p0", "p1", "p2", "p3"] for spec in specs)
    assert any(
        any(field.primary and field.auto_id for field in spec.fields) for spec in specs
    )
    assert any(
        any(function.function_type == "BM25" for function in spec.functions)
        for spec in specs
    )

    dtypes = {field.dtype for spec in specs for field in spec.fields}
    assert {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "FLOAT",
        "DOUBLE",
        "BOOL",
        "VARCHAR",
        "JSON",
        "ARRAY",
        "FLOAT_VECTOR",
        "FLOAT16_VECTOR",
        "BFLOAT16_VECTOR",
        "INT8_VECTOR",
        "BINARY_VECTOR",
        "SPARSE_FLOAT_VECTOR",
        "GEOMETRY",
    }.issubset(dtypes)

    index_types = {index.index_type for spec in specs for index in spec.indexes}
    assert {
        "HNSW",
        "IVF_RABITQ",
        "DISKANN",
        "AUTOINDEX",
        "BIN_IVF_FLAT",
        "SPARSE_INVERTED_INDEX",
        "FLAT",
        "IVF_FLAT",
        "IVF_SQ8",
        "IVF_PQ",
        "SCANN",
        "BIN_FLAT",
        "SPARSE_WAND",
        "HNSW_SQ",
        "RTREE",
    }.issubset(index_types)
    assert {"STL_SORT", "INVERTED", "BITMAP", "TRIE", "NGRAM"}.issubset(index_types)
    nullable_vector_types = {
        field.dtype
        for spec in specs
        if spec.name == "nullable_vectors_all"
        for field in spec.fields
        if field.nullable
    }
    assert nullable_vector_types == {
        "FLOAT_VECTOR",
        "FLOAT16_VECTOR",
        "BFLOAT16_VECTOR",
        "INT8_VECTOR",
        "BINARY_VECTOR",
        "SPARSE_FLOAT_VECTOR",
    }
    assert any(spec.struct_arrays for spec in specs)


def test_schema_matrix_3_0_covers_forward_schema_evolution_shapes():
    specs = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")

    assert [spec.name for spec in specs] == [
        "nullable_vector",
        "geometry_rtree",
        "timestamptz_entity_ttl",
        "bm25_schema_evolution",
        "struct_array_scalar_indexes",
        "struct_array_float16_diskann",
        "faiss_float_binary",
        "minhash_lsh",
    ]
    dtypes = {field.dtype for spec in specs for field in spec.fields}
    assert {
        "FLOAT_VECTOR",
        "GEOMETRY",
        "TIMESTAMPTZ",
        "SPARSE_FLOAT_VECTOR",
        "VARCHAR",
    }.issubset(dtypes)
    assert any(
        any(field.nullable and field.dtype == "FLOAT_VECTOR" for field in spec.fields)
        for spec in specs
    )
    assert any(
        any(function.function_type == "BM25" for function in spec.functions)
        for spec in specs
    )
    assert any(
        any(index.index_type == "RTREE" for index in spec.indexes) for spec in specs
    )
    struct_spec = next(
        spec for spec in specs if spec.name == "struct_array_scalar_indexes"
    )
    struct_fields = {
        f"{struct_array.name}[{field.name}]": field.dtype
        for struct_array in struct_spec.struct_arrays
        for field in struct_array.fields
    }
    struct_indexes = {index.field: index.index_type for index in struct_spec.indexes}
    assert struct_fields["attributes[score_sort]"] == "FLOAT"
    assert struct_indexes["attributes[score_sort]"] == "STL_SORT"
    assert struct_fields["attributes[score_inverted]"] == "FLOAT"
    assert struct_indexes["attributes[score_inverted]"] == "INVERTED"
    assert struct_fields["attributes[category_inverted]"] == "VARCHAR"
    assert struct_indexes["attributes[category_inverted]"] == "INVERTED"
    assert struct_fields["attributes[tag_bitmap]"] == "VARCHAR"
    assert struct_indexes["attributes[tag_bitmap]"] == "BITMAP"


def test_storage_v3_and_index_version_matrices_cover_promoted_features():
    storage_specs = load_schema_matrix(
        ROOT / "manifests" / "schema_matrix_3_0_storage_v3.yaml"
    )
    index_specs = load_schema_matrix(
        ROOT / "manifests" / "schema_matrix_3_0_index_v10_v4.yaml"
    )

    text_spec = storage_specs[0]
    text_field = next(field for field in text_spec.fields if field.dtype == "TEXT")
    assert text_field.value_profile == "text_lob_boundary"
    assert {"text_lob_round_trip", "text_match_phrase_match"} <= set(
        text_spec.validators
    )

    index_types = {index.index_type for spec in index_specs for index in spec.indexes}
    algorithms = {
        index.params.get("inverted_index_algo")
        for spec in index_specs
        for index in spec.indexes
    }
    assert {"BITMAP", "STL_SORT", "NGRAM", "AUTOINDEX"} <= index_types
    assert {"SINDI", "BLOCK_MAX_MAXSCORE", "BLOCK_MAX_WAND"} <= algorithms
    assert any(
        index.expected_resolved_index_type == "HYBRID"
        for spec in index_specs
        for index in spec.indexes
    )


def test_json_shredding_schema_matrix_covers_nested_and_dynamic_json():
    specs = load_schema_matrix(ROOT / "manifests" / "schema_matrix_json_shredding.yaml")

    assert [spec.name for spec in specs] == ["json_shredding_nested"]
    spec = specs[0]
    assert spec.enable_dynamic_field is True
    assert {field.name for field in spec.fields if field.dtype == "JSON"} == {
        "json_profile",
        "json_nested",
    }
    assert any(
        index.field == "json_nested"
        and index.params.get("json_path") == "json_nested['nested']['score']"
        for index in spec.indexes
    )
    assert {"dyn_bucket", "dyn_text", "dyn_json"} <= set(spec.checksum_fields)


def test_schema_validation_rejects_invalid_partition_key_shapes():
    specs = [
        SchemaSpec(
            name="bad_partition_key_type",
            version="test",
            fields=[
                FieldSpec(name="id", dtype="INT64", primary=True),
                FieldSpec(name="bad_key", dtype="BOOL", is_partition_key=True),
            ],
        ),
        SchemaSpec(
            name="bad_num_partitions_without_key",
            version="test",
            fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
            num_partitions=4,
        ),
    ]

    errors = validate_schema_matrix(specs)

    assert (
        "bad_partition_key_type.bad_key: partition key field must be INT64 or VARCHAR"
        in errors
    )
    assert (
        "bad_num_partitions_without_key: num_partitions can only be specified when a partition key is defined"
        in errors
    )


def test_load_schema_matrix_supports_struct_arrays_and_index_options(tmp_path):
    matrix = tmp_path / "struct-array.yaml"
    matrix.write_text(
        """\
version: "3.0"
schemas:
  - name: struct_array_indexes
    properties: {ttl_field: event_time, timezone: UTC}
    fields:
      - {name: id, dtype: INT64, primary: true}
      - {name: event_time, dtype: TIMESTAMPTZ}
    struct_arrays:
      - name: attributes
        max_capacity: 8
        nullable: true
        fields:
          - {name: embedding, dtype: FLOAT_VECTOR, dim: 64}
          - {name: score, dtype: FLOAT}
          - {name: category, dtype: VARCHAR, max_length: 64}
    indexes:
      - field: attributes[embedding]
        index_type: HNSW
        index_name: attributes_embedding
        metric_type: MAX_SIM_COSINE
        params: {M: 8, efConstruction: 64}
        search_params: {ef: 32}
      - {field: "attributes[score]", index_type: STL_SORT}
"""
    )

    spec = load_schema_matrix(matrix)[0]

    assert validate_schema_matrix([spec]) == []
    assert spec.struct_arrays[0].name == "attributes"
    assert resolve_field(spec, "attributes[score]").dtype == "FLOAT"
    assert spec.indexes[0].index_name == "attributes_embedding"
    assert spec.indexes[0].search_params == {"ef": 32}
    assert create_collection_kwargs(spec) == {
        "properties": {"ttl_field": "event_time", "timezone": "UTC"}
    }
    schema = build_milvus_schema(spec)
    assert "attributes" in {field.name for field in schema.struct_fields}


def test_schema_validation_rejects_2_6_extended_struct_array_shapes():
    spec = SchemaSpec(
        name="invalid_2_6_struct",
        version="2.6",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="items",
                max_capacity=8,
                nullable=True,
                fields=[
                    FieldSpec(
                        name="embedding",
                        dtype="FLOAT16_VECTOR",
                        dim=64,
                    ),
                    FieldSpec(name="score", dtype="FLOAT", nullable=True),
                ],
            )
        ],
    )

    errors = validate_schema_matrix([spec])

    assert any("nullable StructArray requires Milvus 3.0" in error for error in errors)
    assert any("only supports FLOAT_VECTOR" in error for error in errors)
    assert any("nullable StructArray sub-field" in error for error in errors)
