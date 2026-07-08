from pathlib import Path

from milvus_client.common.capability import load_capability_catalog
from milvus_client.common.schema import FieldSpec, SchemaSpec, load_feature_inventory, load_schema_matrix, validate_schema_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_schema_matrix_manifests_are_valid():
    features = load_feature_inventory(ROOT / "manifests" / "feature_inventory.yaml")
    capabilities = load_capability_catalog(ROOT / "manifests" / "capability_catalog.yaml")

    for name in ["schema_matrix_2_6.yaml", "schema_matrix_3_0.yaml"]:
        specs = load_schema_matrix(ROOT / "manifests" / name)
        errors = validate_schema_matrix(specs, features, set(capabilities))
        assert errors == []


def test_schema_matrix_2_6_covers_expanded_rollback_safe_shapes():
    specs = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")

    assert [spec.name for spec in specs] == [
        "scalar_dynamic_partition_key",
        "vector_autoid_bm25",
        "explicit_partitions_nullable",
    ]
    assert any(spec.enable_dynamic_field for spec in specs)
    assert any(any(field.is_partition_key for field in spec.fields) and spec.num_partitions for spec in specs)
    assert any(spec.partitions == ["p0", "p1", "p2", "p3"] for spec in specs)
    assert any(any(field.primary and field.auto_id for field in spec.fields) for spec in specs)
    assert any(any(function.function_type == "BM25" for function in spec.functions) for spec in specs)

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
    }.issubset(dtypes)

    index_types = {index.index_type for spec in specs for index in spec.indexes}
    assert {"HNSW", "IVF_RABITQ", "DISKANN", "AUTOINDEX", "BIN_IVF_FLAT", "SPARSE_INVERTED_INDEX"}.issubset(
        index_types
    )
    assert {"STL_SORT", "INVERTED", "BITMAP", "TRIE", "NGRAM"}.issubset(index_types)


def test_schema_matrix_3_0_covers_forward_schema_evolution_shapes():
    specs = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")

    assert [spec.name for spec in specs] == [
        "nullable_vector",
        "geometry_rtree",
        "timestamptz_ttl",
        "bm25_schema_evolution",
    ]
    dtypes = {field.dtype for spec in specs for field in spec.fields}
    assert {"FLOAT_VECTOR", "GEOMETRY", "TIMESTAMPTZ", "SPARSE_FLOAT_VECTOR", "VARCHAR"}.issubset(dtypes)
    assert any(any(field.nullable and field.dtype == "FLOAT_VECTOR" for field in spec.fields) for spec in specs)
    assert any(any(function.function_type == "BM25" for function in spec.functions) for spec in specs)
    assert any(any(index.index_type == "RTREE" for index in spec.indexes) for spec in specs)


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

    assert "bad_partition_key_type.bad_key: partition key field must be INT64 or VARCHAR" in errors
    assert "bad_num_partitions_without_key: num_partitions can only be specified when a partition key is defined" in errors
