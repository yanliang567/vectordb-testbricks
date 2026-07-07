from pathlib import Path

from milvus_client.common.data import checksum_fields_for_spec, generate_rows, stable_checksum
from milvus_client.common.schema import FieldSpec, SchemaSpec, load_schema_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_generate_rows_is_deterministic():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]
    rows1 = generate_rows(spec, start_id=0, count=10, seed=7)
    rows2 = generate_rows(spec, start_id=0, count=10, seed=7)

    assert rows1 == rows2
    assert rows1[0]["id"] == 0
    assert "embedding" in rows1[0]
    assert rows1[0]["dyn_bucket"] == 0


def test_stable_checksum_uses_selected_fields_and_is_order_independent():
    rows = [
        {"id": 2, "category": 20, "embedding": [0.2]},
        {"id": 1, "category": 10, "embedding": [0.1]},
    ]
    reordered_rows = list(reversed(rows))

    checksum = stable_checksum(rows, fields=["id", "category"], primary_field="id")

    assert checksum == stable_checksum(reordered_rows, fields=["id", "category"], primary_field="id")
    assert checksum != stable_checksum(rows, fields=["id", "category", "embedding"], primary_field="id")


def test_stable_checksum_sorts_by_primary_even_when_primary_is_not_digested():
    rows = [
        {"id": 2, "category": 20},
        {"id": 1, "category": 10},
    ]
    queried_rows = list(reversed(rows))

    assert stable_checksum(rows, fields=["category"], primary_field="id") == stable_checksum(
        queried_rows,
        fields=["category"],
        primary_field="id",
    )


def test_stable_checksum_normalizes_repeated_scalar_containers():
    class RepeatedScalarLike:
        def __iter__(self):
            return iter(["tag_0", "tag_1"])

    rows = [{"id": 1, "tags": RepeatedScalarLike()}]
    list_rows = [{"id": 1, "tags": ["tag_0", "tag_1"]}]

    assert stable_checksum(rows, fields=["id", "tags"], primary_field="id") == stable_checksum(
        list_rows,
        fields=["id", "tags"],
        primary_field="id",
    )


def test_stable_checksum_normalizes_float32_round_trip_precision():
    inserted_rows = [{"id": 1, "score": 0.1}]
    queried_rows = [{"id": 1, "score": 0.10000000149011612}]

    assert stable_checksum(inserted_rows, fields=["id", "score"], primary_field="id") == stable_checksum(
        queried_rows,
        fields=["id", "score"],
        primary_field="id",
    )


def test_checksum_fields_exclude_vectors():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    assert "id" in checksum_fields_for_spec(spec)
    assert "embedding" not in checksum_fields_for_spec(spec)
    assert "dyn_bucket" not in checksum_fields_for_spec(spec)


def test_generate_rows_uses_sdk_compatible_vector_values():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[1]
    row = generate_rows(spec, start_id=1, count=1, seed=7)[0]

    assert "id" not in row
    assert row["float16_hnsw"].dtype == "float16"
    assert isinstance(row["bfloat16_diskann"], bytes)
    assert len(row["bfloat16_diskann"]) == 256
    assert row["int8_autoindex"].dtype == "int8"
    assert isinstance(row["binary_ivf"], bytes)
    assert "sparse_bm25" not in row


def test_generate_rows_uses_timestamptz_string():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")[2]
    row = generate_rows(spec, start_id=1, count=1, seed=7)[0]

    assert row["event_time"] == "2024-01-01T00:00:01Z"


def test_generate_rows_uses_canonical_geometry_wkt():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")[1]
    row = generate_rows(spec, start_id=0, count=1, seed=7)[0]

    assert row["location"] == "POINT (-122 37)"


def test_generate_rows_supports_string_primary_key():
    spec = SchemaSpec(
        name="string_pk",
        version="test",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    row = generate_rows(spec, start_id=7, count=1, seed=1)[0]

    assert row["pk"] == "pk_00000000000000000007"


def test_generate_rows_supports_numeric_arrays():
    spec = SchemaSpec(
        name="numeric_array",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="ints", dtype="ARRAY", element_type="INT64"),
            FieldSpec(name="floats", dtype="ARRAY", element_type="FLOAT"),
            FieldSpec(name="bools", dtype="ARRAY", element_type="BOOL"),
        ],
    )

    row = generate_rows(spec, start_id=3, count=1, seed=1)[0]

    assert row["ints"] == [3, 4]
    assert row["floats"] == [3.0, 4.0]
    assert row["bools"] == [False, True]


def test_generate_rows_caps_int64_partition_key_by_field_attribute():
    spec = SchemaSpec(
        name="int64_partition_key",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="tenant_id", dtype="INT64", is_partition_key=True),
        ],
    )

    row = generate_rows(spec, start_id=2049, count=1, seed=1)[0]

    assert row["tenant_id"] == 1
