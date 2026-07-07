from pathlib import Path

from milvus_client.common.data import checksum_fields_for_spec, generate_rows, stable_checksum
from milvus_client.common.schema import load_schema_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_generate_rows_is_deterministic():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]
    rows1 = generate_rows(spec, start_id=0, count=10, seed=7)
    rows2 = generate_rows(spec, start_id=0, count=10, seed=7)

    assert rows1 == rows2
    assert rows1[0]["id"] == 0
    assert "embedding" in rows1[0]


def test_stable_checksum_uses_selected_fields_and_is_order_independent():
    rows = [
        {"id": 2, "category": 20, "embedding": [0.2]},
        {"id": 1, "category": 10, "embedding": [0.1]},
    ]
    reordered_rows = list(reversed(rows))

    checksum = stable_checksum(rows, fields=["id", "category"], primary_field="id")

    assert checksum == stable_checksum(reordered_rows, fields=["id", "category"], primary_field="id")
    assert checksum != stable_checksum(rows, fields=["id", "category", "embedding"], primary_field="id")


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


def test_checksum_fields_exclude_vectors():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    assert checksum_fields_for_spec(spec) == ["id", "category", "content", "flag"]


def test_generate_rows_uses_sdk_compatible_vector_values():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[1]
    row = generate_rows(spec, start_id=1, count=1, seed=7)[0]

    assert row["float16_vec"].dtype == "float16"
    assert isinstance(row["bfloat16_vec"], bytes)
    assert len(row["bfloat16_vec"]) == 256
    assert row["int8_vec"].dtype == "int8"


def test_generate_rows_uses_timestamptz_string():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")[2]
    row = generate_rows(spec, start_id=1, count=1, seed=7)[0]

    assert row["event_time"] == "2024-01-01T00:00:01Z"


def test_generate_rows_uses_canonical_geometry_wkt():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")[1]
    row = generate_rows(spec, start_id=0, count=1, seed=7)[0]

    assert row["location"] == "POINT (-122 37)"
