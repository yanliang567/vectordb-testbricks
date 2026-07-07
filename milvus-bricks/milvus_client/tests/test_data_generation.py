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


def test_checksum_fields_exclude_vectors():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    assert checksum_fields_for_spec(spec) == ["id", "category", "content", "flag"]
