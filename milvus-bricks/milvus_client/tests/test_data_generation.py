from pathlib import Path

from milvus_client.common.data import generate_rows
from milvus_client.common.schema import load_schema_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_generate_rows_is_deterministic():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]
    rows1 = generate_rows(spec, start_id=0, count=10, seed=7)
    rows2 = generate_rows(spec, start_id=0, count=10, seed=7)

    assert rows1 == rows2
    assert rows1[0]["id"] == 0
    assert "embedding" in rows1[0]

