from pathlib import Path

from milvus_client.requests.create_schema_matrix import run_dry_run


ROOT = Path(__file__).resolve().parents[1]


def test_create_schema_matrix_dry_run_loads_manifest():
    result = run_dry_run(
        str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
        str(ROOT / "manifests" / "feature_inventory.yaml"),
        str(ROOT / "manifests" / "capability_catalog.yaml"),
    )

    assert result["schemas_total"] > 0
    assert result["errors"] == []

