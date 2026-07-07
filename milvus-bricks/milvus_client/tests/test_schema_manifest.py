from pathlib import Path

from milvus_client.common.capability import load_capability_catalog
from milvus_client.common.schema import load_feature_inventory, load_schema_matrix, validate_schema_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_schema_matrix_manifests_are_valid():
    features = load_feature_inventory(ROOT / "manifests" / "feature_inventory.yaml")
    capabilities = load_capability_catalog(ROOT / "manifests" / "capability_catalog.yaml")

    for name in ["schema_matrix_2_6.yaml", "schema_matrix_3_0.yaml"]:
        specs = load_schema_matrix(ROOT / "manifests" / name)
        errors = validate_schema_matrix(specs, features, set(capabilities))
        assert errors == []

