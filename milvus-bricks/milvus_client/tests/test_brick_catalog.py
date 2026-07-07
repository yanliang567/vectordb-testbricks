from importlib import import_module
from pathlib import Path

import yaml

from milvus_client.common.args import COMPAT_MODES, LIFECYCLE_PHASES
from milvus_client.common.capability import load_capability_catalog
from milvus_client.common.schema import load_feature_inventory


ROOT = Path(__file__).resolve().parents[1]


def test_brick_catalog_is_valid():
    catalog = yaml.safe_load((ROOT / "manifests" / "brick_catalog.yaml").read_text())
    features = load_feature_inventory(ROOT / "manifests" / "feature_inventory.yaml")
    capabilities = load_capability_catalog(ROOT / "manifests" / "capability_catalog.yaml")
    names = set()
    for brick in catalog["bricks"]:
        assert brick["name"] not in names
        names.add(brick["name"])
        import_module(brick["module"])
        assert brick["category"] in {"environment", "schema", "dml", "validation", "workload"}
        assert brick["milvus_versions"]
        assert brick["compat_mode"] in COMPAT_MODES
        for phase in brick["lifecycle_phases"]:
            assert phase in LIFECYCLE_PHASES
        for tag in brick["feature_tags"]:
            assert tag in features
        for capability in brick["required_capabilities"]:
            assert capability in capabilities

    for feature in features.values():
        for brick_name in feature.bricks:
            assert brick_name in names
