from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "manifests" / "deploy_profiles"


def _load_profile(name: str) -> dict:
    return yaml.safe_load((PROFILE_DIR / name).read_text())


def test_deploy_profiles_are_valid_yaml():
    profile_names = {
        "standalone-rocksmq.yaml",
        "cluster-woodpecker-1cu.yaml",
        "cluster-woodpecker-2cu.yaml",
    }
    assert profile_names <= {path.name for path in PROFILE_DIR.glob("*.yaml")}
    for name in profile_names:
        profile = _load_profile(name)
        assert profile["name"]
        assert profile["mode"] in {"standalone", "cluster"}
        assert profile["components"]
        assert profile["dependencies"]["msgStreamType"]


def test_standalone_profile_declares_standalone_resources_and_in_cluster_dependencies():
    profile = _load_profile("standalone-rocksmq.yaml")

    assert profile["mode"] == "standalone"
    assert profile["dependencies"]["msgStreamType"] == "rocksmq"
    resources = profile["components"]["standalone"]["resources"]
    assert set(resources) == {"requests", "limits"}
    assert resources["requests"]["cpu"] == "2"
    assert resources["limits"]["memory"] == "8Gi"
    assert profile["dependencies"]["etcd"]["inCluster"]["deletionPolicy"] == "Delete"
    assert profile["dependencies"]["etcd"]["inCluster"]["pvcDeletion"] is True
    assert profile["dependencies"]["storage"]["inCluster"]["deletionPolicy"] == "Delete"
    assert profile["dependencies"]["storage"]["inCluster"]["pvcDeletion"] is True


def test_cluster_profiles_declare_required_components_resources_and_woodpecker():
    required_components = {"mixCoord", "proxy", "queryNode", "dataNode", "streamingNode"}

    for name in ["cluster-woodpecker-1cu.yaml", "cluster-woodpecker-2cu.yaml"]:
        profile = _load_profile(name)
        assert profile["mode"] == "cluster"
        assert profile["dependencies"]["msgStreamType"] == "woodpecker"
        assert required_components <= set(profile["components"])
        for component in required_components:
            spec = profile["components"][component]
            assert spec["replicas"] >= 1
            assert set(spec["resources"]) == {"requests", "limits"}
            assert spec["resources"]["requests"]["cpu"]
            assert spec["resources"]["requests"]["memory"]
            assert spec["resources"]["limits"]["cpu"]
            assert spec["resources"]["limits"]["memory"]
        assert profile["dependencies"]["etcd"]["inCluster"]["deletionPolicy"] == "Delete"
        assert profile["dependencies"]["storage"]["inCluster"]["pvcDeletion"] is True
