import json
from pathlib import Path

import yaml

from milvus_client.common.deploy import load_deploy_profile, render_milvus_cr
from milvus_client.requests import render_milvus_cr as render_cli


ROOT = Path(__file__).resolve().parents[1]


def test_render_standalone_cr_from_profile():
    profile = load_deploy_profile(ROOT / "manifests" / "deploy_profiles" / "standalone-rocksmq.yaml")

    cr = render_milvus_cr(
        profile=profile,
        name="upgrade-test",
        namespace="qa-milvus",
        image="harbor.milvus.io/milvusdb/milvus:v2.6.18",
        version="2.6.18",
        image_update_mode="all",
        json_shredding_enabled=True,
        labels={"zilliz.com/workflow-run-id": "uid-1"},
        annotations={"zilliz.com/workflow-name": "wf-1"},
    )

    assert cr["kind"] == "Milvus"
    assert cr["metadata"]["name"] == "upgrade-test"
    assert cr["spec"]["mode"] == "standalone"
    assert cr["spec"]["components"]["image"] == "harbor.milvus.io/milvusdb/milvus:v2.6.18"
    assert cr["spec"]["components"]["version"] == "2.6.18"
    assert cr["spec"]["components"]["standalone"]["resources"]["requests"]["cpu"] == "2"
    assert cr["spec"]["config"]["common"]["storage"]["jsonShreddingEnabled"] is True
    assert cr["spec"]["dependencies"]["msgStreamType"] == "rocksmq"


def test_render_cluster_cr_from_profile_omits_3_0_storage_fields_by_default():
    profile = load_deploy_profile(ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml")

    cr = render_milvus_cr(
        profile=profile,
        name="cluster-upgrade-test",
        namespace="qa-milvus",
        image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
        version="3.0.0",
        image_update_mode="all",
    )

    assert cr["spec"]["mode"] == "cluster"
    assert cr["spec"]["dependencies"]["msgStreamType"] == "woodpecker"
    for component in ["mixCoord", "proxy", "queryNode", "dataNode", "streamingNode"]:
        assert cr["spec"]["components"][component]["replicas"] >= 1
        assert cr["spec"]["components"][component]["resources"]["requests"]["cpu"]
    assert "useLoonFFI" not in cr["spec"]["config"]["common"]["storage"]
    assert "storageV3Enabled" not in cr["spec"]["config"]["common"]["storage"]
    assert "dataNode" not in cr["spec"]["config"]


def test_render_cluster_cr_from_profile_emits_enabled_3_0_storage_fields():
    profile = load_deploy_profile(ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml")

    cr = render_milvus_cr(
        profile=profile,
        name="cluster-upgrade-test",
        namespace="qa-milvus",
        image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
        version="3.0.0",
        image_update_mode="all",
        loon_ffi_enabled=True,
        vortex_enabled=True,
    )

    assert cr["spec"]["config"]["common"]["storage"]["useLoonFFI"] is True
    assert "storageV3Enabled" not in cr["spec"]["config"]["common"]["storage"]
    assert cr["spec"]["config"]["dataNode"]["storage"]["format"] == "vortex"


def test_render_milvus_cr_cli_writes_yaml_and_topology_summary(tmp_path):
    output_yaml = tmp_path / "milvus.yaml"
    summary_json = tmp_path / "deploy_topology.json"

    rc = render_cli.main(
        [
            "--deploy-profile",
            str(ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml"),
            "--name",
            "cluster-upgrade-test",
            "--namespace",
            "qa-milvus",
            "--image",
            "harbor.milvus.io/milvusdb/milvus:3.0-latest",
            "--version",
            "3.0.0",
            "--image-update-mode",
            "all",
            "--workflow-name",
            "wf-1",
            "--workflow-uid",
            "uid-1",
            "--app-name",
            "milvus-cluster-upgrade-rollback",
            "--output-yaml",
            str(output_yaml),
            "--summary-json",
            str(summary_json),
        ]
    )

    cr = yaml.safe_load(output_yaml.read_text())
    summary = json.loads(summary_json.read_text())
    assert rc == 0
    assert cr["spec"]["mode"] == "cluster"
    assert cr["metadata"]["labels"]["app.kubernetes.io/name"] == "milvus-cluster-upgrade-rollback"
    assert cr["metadata"]["annotations"]["zilliz.com/deploy-profile"].endswith("cluster-woodpecker-1cu.yaml")
    assert summary["profile"] == "cluster-woodpecker-1cu"
    assert summary["mode"] == "cluster"
    assert summary["components"]["queryNode"]["replicas"] == 1
