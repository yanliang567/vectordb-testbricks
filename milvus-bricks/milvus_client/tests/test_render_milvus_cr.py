import json
import os
from pathlib import Path
import shutil
import subprocess
from copy import deepcopy

import pytest
import yaml

from milvus_client.common.deploy import (
    load_deploy_profile,
    render_milvus_cr,
    render_milvus_helm_values,
)
from milvus_client.requests import render_milvus_cr as render_cli
from milvus_client.requests import render_milvus_helm_values as render_helm_cli


ROOT = Path(__file__).resolve().parents[1]


def test_render_standalone_cr_from_profile():
    profile = load_deploy_profile(
        ROOT / "manifests" / "deploy_profiles" / "standalone-rocksmq.yaml"
    )

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
    assert (
        cr["spec"]["components"]["image"] == "harbor.milvus.io/milvusdb/milvus:v2.6.18"
    )
    assert cr["spec"]["components"]["version"] == "2.6.18"
    assert cr["spec"]["components"]["standalone"]["resources"]["requests"]["cpu"] == "2"
    assert cr["spec"]["config"]["common"]["storage"]["jsonShreddingEnabled"] is True
    assert cr["spec"]["dependencies"]["msgStreamType"] == "rocksmq"


def test_render_milvus_cr_rejects_helm_deploy_profile():
    profile = load_deploy_profile(
        ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml"
    )

    with pytest.raises(ValueError, match="deployer: operator"):
        render_milvus_cr(
            profile=profile,
            name="cluster-upgrade-test",
            namespace="qa-milvus",
            image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
            version="3.0.0",
            image_update_mode="all",
        )


def test_render_cluster_helm_values_from_profile_omits_3_0_storage_fields_by_default():
    profile = load_deploy_profile(
        ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml"
    )

    values = render_milvus_helm_values(
        profile=profile,
        name="cluster-upgrade-test",
        namespace="qa-milvus",
        image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
        version="3.0.0",
    )

    assert values["cluster"]["enabled"] is True
    assert values["woodpecker"]["enabled"] is True
    assert values["streaming"]["woodpecker"]["embedded"] is False
    assert values["image"]["all"]["repository"] == "harbor.milvus.io/milvusdb/milvus"
    assert values["image"]["all"]["tag"] == "3.0-latest"
    user_config = yaml.safe_load(values["extraConfigFiles"]["user.yaml"])
    assert user_config["common"]["storage"]["jsonShreddingEnabled"] is False
    assert "useLoonFFI" not in user_config["common"]["storage"]
    assert "storageV3Enabled" not in user_config["common"]["storage"]
    assert "dataNode" not in user_config


def test_render_cluster_helm_values_from_profile_emits_enabled_3_0_storage_fields():
    profile = load_deploy_profile(
        ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml"
    )

    values = render_milvus_helm_values(
        profile=profile,
        name="cluster-upgrade-test",
        namespace="qa-milvus",
        image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
        version="3.0.0",
        loon_ffi_enabled=True,
        vortex_enabled=True,
    )

    user_config = yaml.safe_load(values["extraConfigFiles"]["user.yaml"])
    assert user_config["common"]["storage"]["useLoonFFI"] is True
    assert "storageV3Enabled" not in user_config["common"]["storage"]
    assert user_config["dataNode"]["storage"]["format"] == "vortex"


def test_render_milvus_cr_cli_writes_yaml_and_topology_summary(tmp_path):
    output_yaml = tmp_path / "milvus.yaml"
    summary_json = tmp_path / "deploy_topology.json"

    rc = render_cli.main(
        [
            "--deploy-profile",
            str(ROOT / "manifests" / "deploy_profiles" / "standalone-rocksmq.yaml"),
            "--name",
            "standalone-upgrade-test",
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
    assert cr["spec"]["mode"] == "standalone"
    assert (
        cr["metadata"]["labels"]["app.kubernetes.io/name"]
        == "milvus-cluster-upgrade-rollback"
    )
    assert cr["metadata"]["annotations"]["zilliz.com/deploy-profile"].endswith(
        "standalone-rocksmq.yaml"
    )
    assert summary["profile"] == "standalone-rocksmq"
    assert summary["mode"] == "standalone"
    assert summary["components"]["standalone"]["replicas"] == 1


def test_render_milvus_helm_values_cli_writes_yaml_and_topology_summary(tmp_path):
    output_yaml = tmp_path / "values.yaml"
    summary_json = tmp_path / "deploy_topology.json"

    rc = render_helm_cli.main(
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

    values = yaml.safe_load(output_yaml.read_text())
    summary = json.loads(summary_json.read_text())
    assert rc == 0
    assert values["cluster"]["enabled"] is True
    assert "commonLabels" not in values
    assert "commonAnnotations" not in values
    assert (
        values["labels"]["zilliz.com/workflow-app"] == "milvus-cluster-upgrade-rollback"
    )
    assert "app.kubernetes.io/name" not in values["labels"]
    assert "app.kubernetes.io/managed-by" not in values["labels"]
    assert values["annotations"]["zilliz.com/deploy-profile"].endswith(
        "cluster-woodpecker-1cu.yaml"
    )
    assert summary["profile"] == "cluster-woodpecker-1cu"
    assert summary["mode"] == "cluster"
    assert summary["deployer"] == "helm"
    assert summary["helm"]["chart"] == "zilliztech/milvus"
    assert summary["helm"]["chart_version"] == "5.0.24"
    assert summary["components"]["queryNode"]["replicas"] == 1


def test_render_cluster_helm_values_rejects_chart_managed_labels():
    profile = load_deploy_profile(
        ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml"
    )

    with pytest.raises(ValueError, match="chart-managed selector labels"):
        render_milvus_helm_values(
            profile=profile,
            name="cluster-upgrade-test",
            namespace="qa-milvus",
            image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
            version="3.0.0",
            labels={
                "app.kubernetes.io/name": "milvus-cluster-upgrade-rollback",
                "zilliz.com/workflow-run-id": "uid-1",
            },
        )


def test_render_cluster_helm_values_rejects_profile_chart_managed_labels():
    profile = load_deploy_profile(
        ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml"
    )
    profile = deepcopy(profile)
    profile["helm_values"]["labels"] = {
        "app.kubernetes.io/name": "milvus-cluster-upgrade-rollback",
        "zilliz.com/workflow-run-id": "uid-1",
    }

    with pytest.raises(ValueError, match="chart-managed selector labels"):
        render_milvus_helm_values(
            profile=profile,
            name="cluster-upgrade-test",
            namespace="qa-milvus",
            image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
            version="3.0.0",
        )


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_rendered_cluster_helm_values_apply_metadata_to_chart_resources(tmp_path):
    profile_path = (
        ROOT / "manifests" / "deploy_profiles" / "cluster-woodpecker-1cu.yaml"
    )
    profile = load_deploy_profile(profile_path)
    output_yaml = tmp_path / "values.yaml"
    values = render_milvus_helm_values(
        profile=profile,
        name="cluster-upgrade-test",
        namespace="qa-milvus",
        image="harbor.milvus.io/milvusdb/milvus:3.0-latest",
        version="3.0.0",
        labels={
            "zilliz.com/workflow-app": "milvus-cluster-upgrade-rollback",
            "zilliz.com/workflow-run-id": "uid-1",
        },
        annotations={
            "zilliz.com/workflow-name": "wf-1",
            "zilliz.com/workflow-uid": "uid-1",
            "zilliz.com/deploy-profile": str(profile_path),
        },
    )
    output_yaml.write_text(yaml.safe_dump(values, sort_keys=False))

    env = os.environ.copy()
    env["HELM_CACHE_HOME"] = str(tmp_path / "helm-cache")
    env["HELM_CONFIG_HOME"] = str(tmp_path / "helm-config")
    env["HELM_DATA_HOME"] = str(tmp_path / "helm-data")
    helm = profile["helm"]
    subprocess.run(
        ["helm", "repo", "add", helm["repo_name"], helm["repo_url"]],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "cluster-upgrade-test",
            helm["chart"],
            "--version",
            str(helm["chart_version"]),
            "--namespace",
            "qa-milvus",
            "--values",
            str(output_yaml),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    docs = [doc for doc in yaml.safe_load_all(rendered.stdout) if doc]
    workloads = [
        doc
        for doc in docs
        if doc.get("kind") in {"Deployment", "StatefulSet"}
        and doc.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/instance")
        == "cluster-upgrade-test"
        and doc.get("metadata", {}).get("labels", {}).get("component")
        in {"datanode", "mixcoord", "proxy", "querynode", "streamingnode"}
    ]
    assert {workload["metadata"]["labels"]["component"] for workload in workloads} == {
        "datanode",
        "mixcoord",
        "proxy",
        "querynode",
        "streamingnode",
    }
    for workload in workloads:
        metadata = workload["metadata"]
        selector = workload["spec"]["selector"]["matchLabels"]
        pod_labels = workload["spec"]["template"]["metadata"]["labels"]
        assert selector.items() <= pod_labels.items()
        assert (
            metadata["labels"]["zilliz.com/workflow-app"]
            == "milvus-cluster-upgrade-rollback"
        )
        assert metadata["labels"]["zilliz.com/workflow-run-id"] == "uid-1"
        assert metadata["annotations"]["zilliz.com/workflow-uid"] == "uid-1"
