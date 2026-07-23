import json
from pathlib import Path

import yaml
from milvus_client.common.gates import (
    load_gate_manifest,
    render_submission,
    resolve_gate_scenario,
)
from milvus_client.requests import render_upgrade_rollback_params as render_cli


ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "manifests" / "upgrade_rollback_gates.yaml"


def test_render_standalone_2_6_to_3_0_gate_parameters():
    manifest = load_gate_manifest(GATES)
    scenario = resolve_gate_scenario(
        manifest,
        "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
    )

    submission = render_submission(scenario, manifest, allow_placeholder=True)
    params = submission["parameters"]

    assert submission["workflow_template"] == "milvus-standalone-2-6-upgrade-rollback"
    assert (
        params["scenario-id"] == "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest"
    )
    assert (
        params["deploy-profile"]
        == "milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml"
    )
    assert params["base-milvus-image"] == "harbor.milvus.io/milvusdb/milvus:v2.6.18"
    assert (
        params["target-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:3.0-latest-placeholder"
    )
    assert (
        params["rollback-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:2.6-latest-placeholder"
    )
    assert params["base-version"] == "2.6.18"
    assert params["target-version"] == "3.0.0"
    assert params["rollback-version"] == "2.6.0"
    assert params["schema-matrix"] == "milvus_client/manifests/schema_matrix_2_6.yaml"
    assert params["forward-workload-enabled"] == "false"
    assert params["rollback-forward-validation-enabled"] == "false"
    assert params["index-compatibility-validation-enabled"] == "true"
    assert params["schema-evolution-existing-enabled"] == "false"
    assert params["target-loon-ffi-enabled"] == "false"
    assert params["target-json-shredding-enabled"] == "false"
    assert params["base-loon-ffi-enabled"] == "false"
    assert params["rollback-loon-ffi-enabled"] == "false"
    assert params["base-vortex-enabled"] == "false"
    assert params["target-vortex-enabled"] == "false"
    assert params["rollback-vortex-enabled"] == "false"
    assert params["pressure-fail-on-error"] == "true"
    assert params["gate-allow-warning"] == "false"
    assert params["allow-unsafe-negative-coverage"] == "false"
    assert params["rows-per-collection"] == "5000"


def test_render_cluster_3_0_gate_parameters():
    manifest = load_gate_manifest(GATES)
    scenario = resolve_gate_scenario(
        manifest,
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
    )

    submission = render_submission(scenario, manifest, allow_placeholder=True)
    params = submission["parameters"]

    assert submission["workflow_template"] == "milvus-cluster-upgrade-rollback"
    assert (
        params["deploy-profile"]
        == "milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml"
    )
    assert (
        params["base-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:3.0-baseline-placeholder"
    )
    assert (
        params["target-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:3.0-latest-placeholder"
    )
    assert (
        params["rollback-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:3.0-baseline-placeholder"
    )
    assert params["schema-matrix"] == "milvus_client/manifests/schema_matrix_3_0.yaml"
    assert params["schema-evolution-existing-enabled"] == "true"
    assert params["rollback-forward-validation-enabled"] == "true"
    assert params["index-compatibility-validation-enabled"] == "true"


def test_render_cluster_2_6_to_3_0_gate_uses_pulsar_profile():
    manifest = load_gate_manifest(GATES)
    scenario = resolve_gate_scenario(
        manifest,
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest",
    )

    submission = render_submission(scenario, manifest, allow_placeholder=True)
    params = submission["parameters"]

    assert submission["workflow_template"] == "milvus-cluster-upgrade-rollback"
    assert (
        params["deploy-profile"]
        == "milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml"
    )
    assert params["schema-matrix"] == "milvus_client/manifests/schema_matrix_2_6.yaml"
    assert params["index-compatibility-validation-enabled"] == "true"
    assert params["base-milvus-image"] == "harbor.milvus.io/milvusdb/milvus:v2.6.18"
    assert (
        params["rollback-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:2.6-latest-placeholder"
    )


def test_render_negative_scenario_enables_explicit_unsafe_coverage_escape_hatch():
    manifest = load_gate_manifest(GATES)
    scenario = resolve_gate_scenario(
        manifest,
        "standalone-3-0-loon-vortex-to-2-6-negative",
    )

    submission = render_submission(scenario, manifest, allow_placeholder=True)
    params = submission["parameters"]

    assert scenario["classification"] == "negative"
    assert params["allow-unsafe-negative-coverage"] == "true"
    assert params["target-loon-ffi-enabled"] == "true"
    assert params["target-vortex-enabled"] == "true"
    assert params["gate-allow-warning"] == "true"


def test_render_params_cli_writes_json(tmp_path):
    output = tmp_path / "params.json"

    rc = render_cli.main(
        [
            "--manifest",
            str(GATES),
            "--scenario-id",
            "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
            "--format",
            "json",
            "--allow-placeholder",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert rc == 0
    assert payload["workflow_template"] == "milvus-standalone-2-6-upgrade-rollback"
    assert (
        payload["parameters"]["rollback-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:2.6-latest-placeholder"
    )


def test_render_params_cli_writes_yaml_and_supports_deploy_profile_override(tmp_path):
    output = tmp_path / "params.yaml"

    rc = render_cli.main(
        [
            "--manifest",
            str(GATES),
            "--scenario-id",
            "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
            "--deploy-profile",
            "milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml",
            "--format",
            "yaml",
            "--allow-placeholder",
            "--output",
            str(output),
        ]
    )

    payload = yaml.safe_load(output.read_text())
    assert rc == 0
    assert payload["workflow_template"] == "milvus-cluster-upgrade-rollback"
    assert payload["parameters"]["deploy-profile"] == (
        "milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml"
    )


def test_render_params_cli_writes_argo_args(tmp_path):
    output = tmp_path / "params.args"

    rc = render_cli.main(
        [
            "--manifest",
            str(GATES),
            "--scenario-id",
            "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
            "--format",
            "argo-args",
            "--allow-placeholder",
            "--output",
            str(output),
        ]
    )

    args = output.read_text()
    assert rc == 0
    assert "--from workflowtemplate/milvus-standalone-3-0-upgrade-rollback" in args
    assert (
        "-p scenario-id=standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline"
        in args
    )
    assert (
        "-p target-milvus-image=harbor.milvus.io/milvusdb/milvus:3.0-latest-placeholder"
        in args
    )


def test_render_params_cli_rejects_placeholder_images_for_promoted_gate(tmp_path):
    output = tmp_path / "params.json"

    rc = render_cli.main(
        [
            "--manifest",
            str(GATES),
            "--scenario-id",
            "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    assert rc == 2
    assert not output.exists()


def test_render_params_cli_rejects_placeholder_images_for_negative_scenario(tmp_path):
    output = tmp_path / "params.json"

    rc = render_cli.main(
        [
            "--manifest",
            str(GATES),
            "--scenario-id",
            "standalone-3-0-loon-vortex-to-2-6-negative",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    assert rc == 2
    assert not output.exists()
