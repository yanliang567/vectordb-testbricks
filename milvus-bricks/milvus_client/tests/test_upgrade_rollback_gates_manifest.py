from copy import deepcopy
from pathlib import Path

import pytest

from milvus_client.common.gates import (
    load_gate_manifest,
    render_argo_parameters,
    resolve_gate_scenario,
    validate_gate_manifest,
    validate_resolved_gate_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "manifests" / "upgrade_rollback_gates.yaml"
MILVUS_3_0_BASELINE_IMAGE = (
    "harbor.milvus.io/milvusdb/milvus:v3.0.0@"
    "sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862"
)


def _manifest() -> dict:
    return load_gate_manifest(GATES)


def test_upgrade_rollback_gates_manifest_contains_required_gate_scenarios():
    manifest = _manifest()
    assert manifest["defaults"]["index_compatibility_validation_enabled"] is True
    assert manifest["defaults"]["phase_dml_dql_validation_enabled"] is True
    assert manifest["defaults"]["phase_new_collection_rows"] == 3000
    assert manifest["defaults"]["phase_existing_dml_rows"] == 1000
    assert manifest["defaults"]["phase_existing_delete_rows"] == 100
    scenarios = {
        scenario["id"]: resolve_gate_scenario(manifest, scenario["id"])
        for scenario in manifest["scenarios"]
    }

    assert {
        "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
        "standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline",
    } <= set(scenarios)
    for scenario_id in [
        "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
        "standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline",
    ]:
        scenario = scenarios[scenario_id]
        assert scenario["classification"] == "gate"
        assert scenario["workflow_template"].startswith("milvus-")
        assert scenario["deploy_profile"].endswith(".yaml")
        assert scenario["schema_matrix"].endswith(".yaml")
        for phase in ["base", "target", "rollback"]:
            assert scenario[phase]["image"].startswith(
                "harbor.milvus.io/milvusdb/milvus:"
            )
            assert scenario[phase]["version"]
            assert "image_ref" in scenario[phase]
        assert scenario["validation_policy"]["data_integrity"] == "strict"
        assert scenario["validation_policy"]["serviceability"] == "strict"
        assert scenario["validation_policy"]["pressure_fail_on_error"] is True
        assert scenario["validation_policy"]["gate_allow_warning"] is False
        assert scenario["index_compatibility_validation_enabled"] is True


def test_standalone_2_6_target_only_feature_gate_contract():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
    )

    assert scenario["workflow_template"] == ("milvus-standalone-2-6-upgrade-rollback")
    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6.yaml"
    )
    assert scenario["forward_schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_3_0.yaml"
    )
    assert scenario["forward_workload_enabled"] is True
    assert scenario["schema_evolution_existing_enabled"] is False
    assert scenario["schema_evolution_forward_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is False
    assert scenario["base"]["version"].startswith("2.6")
    assert scenario["target"]["version"].startswith("3.0")
    assert scenario["rollback"]["version"].startswith("2.6")


@pytest.mark.parametrize(
    "scenario_id",
    [
        "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
    ],
)
def test_2_6_target_only_gates_reject_renamed_forward_only_matrix_after_rollback(
    tmp_path,
    scenario_id,
):
    manifest = _manifest()
    broken = deepcopy(manifest)
    renamed_matrix = tmp_path / "custom_forward_features.yaml"
    renamed_matrix.write_text(
        (ROOT / "manifests" / "schema_matrix_3_0.yaml").read_text()
    )
    scenario = next(item for item in broken["scenarios"] if item["id"] == scenario_id)
    scenario["forward_schema_matrix"] = str(renamed_matrix)
    scenario["rollback_forward_validation_enabled"] = True

    with pytest.raises(
        ValueError,
        match="forward schemas cannot be required after rollback to 2.6.0",
    ):
        resolve_gate_scenario(broken, scenario["id"])


def test_target_only_gate_ignores_forward_rollback_contract_when_rollback_disabled():
    manifest = _manifest()
    scenario = next(
        item
        for item in manifest["scenarios"]
        if item["id"]
        == "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest"
    )
    scenario["rollback_enabled"] = False
    scenario["rollback_forward_validation_enabled"] = True

    resolved = resolve_gate_scenario(manifest, scenario["id"])

    assert resolved["rollback_enabled"] is False
    assert resolved["rollback_forward_validation_enabled"] is True


def test_target_only_gate_rejects_future_matrix_after_older_rollback(tmp_path):
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
    )
    future_matrix = tmp_path / "future_capabilities.yaml"
    future_matrix.write_text(
        """\
version: "3.1"
schemas:
  - name: future_feature
    compat_mode: forward_only
    fields:
      - {name: id, dtype: INT64, primary: true}
"""
    )
    scenario["forward_schema_matrix"] = str(future_matrix)
    scenario["target"] = {
        **scenario["target"],
        "image": "harbor.milvus.io/milvusdb/milvus:v3.1.0-build",
        "version": "3.1.0",
    }
    scenario["rollback_forward_validation_enabled"] = True

    with pytest.raises(
        ValueError,
        match="forward schemas cannot be required after rollback to 2.6.0",
    ):
        validate_resolved_gate_scenario(scenario)


def test_cluster_gate_scenarios_use_cluster_workflow_and_deploy_profile():
    manifest = _manifest()
    cluster_scenarios = [
        resolve_gate_scenario(manifest, scenario["id"])
        for scenario in manifest["scenarios"]
        if scenario["classification"] == "gate" and scenario["mode"] == "cluster"
    ]

    assert len(cluster_scenarios) == 6
    by_id = {scenario["id"]: scenario for scenario in cluster_scenarios}
    assert (
        by_id["cluster-2-6-18-to-3-0-latest-rollback-2-6-latest"]["deploy_profile"]
        == "milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml"
    )
    assert (
        by_id["cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest"][
            "deploy_profile"
        ]
        == "milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml"
    )
    assert (
        by_id["cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline"][
            "deploy_profile"
        ]
        == "milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml"
    )
    assert (
        by_id["cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline"][
            "deploy_profile"
        ]
        == "milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml"
    )
    assert (
        by_id[
            "cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline"
        ]["deploy_profile"]
        == "milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml"
    )
    assert (
        by_id[
            "cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline"
        ]["deploy_profile"]
        == "milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml"
    )
    for scenario in cluster_scenarios:
        assert scenario["workflow_template"] == "milvus-cluster-upgrade-rollback"


def test_woodpecker_2cu_ha_gate_rejects_single_replica_profile_override():
    manifest = _manifest()
    scenario_id = (
        "cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline"
    )

    scenario = resolve_gate_scenario(manifest, scenario_id)

    assert scenario["submit_generate_name"] == "c30-2cu-ha-"
    assert scenario["topology_requirements"]["min_replicas"] == {
        "proxy": 2,
        "queryNode": 2,
        "dataNode": 2,
        "streamingNode": 2,
    }
    with pytest.raises(
        ValueError,
        match="does not satisfy topology_requirements.min_replicas",
    ):
        resolve_gate_scenario(
            manifest,
            scenario_id,
            deploy_profile_override=(
                "milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml"
            ),
        )


def test_gate_scenario_rejects_deploy_profile_mode_mismatch():
    manifest = _manifest()

    with pytest.raises(ValueError, match="mode cluster does not match deploy profile"):
        resolve_gate_scenario(
            manifest,
            "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
            deploy_profile_override=(
                "milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml"
            ),
        )


def test_gate_scenario_rejects_version_override_outside_declared_family():
    manifest = _manifest()

    with pytest.raises(
        ValueError, match="rollback version override must remain in 2.6"
    ):
        resolve_gate_scenario(
            manifest,
            "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
            phase_overrides={"rollback": {"version": "3.0.0"}},
        )


def test_gate_scenario_rejects_parseable_image_override_outside_version_family():
    manifest = _manifest()

    with pytest.raises(
        ValueError,
        match="target image version family 2.6 does not match declared version family 3.0",
    ):
        resolve_gate_scenario(
            manifest,
            "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
            phase_overrides={
                "target": {
                    "image": "harbor.milvus.io/milvusdb/milvus:v2.6.18",
                    "version": "3.0.1",
                }
            },
        )


def test_gate_scenario_rejects_mutable_image_override():
    manifest = _manifest()

    with pytest.raises(ValueError, match="target image override must be immutable"):
        resolve_gate_scenario(
            manifest,
            "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
            phase_overrides={
                "target": {
                    "image": "harbor.milvus.io/milvusdb/milvus:master-latest",
                    "version": "3.0.1",
                }
            },
        )


def test_gate_scenario_allows_concrete_master_build_tag():
    manifest = _manifest()

    scenario = resolve_gate_scenario(
        manifest,
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        phase_overrides={
            "target": {
                "image": "harbor.milvus.io/milvusdb/milvus:master-20260804-a1b2c3d4",
                "version": "3.0.1",
            }
        },
    )

    assert scenario["target"]["version"] == "3.0.1"


def test_gate_scenario_allows_mutable_tag_when_pinned_by_digest():
    manifest = _manifest()
    digest = "a" * 64

    scenario = resolve_gate_scenario(
        manifest,
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        phase_overrides={
            "target": {
                "image": f"milvusdb/milvus:master-latest@sha256:{digest}",
                "version": "3.0.1",
            }
        },
    )

    assert scenario["target"]["image"].endswith(digest)
    parameters = render_argo_parameters(scenario, manifest)
    assert parameters["target-milvus-image"].endswith(digest)


def test_formal_gate_render_rejects_mutable_image_reference():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        phase_overrides={
            "target": {
                "image": "milvusdb/milvus:master-20260804-a1b2c3d4",
                "version": "3.0.1",
            }
        },
    )
    scenario["target"]["image"] = "milvusdb/milvus:master-latest"

    with pytest.raises(ValueError, match="runnable gate contains mutable images"):
        render_argo_parameters(scenario, manifest)


def test_gate_scenario_validates_versioned_image_tag_before_digest():
    manifest = _manifest()

    with pytest.raises(ValueError, match="target image version family 2.6"):
        resolve_gate_scenario(
            manifest,
            "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
            phase_overrides={
                "target": {
                    "image": "milvusdb/milvus:v2.6.18@sha256:deadbeef",
                    "version": "3.0.1",
                }
            },
        )


def test_gate_scenario_rejects_unknown_phase_override():
    manifest = _manifest()

    with pytest.raises(ValueError, match="unsupported phase overrides: post-config"):
        resolve_gate_scenario(
            manifest,
            "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
            phase_overrides={"post-config": {"image": "milvus:test"}},
        )


def test_cluster_2_6_gate_scenario_uses_pulsar_profile():
    manifest = _manifest()
    cluster_2_6_scenarios = [
        resolve_gate_scenario(manifest, scenario["id"])
        for scenario in manifest["scenarios"]
        if scenario["mode"] == "cluster"
        and scenario["id"].startswith("cluster-2-6-18-to-3-0-latest")
    ]

    assert {scenario["classification"] for scenario in cluster_2_6_scenarios} == {
        "gate",
    }
    expected_generate_names = {
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest": "c26rb-",
        "cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest": "c26to-",
    }
    for scenario in cluster_2_6_scenarios:
        assert (
            scenario["deploy_profile"]
            == "milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml"
        )
        assert (
            scenario["submit_generate_name"] == expected_generate_names[scenario["id"]]
        )
        assert len(scenario["submit_generate_name"]) <= 20


def test_cluster_target_only_feature_gate_contract():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
    )

    assert scenario["workflow_template"] == "milvus-cluster-upgrade-rollback"
    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6.yaml"
    )
    assert scenario["forward_schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_3_0.yaml"
    )
    assert scenario["forward_workload_enabled"] is True
    assert scenario["schema_evolution_existing_enabled"] is False
    assert scenario["schema_evolution_forward_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is False


def test_cluster_json_shredding_gate_writes_forward_data_after_config_toggle():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
    )

    assert scenario["workflow_template"] == "milvus-cluster-upgrade-rollback"
    assert scenario["base"]["json_shredding_enabled"] is False
    assert scenario["target"]["json_shredding_enabled"] is False
    assert scenario["post_upgrade_config_toggle_enabled"] is True
    assert scenario["post_upgrade_json_shredding_enabled"] is True
    assert scenario["rollback"]["json_shredding_enabled"] is True
    assert scenario["forward_workload_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is True
    assert scenario["forward_schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_json_shredding.yaml"
    )


def test_2_6_to_3_0_rollback_gate_scenarios_forbid_storage_v3_and_vortex():
    manifest = _manifest()
    resolved = [
        resolve_gate_scenario(manifest, scenario["id"])
        for scenario in manifest["scenarios"]
        if scenario["classification"] == "gate"
    ]
    scenarios = [
        scenario
        for scenario in resolved
        if scenario["base"]["version"].startswith("2.6")
        and scenario["target"]["version"].startswith("3.0")
        and scenario["rollback"]["version"].startswith("2.6")
    ]

    assert scenarios
    for scenario in scenarios:
        assert scenario["support_status"] == "supported_with_config_constraints"
        assert {"storage_v3", "vortex"} <= set(scenario["forbidden_after_upgrade"])
        for phase in ["base", "target", "rollback"]:
            assert scenario[phase].get("loon_ffi_enabled", False) is False
            assert scenario[phase]["vortex_enabled"] is False


@pytest.mark.parametrize(
    ("phase", "field", "match"),
    [
        ("base", "loon_ffi_enabled", "base.storage_v3"),
        ("base", "vortex_enabled", "base.vortex"),
        ("target", "loon_ffi_enabled", "target.storage_v3"),
        ("target", "vortex_enabled", "target.vortex"),
        ("rollback", "loon_ffi_enabled", "rollback.storage_v3"),
        ("rollback", "vortex_enabled", "rollback.vortex"),
    ],
)
def test_2_6_to_3_0_rollback_gate_rejects_effective_storage_v3_or_vortex_in_any_phase(
    phase: str,
    field: str,
    match: str,
):
    manifest = _manifest()
    unsafe = deepcopy(manifest)
    scenario = next(
        item
        for item in unsafe["scenarios"]
        if item["id"] == "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest"
    )
    scenario[phase][field] = True

    with pytest.raises(ValueError, match=match):
        resolve_gate_scenario(unsafe, scenario["id"])


def test_manifest_references_are_centralized():
    manifest = _manifest()
    assert set(manifest["image_aliases"]) == {
        "milvus-2-6-18",
        "milvus-2-6-latest",
        "milvus-3-0-baseline",
        "milvus-3-0-latest",
    }
    assert manifest["image_aliases"]["milvus-3-0-baseline"] == {
        "image": MILVUS_3_0_BASELINE_IMAGE,
        "version": "3.0.0",
    }
    for scenario in manifest["scenarios"]:
        for phase in ["base", "target", "rollback"]:
            assert "image_ref" in scenario[phase]
            assert "image" not in scenario[phase]
            assert "version" not in scenario[phase]


def test_3_0_loon_vortex_gate_scenarios_keep_storage_features_enabled_after_upgrade():
    manifest = _manifest()
    scenarios = [
        resolve_gate_scenario(manifest, scenario_id)
        for scenario_id in [
            "standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
            "cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
        ]
    ]

    for scenario in scenarios:
        assert scenario["classification"] == "gate"
        assert scenario["support_status"] == "supported"
        assert scenario.get("allow_unsafe_negative_coverage") is not True
        assert (
            scenario["schema_matrix"]
            == "milvus_client/manifests/schema_matrix_3_0.yaml"
        )
        assert scenario["base"].get("loon_ffi_enabled", False) is False
        assert scenario["base"]["vortex_enabled"] is False
        assert scenario["target"]["loon_ffi_enabled"] is True
        assert scenario["target"]["vortex_enabled"] is True
        assert scenario["rollback"]["loon_ffi_enabled"] is True
        assert scenario["rollback"]["vortex_enabled"] is True
        assert scenario["rollback"]["image"] == scenario["base"]["image"]
        assert scenario["validation_policy"]["pressure_fail_on_error"] is True
        assert scenario["validation_policy"]["gate_allow_warning"] is False


def test_standalone_json_shredding_gate_writes_forward_data_after_config_toggle():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
    )

    assert scenario["classification"] == "gate"
    assert scenario["workflow_template"] == ("milvus-standalone-3-0-upgrade-rollback")
    assert scenario["base"]["json_shredding_enabled"] is False
    assert scenario["target"]["json_shredding_enabled"] is False
    assert scenario["post_upgrade_config_toggle_enabled"] is True
    assert scenario["post_upgrade_json_shredding_enabled"] is True
    assert scenario["rollback"]["json_shredding_enabled"] is True
    assert scenario["forward_workload_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is True
    assert scenario["forward_schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_json_shredding.yaml"
    )
    assert scenario["validation_policy"]["pressure_fail_on_error"] is True
    assert scenario["validation_policy"]["gate_allow_warning"] is False


def test_negative_vortex_to_2_6_scenario_is_not_a_gate():
    manifest = _manifest()
    negative = resolve_gate_scenario(
        manifest, "standalone-3-0-loon-vortex-to-2-6-negative"
    )

    assert negative["classification"] == "negative"
    assert negative["support_status"] == "unsupported"
    assert negative["allow_unsafe_negative_coverage"] is True
    assert negative["target"]["vortex_enabled"] is True
    assert negative["validation_policy"]["gate_allow_warning"] is True


def test_manifest_validator_rejects_string_bool_values():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["scenarios"][0]["target"]["loon_ffi_enabled"] = "false"

    with pytest.raises(
        ValueError, match="target.loon_ffi_enabled must be a YAML boolean"
    ):
        validate_gate_manifest(broken)


def test_manifest_validator_rejects_string_bool_values_in_defaults():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["defaults"]["index_compatibility_validation_enabled"] = "false"

    with pytest.raises(
        ValueError,
        match="index_compatibility_validation_enabled must be a YAML boolean",
    ):
        validate_gate_manifest(broken)


def test_manifest_validator_rejects_string_phase_bool_value_in_defaults():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["defaults"]["phase_dml_dql_validation_enabled"] = "false"

    with pytest.raises(
        ValueError,
        match="phase_dml_dql_validation_enabled must be a YAML boolean",
    ):
        validate_gate_manifest(broken)


def test_manifest_validator_rejects_unsafe_negative_escape_hatch_on_gate():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["scenarios"][0]["allow_unsafe_negative_coverage"] = True

    with pytest.raises(ValueError, match="only when classification is negative"):
        validate_gate_manifest(broken)


def test_manifest_validator_rejects_invalid_topology_replica_requirement():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["scenarios"][0]["topology_requirements"] = {"min_replicas": {"proxy": 0}}

    with pytest.raises(
        ValueError,
        match="topology_requirements.min_replicas.proxy must be a positive integer",
    ):
        validate_gate_manifest(broken)


def test_manifest_validator_rejects_long_submit_generate_name():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["scenarios"][0]["submit_generate_name"] = "this-prefix-is-too-long-"

    with pytest.raises(ValueError, match="submit_generate_name must be at most"):
        validate_gate_manifest(broken)


def test_manifest_validator_rejects_invalid_submit_generate_name():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["scenarios"][0]["submit_generate_name"] = "BadPrefix"

    with pytest.raises(ValueError, match="submit_generate_name must end with"):
        validate_gate_manifest(broken)


def test_manifest_validator_rejects_unknown_refs():
    manifest = _manifest()
    broken = deepcopy(manifest)
    broken["scenarios"][0]["target"]["image_ref"] = "missing-image-alias"

    with pytest.raises(ValueError, match="missing-image-alias"):
        validate_gate_manifest(broken)
