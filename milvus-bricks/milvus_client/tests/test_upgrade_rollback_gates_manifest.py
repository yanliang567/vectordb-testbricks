from copy import deepcopy
from pathlib import Path

import pytest

from milvus_client.common.gates import (
    load_gate_manifest,
    resolve_gate_scenario,
    validate_gate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "manifests" / "upgrade_rollback_gates.yaml"


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
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
    } <= set(scenarios)
    for scenario_id in [
        "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "standalone-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-loon-vortex-rollback-3-0-baseline",
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


def test_cluster_gate_scenarios_use_cluster_workflow_and_deploy_profile():
    manifest = _manifest()
    cluster_scenarios = [
        resolve_gate_scenario(manifest, scenario["id"])
        for scenario in manifest["scenarios"]
        if scenario["classification"] == "gate" and scenario["mode"] == "cluster"
    ]

    assert len(cluster_scenarios) == 3
    by_id = {scenario["id"]: scenario for scenario in cluster_scenarios}
    assert (
        by_id["cluster-2-6-18-to-3-0-latest-rollback-2-6-latest"]["deploy_profile"]
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
    for scenario in cluster_scenarios:
        assert scenario["workflow_template"] == "milvus-cluster-upgrade-rollback"


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
    for scenario in cluster_2_6_scenarios:
        assert (
            scenario["deploy_profile"]
            == "milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml"
        )
        assert scenario["submit_generate_name"] == "c26rb-"
        assert len(scenario["submit_generate_name"]) <= 20


def test_2_6_to_3_0_rollback_gate_scenarios_forbid_storage_v3_and_vortex():
    manifest = _manifest()
    scenarios = [
        resolve_gate_scenario(manifest, scenario["id"])
        for scenario in manifest["scenarios"]
        if scenario["classification"] == "gate"
        and scenario["id"].endswith("to-3-0-latest-rollback-2-6-latest")
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
