from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from milvus_client.common.gates import (
    load_gate_manifest,
    render_argo_parameters,
    resolve_gate_scenario,
    validate_gate_manifest,
    validate_registered_scenario_parameters,
    validate_resolved_gate_scenario,
)
from milvus_client.common.schema import load_schema_matrix

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "manifests" / "upgrade_rollback_gates.yaml"
EXECUTION_PATH_FIXTURE = (
    ROOT / "tests" / "fixtures" / "upgrade_rollback_execution_paths_v1.yaml"
)
MILVUS_3_0_BASELINE_IMAGE = (
    "harbor.milvus.io/milvusdb/milvus:v3.0.0@"
    "sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862"
)
MILVUS_3_0_VORTEX_CANDIDATE_BASELINE_IMAGE = (
    "harbor.milvus.io/milvusdb/milvus:3.0-20260807-697431f2@"
    "sha256:e29d3275d1184ecf5e00995dd8b2c234e6912ea3c899c6b9d6f8807f7d6db5a3"
)
MILVUS_3_0_VORTEX_CANDIDATE_TARGET_IMAGE = (
    "harbor.milvus.io/milvusdb/milvus:3.0-20260807-1439dc7d@"
    "sha256:ed46e16fcb58bd460722e6fc1c0e6294e86fd4e062431877d0a872dcb510cd64"
)
MILVUS_2_6_18_BASELINE_IMAGE = (
    "harbor.milvus.io/milvusdb/milvus:v2.6.18@"
    "sha256:c6e332d3783c2c42649d5f76c5dae79d553927196a60547f619be13484ab44f6"
)


def _manifest() -> dict:
    return load_gate_manifest(GATES)


def _execution_path_signatures(manifest):
    contract_metadata = {
        "index-engine-contract-mode",
        "index-engine-capability",
        "index-engine-qualification-status",
    }
    signatures = {}
    for raw in manifest["scenarios"]:
        scenario = resolve_gate_scenario(manifest, raw["id"])
        rendered = render_argo_parameters(scenario, manifest, allow_placeholder=True)
        signatures[raw["id"]] = {
            name: value
            for name, value in rendered.items()
            if name not in contract_metadata
        }
    return signatures


def test_manifest_v2_contract_migration_preserves_existing_execution_paths():
    expected = yaml.safe_load(EXECUTION_PATH_FIXTURE.read_text())

    assert len(expected) == 26
    assert _execution_path_signatures(_manifest()) == expected


@pytest.mark.parametrize(
    ("scenario_id", "expected_eligible"),
    [
        (
            scenario["id"],
            scenario["classification"] == "gate"
            and scenario["support_status"]
            in {"supported", "supported_with_config_constraints"},
        )
        for scenario in _manifest()["scenarios"]
    ],
)
def test_all_manifest_scenarios_render_expected_release_gate_eligibility(
    scenario_id,
    expected_eligible,
):
    manifest = _manifest()
    scenario = resolve_gate_scenario(manifest, scenario_id)

    params = render_argo_parameters(scenario, manifest, allow_placeholder=True)

    assert params["release-gate-eligible"] == str(expected_eligible).lower()


def test_unregistered_scenario_accepts_only_safe_report_metadata():
    runtime = {
        "scenario-classification": "unregistered",
        "scenario-support-status": "unknown",
        "release-gate-eligible": "false",
        "index-engine-contract-mode": "none",
        "index-engine-capability": "none",
        "index-engine-qualification-status": "not_applicable",
    }

    resolved = validate_registered_scenario_parameters(
        _manifest(), "custom-unregistered-scenario", runtime
    )

    assert resolved is None


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("scenario-classification", "gate"),
        ("scenario-support-status", "supported"),
        ("release-gate-eligible", "true"),
        ("index-engine-contract-mode", "round_trip"),
        ("index-engine-capability", "IndexEngineV10V4"),
        ("index-engine-qualification-status", "passed"),
    ],
)
def test_unregistered_scenario_rejects_release_gate_metadata_claims(parameter, value):
    runtime = {
        "scenario-classification": "unregistered",
        "scenario-support-status": "unknown",
        "release-gate-eligible": "false",
        "index-engine-contract-mode": "none",
        "index-engine-capability": "none",
        "index-engine-qualification-status": "not_applicable",
    }
    runtime[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        validate_registered_scenario_parameters(
            _manifest(), "custom-unregistered-scenario", runtime
        )


def test_upgrade_rollback_gates_manifest_contains_required_gate_scenarios():
    manifest = _manifest()
    assert manifest["version"] == "2"
    assert manifest["defaults"]["index_compatibility_validation_enabled"] is True
    assert manifest["defaults"]["phase_dml_dql_validation_enabled"] is True
    assert manifest["defaults"]["phase_new_collection_rows"] == 3000
    assert manifest["defaults"]["phase_existing_dml_rows"] == 1000
    assert manifest["defaults"]["phase_existing_delete_rows"] == 100
    assert manifest["capability_qualifications"] == {}
    scenarios = {
        scenario["id"]: resolve_gate_scenario(manifest, scenario["id"])
        for scenario in manifest["scenarios"]
    }

    assert {
        "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "standalone-3-0-index-v10-v4-upgrade-rollback",
        "standalone-3-0-index-v11-v4-upgrade-rollback",
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline",
        "cluster-3-0-index-v10-v4-upgrade-rollback",
        "cluster-3-0-index-v11-v4-upgrade-rollback",
    } <= set(scenarios)
    for scenario_id in [
        "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "standalone-3-0-index-v10-v4-upgrade-rollback",
        "standalone-3-0-index-v11-v4-upgrade-rollback",
        "cluster-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest",
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline",
        "cluster-3-0-index-v10-v4-upgrade-rollback",
        "cluster-3-0-index-v11-v4-upgrade-rollback",
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
        "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "cluster-2-6-18-to-3-0-latest-rollback-2-6-latest",
    ],
)
def test_2_6_round_trip_known_limitations_preserve_full_regression_matrix(
    scenario_id,
):
    manifest = _manifest()
    scenario = resolve_gate_scenario(manifest, scenario_id)
    params = render_argo_parameters(scenario, manifest, allow_placeholder=True)

    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6.yaml"
    )
    assert scenario["classification"] == "known_limitation"
    assert scenario["support_status"] == "unsupported"
    assert params["release-gate-eligible"] == "false"
    specs = load_schema_matrix(ROOT.parent / scenario["schema_matrix"])
    names = {spec.name for spec in specs}

    assert len(specs) == 11
    assert "struct_array_element_rollback_safe" in names
    assert "struct_array_varchar_autoindex_rollback_safe" in names
    assert "struct_array_numeric_autoindex_rollback_safe" in names


@pytest.mark.parametrize(
    "scenario_id",
    [
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
        "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
    ],
)
def test_3_0_core_gates_validate_target_created_indexes_after_rollback(
    scenario_id,
):
    scenario = resolve_gate_scenario(_manifest(), scenario_id)

    assert scenario["forward_schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_3_0.yaml"
    )
    assert scenario["forward_workload_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is True


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

    assert len(cluster_scenarios) == 9
    by_id = {scenario["id"]: scenario for scenario in cluster_scenarios}
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
        by_id[
            "cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline"
        ]["deploy_profile"]
        == "milvus_client/manifests/deploy_profiles/cluster-woodpecker-2cu.yaml"
    )
    for scenario_id in [
        "cluster-3-0-index-v10-v4-upgrade-rollback",
        "cluster-3-0-index-v11-v4-upgrade-rollback",
    ]:
        assert by_id[scenario_id]["deploy_profile"] == (
            "milvus_client/manifests/deploy_profiles/cluster-pulsar-1cu.yaml"
        )
    for scenario_id in [
        "cluster-3-0-1-vortex-self-compat-upgrade-rollback",
        "cluster-3-0-1-json-shredding-vortex-rollback",
        "cluster-3-0-0-to-3-0-1-vortex-enable-rollback",
        "cluster-3-0-1-loon-ffi-rollback",
    ]:
        assert (
            by_id[scenario_id]["deploy_profile"]
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


def test_registered_scenario_runtime_allows_operational_overrides():
    manifest = _manifest()
    scenario_id = "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline"
    scenario = resolve_gate_scenario(manifest, scenario_id)
    runtime = render_argo_parameters(scenario, manifest, allow_placeholder=True)
    runtime.update(
        {
            "workflow-template": scenario["workflow_template"],
            "repo-revision": "1102cc54b52050b3f156188cda435b54e8888680",
            "collection-prefix": "qa_pr25_runtime",
            "forward-collection-prefix": "qa_pr25_runtime_forward",
            "target-milvus-image": (
                "harbor.milvus.io/milvusdb/milvus:master-20260810-a1b2c3d4"
            ),
            "target-version": "3.0.1",
        }
    )

    resolved = validate_registered_scenario_parameters(manifest, scenario_id, runtime)

    assert resolved is not None
    assert resolved["target"]["version"] == "3.0.1"
    assert resolved["target"]["image"].endswith("master-20260810-a1b2c3d4")


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("schema-matrix", "milvus_client/manifests/schema_matrix_2_6.yaml"),
        ("target-target-vec-index-version", "10"),
        ("index-compatibility-validation-enabled", "false"),
        ("schema-evolution-existing-enabled", "false"),
        ("scenario-classification", "candidate"),
        ("scenario-support-status", "pre_release_candidate"),
        ("release-gate-eligible", "false"),
        ("index-engine-contract-mode", "round_trip"),
        ("index-engine-capability", "IndexEngineV10V4"),
        ("index-engine-qualification-status", "passed"),
    ],
)
def test_registered_scenario_runtime_rejects_protected_parameter_drift(
    parameter,
    value,
):
    manifest = _manifest()
    scenario_id = "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline"
    runtime = render_argo_parameters(
        resolve_gate_scenario(manifest, scenario_id),
        manifest,
        allow_placeholder=True,
    )
    runtime["workflow-template"] = "milvus-standalone-3-0-upgrade-rollback"
    runtime["target-milvus-image"] = (
        "harbor.milvus.io/milvusdb/milvus:master-20260810-a1b2c3d4"
    )
    runtime["target-version"] = "3.0.1"
    runtime[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        validate_registered_scenario_parameters(manifest, scenario_id, runtime)


def test_registered_scenario_runtime_rejects_wrong_workflow_template():
    manifest = _manifest()
    scenario_id = "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline"
    scenario = resolve_gate_scenario(manifest, scenario_id)
    runtime = render_argo_parameters(scenario, manifest, allow_placeholder=True)
    runtime.update(
        {
            "workflow-template": "milvus-standalone-3-0-upgrade-rollback",
            "target-milvus-image": (
                "harbor.milvus.io/milvusdb/milvus:master-20260810-a1b2c3d4"
            ),
            "target-version": "3.0.1",
        }
    )

    with pytest.raises(ValueError, match="workflow-template"):
        validate_registered_scenario_parameters(manifest, scenario_id, runtime)


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
        "known_limitation",
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


def test_cluster_json_shredding_known_limitation_writes_forward_data_after_config_toggle():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
    )

    assert scenario["classification"] == "known_limitation"
    assert scenario["support_status"] == "unsupported"
    assert "milvus-io/milvus#52341" in scenario["description"]
    assert "milvus-io/milvus#52768" in scenario["description"]
    assert scenario["workflow_template"] == "milvus-cluster-upgrade-rollback"
    assert scenario["deploy_profile"] == (
        "milvus_client/manifests/deploy_profiles/cluster-woodpecker-v0-1-38-1cu.yaml"
    )
    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6_woodpecker_reader_recovery.yaml"
    )
    assert [
        spec.name
        for spec in load_schema_matrix(ROOT.parent / scenario["schema_matrix"])
    ] == [
        "scalar_dynamic_partition_key",
        "scalar_autoindex_formats_rollback_safe",
        "scalar_explicit_index_formats_rollback_safe",
        "vector_autoid_bm25",
        "explicit_partitions_nullable",
        "struct_array_element_rollback_safe",
        "nullable_vectors_all",
        "geometry_rtree_rollback_safe",
        "legacy_index_rollback_safe",
    ]
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
        if item["id"]
        == "standalone-2-6-18-to-3-0-latest-target-only-features-rollback-2-6-latest"
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
        "milvus-3-0-1",
        "milvus-3-0-vortex-candidate-baseline",
        "milvus-3-0-vortex-candidate-target",
    }
    assert manifest["image_aliases"]["milvus-3-0-1"] == {
        "image": "harbor.milvus.io/milvusdb/milvus:v3.0.1-placeholder",
        "version": "3.0.1",
    }
    assert manifest["image_aliases"]["milvus-3-0-baseline"] == {
        "image": MILVUS_3_0_BASELINE_IMAGE,
        "version": "3.0.0",
    }
    assert manifest["image_aliases"]["milvus-2-6-18"] == {
        "image": MILVUS_2_6_18_BASELINE_IMAGE,
        "version": "2.6.18",
    }
    assert manifest["image_aliases"]["milvus-3-0-vortex-candidate-baseline"] == {
        "image": MILVUS_3_0_VORTEX_CANDIDATE_BASELINE_IMAGE,
        "version": "3.0.0",
        "source_commit": "697431f2c36c146e5033c71de6b91133aedc8f4c",
        "milvus_storage_commit": "63c29c674bf8c75a84c49cca2c8ab088e771e57e",
        "vortex_compatibility": "vortex-0.75+",
    }
    assert manifest["image_aliases"]["milvus-3-0-vortex-candidate-target"] == {
        "image": MILVUS_3_0_VORTEX_CANDIDATE_TARGET_IMAGE,
        "version": "3.0.0",
        "source_commit": "1439dc7de8b198a01c2afa0ae20c0c473e0e1abc",
        "milvus_storage_commit": "63c29c674bf8c75a84c49cca2c8ab088e771e57e",
        "vortex_compatibility": "vortex-0.75+",
    }
    for scenario in manifest["scenarios"]:
        for phase in ["base", "target", "rollback"]:
            assert "image_ref" in scenario[phase]
            assert "image" not in scenario[phase]
            assert "version" not in scenario[phase]


def test_3_0_loon_vortex_candidate_scenarios_keep_storage_features_enabled_after_upgrade():
    manifest = _manifest()
    scenarios = [
        resolve_gate_scenario(manifest, scenario_id)
        for scenario_id in [
            "standalone-3-0-vortex-candidate-upgrade-rollback",
            "cluster-3-0-vortex-candidate-upgrade-rollback",
        ]
    ]

    for scenario in scenarios:
        assert scenario["classification"] == "candidate"
        assert scenario["support_status"] == "pre_release_candidate"
        assert scenario.get("allow_unsafe_negative_coverage") is not True
        assert scenario["schema_matrix"] == (
            "milvus_client/manifests/schema_matrix_2_6.yaml"
        )
        assert scenario["base"].get("loon_ffi_enabled", False) is False
        assert scenario["base"]["vortex_enabled"] is False
        assert scenario["target"]["loon_ffi_enabled"] is True
        assert scenario["target"]["vortex_enabled"] is True
        assert scenario["rollback"]["loon_ffi_enabled"] is True
        assert scenario["rollback"]["vortex_enabled"] is True
        assert scenario["rollback"]["image"] == scenario["base"]["image"]
        assert scenario["base"]["image"] == (MILVUS_3_0_VORTEX_CANDIDATE_BASELINE_IMAGE)
        assert scenario["target"]["image"] == MILVUS_3_0_VORTEX_CANDIDATE_TARGET_IMAGE
        assert scenario["target"]["vortex_compatibility"] == "vortex-0.75+"
        assert scenario["rollback"]["vortex_compatibility"] == "vortex-0.75+"
        assert scenario["forward_workload_enabled"] is True
        assert scenario["rollback_forward_validation_enabled"] is True
        assert scenario["forward_schema_matrix"] == (
            "milvus_client/manifests/schema_matrix_3_0_storage_v3.yaml"
        )
        assert scenario["validation_policy"]["pressure_fail_on_error"] is True
        assert scenario["validation_policy"]["gate_allow_warning"] is False


def test_supported_gate_rejects_vortex_on_v3_0_0():
    scenario = resolve_gate_scenario(
        _manifest(),
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
    )
    scenario["target"]["loon_ffi_enabled"] = True
    scenario["target"]["vortex_enabled"] = True

    with pytest.raises(ValueError, match="target Vortex requires Milvus >= 3.0.1"):
        validate_resolved_gate_scenario(scenario)


def test_supported_gate_rejects_rollback_below_v3_0_1_after_vortex_writes():
    scenario = resolve_gate_scenario(
        _manifest(),
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
    )
    scenario["target"].update(
        {
            "image": "harbor.milvus.io/milvusdb/milvus:v3.0.1@sha256:" + "a" * 64,
            "version": "3.0.1",
            "loon_ffi_enabled": True,
            "vortex_enabled": True,
        }
    )
    scenario["rollback"]["vortex_enabled"] = False

    with pytest.raises(
        ValueError,
        match="rollback must support Vortex data written before rollback",
    ):
        validate_resolved_gate_scenario(scenario)


def test_supported_gate_accepts_v3_0_1_vortex_upgrade_rollback():
    scenario = resolve_gate_scenario(
        _manifest(),
        "standalone-3-0-baseline-to-3-0-latest-rollback-3-0-baseline",
    )
    for phase, digest_char in (("target", "a"), ("rollback", "b")):
        scenario[phase].update(
            {
                "image": "harbor.milvus.io/milvusdb/milvus:v3.0.1@sha256:"
                + digest_char * 64,
                "version": "3.0.1",
                "loon_ffi_enabled": True,
                "vortex_enabled": True,
            }
        )

    validate_resolved_gate_scenario(scenario)


@pytest.mark.parametrize("phase", ["base", "target", "rollback"])
def test_vortex_requires_loon_ffi_in_every_phase(phase):
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-1-vortex-self-compat-upgrade-rollback"
    )
    scenario[phase]["loon_ffi_enabled"] = False

    with pytest.raises(ValueError, match=f"{phase} Vortex requires LoonFFI"):
        validate_resolved_gate_scenario(scenario)


def test_3_0_1_vortex_self_compat_gate_keeps_vortex_in_all_phases():
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-1-vortex-self-compat-upgrade-rollback"
    )

    assert scenario["classification"] == "gate"
    assert scenario["support_status"] == "supported"
    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_3_0_storage_v3.yaml"
    )
    for phase in ("base", "target", "rollback"):
        assert scenario[phase]["loon_ffi_enabled"] is True
        assert scenario[phase]["vortex_enabled"] is True
        assert scenario[phase]["version"] == "3.0.1"


def test_3_0_0_to_3_0_1_vortex_enable_gate_rolls_back_to_3_0_1():
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-0-to-3-0-1-vortex-enable-rollback"
    )

    assert scenario["classification"] == "gate"
    assert scenario["support_status"] == "supported"
    assert scenario["base"]["version"] == "3.0.0"
    assert scenario["base"]["vortex_enabled"] is False
    assert scenario["target"]["version"] == "3.0.1"
    assert scenario["target"]["loon_ffi_enabled"] is True
    assert scenario["target"]["vortex_enabled"] is True
    assert scenario["rollback"]["version"] == "3.0.1"
    assert scenario["rollback"]["vortex_enabled"] is True


def test_3_0_1_loon_ffi_gate_disables_vortex_across_phases():
    scenario = resolve_gate_scenario(_manifest(), "standalone-3-0-1-loon-ffi-rollback")

    assert scenario["classification"] == "gate"
    assert scenario["support_status"] == "supported"
    assert scenario["base"]["loon_ffi_enabled"] is False
    assert scenario["target"]["loon_ffi_enabled"] is True
    assert scenario["rollback"]["loon_ffi_enabled"] is False
    for phase in ("base", "target", "rollback"):
        assert scenario[phase]["vortex_enabled"] is False


def test_3_0_1_json_shredding_vortex_gate_enables_both_features_at_rollback():
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-1-json-shredding-vortex-rollback"
    )

    assert scenario["post_upgrade_config_toggle_enabled"] is True
    assert scenario["post_upgrade_json_shredding_enabled"] is True
    assert scenario["rollback"]["json_shredding_enabled"] is True
    assert scenario["rollback"]["loon_ffi_enabled"] is True
    assert scenario["rollback"]["vortex_enabled"] is True


def test_3_0_1_vortex_disable_rollback_gate():
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-1-vortex-disable-rollback"
    )

    assert scenario["classification"] == "gate"
    assert scenario["support_status"] == "supported"
    assert scenario["target"]["vortex_enabled"] is True
    assert scenario["rollback"]["vortex_enabled"] is False
    assert scenario["forward_workload_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is True
    assert scenario["validation_policy"]["data_integrity"] == "strict"
    assert scenario["validation_policy"]["gate_allow_warning"] is False


def test_3_0_1_vortex_disable_keep_loon_rollback_gate():
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-1-vortex-disable-keep-loon-rollback"
    )

    assert scenario["classification"] == "gate"
    assert scenario["support_status"] == "supported"
    assert scenario["target"]["vortex_enabled"] is True
    assert scenario["rollback"]["loon_ffi_enabled"] is True
    assert scenario["rollback"]["vortex_enabled"] is False
    assert scenario["forward_workload_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is True
    assert scenario["validation_policy"]["data_integrity"] == "strict"


@pytest.mark.parametrize(
    ("scenario_id", "workflow_template"),
    [
        (
            "standalone-3-0-0-to-3-0-1-vortex-enable-rollback",
            "milvus-standalone-3-0-upgrade-rollback",
        ),
        (
            "cluster-3-0-0-to-3-0-1-vortex-enable-rollback",
            "milvus-cluster-upgrade-rollback",
        ),
        (
            "standalone-3-0-1-loon-ffi-rollback",
            "milvus-standalone-3-0-upgrade-rollback",
        ),
        (
            "cluster-3-0-1-loon-ffi-rollback",
            "milvus-cluster-upgrade-rollback",
        ),
    ],
)
def test_new_storage_gate_scenarios_use_expected_workflow(
    scenario_id, workflow_template
):
    scenario = resolve_gate_scenario(_manifest(), scenario_id)

    assert scenario["classification"] == "gate"
    assert scenario["support_status"] == "supported"
    assert scenario["workflow_template"] == workflow_template


def test_3_0_1_gate_rejects_placeholder_image_without_allow_placeholder():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest, "standalone-3-0-1-vortex-self-compat-upgrade-rollback"
    )

    with pytest.raises(ValueError, match="placeholder images"):
        render_argo_parameters(scenario, manifest)


def test_vortex_candidate_rejects_runtime_image_override():
    with pytest.raises(ValueError, match="reviewed candidate image is locked"):
        resolve_gate_scenario(
            _manifest(),
            "standalone-3-0-vortex-candidate-upgrade-rollback",
            phase_overrides={
                "target": {
                    "image": "harbor.milvus.io/milvusdb/milvus:3.0-20260810-deadbeef@sha256:"
                    + "c" * 64,
                    "version": "3.0.0",
                }
            },
        )


def test_registered_vortex_candidate_parameters_require_locked_images():
    manifest = _manifest()
    scenario_id = "standalone-3-0-vortex-candidate-upgrade-rollback"
    scenario = resolve_gate_scenario(manifest, scenario_id)
    runtime = render_argo_parameters(scenario, manifest)
    runtime["workflow-template"] = "milvus-standalone-3-0-upgrade-rollback"

    resolved = validate_registered_scenario_parameters(manifest, scenario_id, runtime)

    assert resolved is not None
    assert resolved["target"]["image"] == MILVUS_3_0_VORTEX_CANDIDATE_TARGET_IMAGE

    runtime["target-milvus-image"] = (
        "harbor.milvus.io/milvusdb/milvus:3.0-20260810-deadbeef@sha256:" + "c" * 64
    )
    with pytest.raises(ValueError, match="reviewed candidate image is locked"):
        validate_registered_scenario_parameters(manifest, scenario_id, runtime)


def test_vortex_candidate_rejects_missing_reviewed_compatibility_metadata():
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-vortex-candidate-upgrade-rollback"
    )
    scenario["rollback"].pop("vortex_compatibility")

    with pytest.raises(ValueError, match="reviewed Vortex compatibility metadata"):
        validate_resolved_gate_scenario(scenario)


def test_vortex_candidate_rejects_source_commit_that_does_not_match_image_tag():
    scenario = resolve_gate_scenario(
        _manifest(), "standalone-3-0-vortex-candidate-upgrade-rollback"
    )
    scenario["target"]["source_commit"] = "a" * 40

    with pytest.raises(ValueError, match="reviewed Vortex compatibility metadata"):
        validate_resolved_gate_scenario(scenario)


@pytest.mark.parametrize(
    ("scenario_id", "target_vec_index_version", "forward_schema_matrix"),
    [
        (
            "standalone-3-0-index-v10-v4-upgrade-rollback",
            10,
            "milvus_client/manifests/schema_matrix_3_0_index_v10_v4.yaml",
        ),
        (
            "cluster-3-0-index-v10-v4-upgrade-rollback",
            10,
            "milvus_client/manifests/schema_matrix_3_0_index_v10_v4.yaml",
        ),
        (
            "standalone-3-0-index-v11-v4-upgrade-rollback",
            11,
            "milvus_client/manifests/schema_matrix_3_0_index_v11_v4.yaml",
        ),
        (
            "cluster-3-0-index-v11-v4-upgrade-rollback",
            11,
            "milvus_client/manifests/schema_matrix_3_0_index_v11_v4.yaml",
        ),
    ],
)
def test_index_engine_scenarios_validate_index_matrix_on_target_only(
    scenario_id, target_vec_index_version, forward_schema_matrix
):
    manifest = _manifest()
    scenario = resolve_gate_scenario(manifest, scenario_id)
    parameters = render_argo_parameters(scenario, manifest, allow_placeholder=True)

    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6.yaml"
    )
    assert scenario["forward_schema_matrix"] == forward_schema_matrix
    assert scenario["forward_workload_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is False
    assert "target_vec_index_version" not in scenario["base"]
    assert "target_scalar_index_version" not in scenario["base"]
    assert scenario["target"]["target_vec_index_version"] == target_vec_index_version
    assert scenario["target"]["target_scalar_index_version"] == 4
    assert "target_vec_index_version" not in scenario["rollback"]
    assert "target_scalar_index_version" not in scenario["rollback"]

    assert parameters["schema-matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6.yaml"
    )
    assert parameters["forward-schema-matrix"] == forward_schema_matrix
    assert parameters["forward-workload-enabled"] == "true"
    assert parameters["rollback-forward-validation-enabled"] == "false"
    assert parameters["drop-forward-before-rollback-enabled"] == "true"
    assert parameters["base-target-vec-index-version"] == "-1"
    assert parameters["target-target-vec-index-version"] == str(
        target_vec_index_version
    )
    assert parameters["rollback-target-vec-index-version"] == "-1"
    assert parameters["base-target-scalar-index-version"] == "-1"
    assert parameters["target-target-scalar-index-version"] == "4"
    assert parameters["rollback-target-scalar-index-version"] == "-1"
    assert parameters["index-engine-contract-mode"] == "target_only"
    assert parameters["index-engine-capability"] == (
        f"IndexEngineV{target_vec_index_version}V4"
    )
    assert parameters["index-engine-qualification-status"] == "unsupported"


def test_standalone_json_shredding_known_limitation_writes_forward_data_after_config_toggle():
    manifest = _manifest()
    scenario = resolve_gate_scenario(
        manifest,
        "standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
    )

    assert scenario["classification"] == "known_limitation"
    assert scenario["support_status"] == "unsupported"
    assert scenario["workflow_template"] == ("milvus-standalone-3-0-upgrade-rollback")
    assert scenario["base"]["json_shredding_enabled"] is False
    assert scenario["target"]["json_shredding_enabled"] is False
    assert scenario["post_upgrade_config_toggle_enabled"] is True
    assert scenario["post_upgrade_json_shredding_enabled"] is True
    assert scenario["rollback"]["json_shredding_enabled"] is True
    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6.yaml"
    )
    assert scenario["forward_workload_enabled"] is True
    assert scenario["rollback_forward_validation_enabled"] is True
    assert scenario["forward_schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_json_shredding.yaml"
    )
    assert scenario["validation_policy"]["pressure_fail_on_error"] is True
    assert scenario["validation_policy"]["gate_allow_warning"] is False


@pytest.mark.parametrize(
    "scenario_id",
    [
        "standalone-3-0-vortex-candidate-upgrade-rollback",
        "standalone-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline",
        "cluster-3-0-vortex-candidate-upgrade-rollback",
    ],
)
def test_storage_feature_gates_use_stable_rollback_safe_base_matrix(scenario_id):
    scenario = resolve_gate_scenario(_manifest(), scenario_id)

    assert scenario["schema_matrix"] == (
        "milvus_client/manifests/schema_matrix_2_6.yaml"
    )


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


INDEX_ENGINE_SCENARIOS = [
    "standalone-3-0-index-v10-v4-upgrade-rollback",
    "standalone-3-0-index-v11-v4-upgrade-rollback",
    "cluster-3-0-index-v10-v4-upgrade-rollback",
    "cluster-3-0-index-v11-v4-upgrade-rollback",
]


def _raw_scenario(manifest, scenario_id):
    return next(item for item in manifest["scenarios"] if item["id"] == scenario_id)


def _promote_contract_to_round_trip(manifest, scenario_id):
    scenario = _raw_scenario(manifest, scenario_id)
    contract = scenario["index_engine_contract"]
    contract["mode"] = "round_trip"
    contract.pop("rollback_safe_matrix_ref")
    image_ref = scenario["base"]["image_ref"]
    immutable_image = manifest["image_aliases"][image_ref]["image"]
    manifest["capability_qualifications"][image_ref] = {
        "immutable_image": immutable_image,
        "capabilities": {
            contract["capability"]: {
                "status": "passed",
                "evidence": {
                    "standalone": "argo://qa/index-contract-standalone",
                    "cluster": "argo://qa/index-contract-cluster",
                },
            }
        },
    }


@pytest.mark.parametrize("scenario_id", INDEX_ENGINE_SCENARIOS)
def test_target_only_contract_expands_expected_execution_flags(scenario_id):
    manifest = _manifest()
    raw = _raw_scenario(manifest, scenario_id)
    contract = raw["index_engine_contract"]
    resolved = resolve_gate_scenario(manifest, scenario_id)

    assert not (
        set(raw)
        & {
            "schema_matrix_ref",
            "forward_schema_matrix_ref",
            "forward_workload_enabled",
            "rollback_forward_validation_enabled",
            "drop_forward_before_rollback_enabled",
        }
    )
    assert contract["mode"] == "target_only"
    assert resolved["index_engine_contract"]["qualification_status"] == "unsupported"
    assert resolved["forward_workload_enabled"] is True
    assert resolved["rollback_enabled"] is True
    assert resolved["rollback_forward_validation_enabled"] is False
    assert resolved["drop_forward_before_rollback_enabled"] is True
    assert resolved["base"].get("target_vec_index_version", -1) == -1
    assert resolved["target"]["target_vec_index_version"] == contract["vector_version"]
    assert resolved["rollback"].get("target_vec_index_version", -1) == -1


def test_round_trip_contract_expands_expected_execution_flags():
    manifest = deepcopy(_manifest())
    scenario_id = "standalone-3-0-index-v10-v4-upgrade-rollback"
    _promote_contract_to_round_trip(manifest, scenario_id)

    resolved = resolve_gate_scenario(manifest, scenario_id)

    assert resolved["schema_matrix"] == resolved["forward_schema_matrix"]
    assert resolved["rollback_forward_validation_enabled"] is True
    assert resolved["drop_forward_before_rollback_enabled"] is False
    assert resolved["index_engine_contract"]["qualification_status"] == "passed"
    for phase in ("base", "target", "rollback"):
        assert resolved[phase]["target_vec_index_version"] == 10
        assert resolved[phase]["target_scalar_index_version"] == 4


def test_manifest_rejects_unknown_index_engine_contract_mode():
    manifest = deepcopy(_manifest())
    _raw_scenario(manifest, INDEX_ENGINE_SCENARIOS[0])["index_engine_contract"][
        "mode"
    ] = "auto"

    with pytest.raises(ValueError, match="index_engine_contract.mode"):
        validate_gate_manifest(manifest)


def test_manifest_rejects_contract_and_derived_fields_together():
    manifest = deepcopy(_manifest())
    _raw_scenario(manifest, INDEX_ENGINE_SCENARIOS[0])["forward_workload_enabled"] = (
        True
    )

    with pytest.raises(ValueError, match="owns derived fields"):
        validate_gate_manifest(manifest)


def test_manifest_rejects_matrix_capability_mismatch():
    manifest = deepcopy(_manifest())
    _raw_scenario(manifest, INDEX_ENGINE_SCENARIOS[0])["index_engine_contract"][
        "capability"
    ] = "IndexEngineV11V4"

    with pytest.raises(ValueError, match="must all require IndexEngineV11V4"):
        resolve_gate_scenario(manifest, INDEX_ENGINE_SCENARIOS[0])


def test_manifest_rejects_matrix_validator_version_mismatch():
    manifest = deepcopy(_manifest())
    _raw_scenario(manifest, INDEX_ENGINE_SCENARIOS[0])["index_engine_contract"][
        "vector_version"
    ] = 11

    with pytest.raises(ValueError, match="target_vec_index_version must be 11"):
        resolve_gate_scenario(manifest, INDEX_ENGINE_SCENARIOS[0])


def test_round_trip_contract_requires_qualified_base_image():
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    contract = _raw_scenario(manifest, scenario_id)["index_engine_contract"]
    contract["mode"] = "round_trip"
    contract.pop("rollback_safe_matrix_ref")

    with pytest.raises(ValueError, match="requires capability qualification"):
        resolve_gate_scenario(manifest, scenario_id)


def test_round_trip_contract_requires_mode_specific_evidence():
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    _promote_contract_to_round_trip(manifest, scenario_id)
    qualification = manifest["capability_qualifications"]["milvus-3-0-baseline"]
    qualification["capabilities"]["IndexEngineV10V4"]["evidence"].pop("standalone")

    with pytest.raises(ValueError, match="requires standalone evidence"):
        resolve_gate_scenario(manifest, scenario_id)


def test_manifest_rejects_qualification_for_different_image():
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    _promote_contract_to_round_trip(manifest, scenario_id)
    qualification = manifest["capability_qualifications"]["milvus-3-0-baseline"]
    qualification["immutable_image"] = (
        "harbor.milvus.io/milvusdb/milvus:3.0-20260826-e47a679a"
    )

    with pytest.raises(
        ValueError, match="must match its immutable image alias exactly"
    ):
        validate_gate_manifest(manifest)


def test_manifest_rejects_tag_only_qualification_image():
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    _promote_contract_to_round_trip(manifest, scenario_id)
    image_ref = "milvus-3-0-baseline"
    tag_only_image = "harbor.milvus.io/milvusdb/milvus:v3.0.0"
    manifest["image_aliases"][image_ref]["image"] = tag_only_image
    manifest["capability_qualifications"][image_ref]["immutable_image"] = tag_only_image

    with pytest.raises(ValueError, match="digest-pinned"):
        validate_gate_manifest(manifest)


def test_manifest_rejects_unstable_qualification_evidence():
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    _promote_contract_to_round_trip(manifest, scenario_id)
    qualification = manifest["capability_qualifications"]["milvus-3-0-baseline"]
    qualification["capabilities"]["IndexEngineV10V4"]["evidence"]["standalone"] = (
        "not-even-a-url"
    )

    with pytest.raises(ValueError, match="stable argo:// or https:// URI"):
        validate_gate_manifest(manifest)


def test_target_only_contract_rejects_forward_only_rollback_safe_matrix():
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    _raw_scenario(manifest, scenario_id)["index_engine_contract"][
        "rollback_safe_matrix_ref"
    ] = "3.0_index_v11_v4"

    with pytest.raises(ValueError, match="must contain only rollback_safe schemas"):
        resolve_gate_scenario(manifest, scenario_id)


def test_target_only_contract_rejects_other_index_engine_capability(tmp_path):
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    matrix = yaml.safe_load(
        (ROOT / "manifests" / "schema_matrix_3_0_index_v11_v4.yaml").read_text()
    )
    for schema in matrix["schemas"]:
        schema["compat_mode"] = "rollback_safe"
    matrix_path = tmp_path / "rollback_safe_but_index_engine.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix))
    manifest["schema_matrices"]["unsafe_other_index_engine"] = str(matrix_path)
    _raw_scenario(manifest, scenario_id)["index_engine_contract"][
        "rollback_safe_matrix_ref"
    ] = "unsafe_other_index_engine"

    with pytest.raises(ValueError, match="must not require index-engine capabilities"):
        resolve_gate_scenario(manifest, scenario_id)


def test_v10_qualification_does_not_authorize_v11():
    manifest = deepcopy(_manifest())
    v10_id = INDEX_ENGINE_SCENARIOS[0]
    v11_id = INDEX_ENGINE_SCENARIOS[1]
    _promote_contract_to_round_trip(manifest, v10_id)
    v11_contract = _raw_scenario(manifest, v11_id)["index_engine_contract"]
    v11_contract["mode"] = "round_trip"
    v11_contract.pop("rollback_safe_matrix_ref")

    with pytest.raises(
        ValueError, match="no passed qualification for IndexEngineV11V4"
    ):
        resolve_gate_scenario(manifest, v11_id)


def test_runtime_image_override_must_match_qualification():
    manifest = deepcopy(_manifest())
    scenario_id = INDEX_ENGINE_SCENARIOS[0]
    _promote_contract_to_round_trip(manifest, scenario_id)

    with pytest.raises(ValueError, match="does not match qualified immutable image"):
        resolve_gate_scenario(
            manifest,
            scenario_id,
            phase_overrides={
                "base": {
                    "image": "harbor.milvus.io/milvusdb/milvus:3.0-20260826-e47a679a"
                }
            },
        )
