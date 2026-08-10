from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from milvus_client.common.deploy import load_deploy_profile
from milvus_client.common.schema import (
    load_schema_matrix,
    rollback_incompatible_specs,
)
from milvus_client.common.version import (
    image_is_immutable,
    image_version_family,
    version_family,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_MANIFEST = ROOT / "manifests" / "upgrade_rollback_gates.yaml"
WORKFLOW_TEMPLATE_MODES = {
    "milvus-standalone-2-6-upgrade-rollback": "standalone",
    "milvus-standalone-3-0-upgrade-rollback": "standalone",
    "milvus-cluster-upgrade-rollback": "cluster",
}
REGISTERED_SCENARIO_MUTABLE_PARAMETERS = {
    "repo-revision",
    "collection-prefix",
    "forward-collection-prefix",
}


def load_gate_manifest(path: str | Path = DEFAULT_GATE_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text()) or {}
    validate_gate_manifest(payload, source=str(manifest_path))
    return payload


def resolve_gate_scenario(
    manifest: dict[str, Any],
    scenario_id: str,
    *,
    deploy_profile_override: str | None = None,
    phase_overrides: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    scenarios = manifest.get("scenarios") or []
    scenario = next((item for item in scenarios if item.get("id") == scenario_id), None)
    if scenario is None:
        available = ", ".join(sorted(str(item.get("id")) for item in scenarios))
        raise ValueError(f"unknown scenario id {scenario_id!r}; available: {available}")

    resolved = deepcopy(scenario)
    resolved["workflow_template"] = _resolve_ref(
        manifest, "workflow_templates", scenario, "workflow_template"
    )
    resolved["deploy_profile"] = deploy_profile_override or _resolve_ref(
        manifest, "deploy_profiles", scenario, "deploy_profile"
    )
    resolved["schema_matrix"] = _resolve_ref(
        manifest, "schema_matrices", scenario, "schema_matrix"
    )
    if scenario.get("submit_generate_name") is not None:
        resolved["submit_generate_name"] = str(scenario["submit_generate_name"])
    if "forward_schema_matrix_ref" in scenario or "forward_schema_matrix" in scenario:
        resolved["forward_schema_matrix"] = _resolve_ref(
            manifest, "schema_matrices", scenario, "forward_schema_matrix"
        )
    else:
        resolved["forward_schema_matrix"] = resolved["schema_matrix"]

    unknown_phases = sorted(set(phase_overrides or {}) - {"base", "target", "rollback"})
    if unknown_phases:
        raise ValueError(
            f"{scenario_id}: unsupported phase overrides: {', '.join(unknown_phases)}"
        )

    for phase in ("base", "target", "rollback"):
        resolved[phase] = _resolve_phase(manifest, scenario, phase)
        override = (phase_overrides or {}).get(phase) or {}
        unknown = sorted(set(override) - {"image", "version"})
        if unknown:
            raise ValueError(
                f"{scenario_id}: unsupported {phase} override fields: "
                f"{', '.join(unknown)}"
            )
        for field in ("image", "version"):
            if override.get(field):
                if field == "image" and resolved.get("classification") == "gate":
                    override_image = str(override[field])
                    if not image_is_immutable(override_image):
                        raise ValueError(
                            f"{scenario_id}: {phase} image override must be immutable; "
                            f"use a concrete build tag or sha256 digest, got {override_image}"
                        )
                if field == "version":
                    declared_family = version_family(resolved[phase]["version"])
                    override_family = version_family(str(override[field]))
                    if override_family != declared_family:
                        raise ValueError(
                            f"{scenario_id}: {phase} version override must remain in "
                            f"{declared_family}; got {override[field]}"
                        )
                resolved[phase][field] = str(override[field])

    defaults = manifest.get("defaults") or {}
    resolved.setdefault(
        "index_compatibility_validation_enabled",
        defaults.get("index_compatibility_validation_enabled", True),
    )
    resolved.setdefault(
        "phase_dml_dql_validation_enabled",
        defaults.get("phase_dml_dql_validation_enabled", True),
    )

    validate_resolved_gate_scenario(resolved)
    return resolved


def render_argo_parameters(
    scenario: dict[str, Any],
    manifest: dict[str, Any],
    *,
    allow_placeholder: bool = False,
) -> dict[str, str]:
    validate_no_gate_placeholders(scenario, allow_placeholder=allow_placeholder)
    defaults = manifest.get("defaults") or {}
    validation_policy = scenario.get("validation_policy") or {}

    params: dict[str, str] = {
        "repo-url": str(defaults.get("repo_url", "")),
        "repo-revision": str(defaults.get("repo_revision", "main")),
        "scenario-id": str(scenario["id"]),
        "deploy-profile": str(scenario["deploy_profile"]),
        "base-milvus-image": str(scenario["base"]["image"]),
        "base-version": str(scenario["base"]["version"]),
        "target-milvus-image": str(scenario["target"]["image"]),
        "target-version": str(scenario["target"]["version"]),
        "rollback-milvus-image": str(scenario["rollback"]["image"]),
        "rollback-version": str(scenario["rollback"]["version"]),
        "base-json-shredding-enabled": _bool_str(
            scenario["base"].get("json_shredding_enabled", False)
        ),
        "target-json-shredding-enabled": _bool_str(
            scenario["target"].get("json_shredding_enabled", False)
        ),
        "rollback-json-shredding-enabled": _bool_str(
            scenario["rollback"].get("json_shredding_enabled", False)
        ),
        "base-loon-ffi-enabled": _bool_str(
            scenario["base"].get("loon_ffi_enabled", False)
        ),
        "target-loon-ffi-enabled": _bool_str(
            scenario["target"].get("loon_ffi_enabled", False)
        ),
        "rollback-loon-ffi-enabled": _bool_str(
            scenario["rollback"].get("loon_ffi_enabled", False)
        ),
        "base-vortex-enabled": _bool_str(scenario["base"].get("vortex_enabled", False)),
        "target-vortex-enabled": _bool_str(
            scenario["target"].get("vortex_enabled", False)
        ),
        "rollback-vortex-enabled": _bool_str(
            scenario["rollback"].get("vortex_enabled", False)
        ),
        "base-target-vec-index-version": str(
            scenario["base"].get("target_vec_index_version", -1)
        ),
        "target-target-vec-index-version": str(
            scenario["target"].get("target_vec_index_version", -1)
        ),
        "rollback-target-vec-index-version": str(
            scenario["rollback"].get("target_vec_index_version", -1)
        ),
        "base-target-scalar-index-version": str(
            scenario["base"].get("target_scalar_index_version", -1)
        ),
        "target-target-scalar-index-version": str(
            scenario["target"].get("target_scalar_index_version", -1)
        ),
        "rollback-target-scalar-index-version": str(
            scenario["rollback"].get("target_scalar_index_version", -1)
        ),
        "post-upgrade-config-toggle-enabled": _bool_str(
            scenario.get("post_upgrade_config_toggle_enabled", False)
        ),
        "post-upgrade-json-shredding-enabled": _bool_str(
            scenario.get(
                "post_upgrade_json_shredding_enabled",
                scenario["target"].get("json_shredding_enabled", False),
            )
        ),
        "forward-workload-enabled": _bool_str(
            scenario.get("forward_workload_enabled", False)
        ),
        "forward-schema-matrix": str(scenario["forward_schema_matrix"]),
        "rollback-enabled": _bool_str(scenario.get("rollback_enabled", True)),
        "rollback-forward-validation-enabled": _bool_str(
            scenario.get("rollback_forward_validation_enabled", False)
        ),
        "index-compatibility-validation-enabled": _bool_str(
            scenario.get(
                "index_compatibility_validation_enabled",
                defaults.get("index_compatibility_validation_enabled", True),
            )
        ),
        "phase-dml-dql-validation-enabled": _bool_str(
            scenario.get(
                "phase_dml_dql_validation_enabled",
                defaults.get("phase_dml_dql_validation_enabled", True),
            )
        ),
        "schema-evolution-existing-enabled": _bool_str(
            scenario.get("schema_evolution_existing_enabled", False)
        ),
        "schema-evolution-forward-enabled": _bool_str(
            scenario.get("schema_evolution_forward_enabled", False)
        ),
        "collection-prefix": str(scenario["collection_prefix"]),
        "forward-collection-prefix": str(
            scenario.get(
                "forward_collection_prefix", f"{scenario['collection_prefix']}_forward"
            )
        ),
        "schema-matrix": str(scenario["schema_matrix"]),
        "rows-per-collection": str(
            scenario.get(
                "rows_per_collection", defaults.get("rows_per_collection", 1000)
            )
        ),
        "batch-size": str(scenario.get("batch_size", defaults.get("batch_size", 100))),
        "phase-new-collection-rows": str(
            scenario.get(
                "phase_new_collection_rows",
                defaults.get("phase_new_collection_rows", 3000),
            )
        ),
        "phase-existing-dml-rows": str(
            scenario.get(
                "phase_existing_dml_rows",
                defaults.get("phase_existing_dml_rows", 1000),
            )
        ),
        "phase-existing-delete-rows": str(
            scenario.get(
                "phase_existing_delete_rows",
                defaults.get("phase_existing_delete_rows", 100),
            )
        ),
        "pressure-modules": " ".join(
            scenario.get("pressure_modules", defaults.get("pressure_modules", []))
        ),
        "pressure-fail-on-error": _bool_str(
            validation_policy.get("pressure_fail_on_error", False)
        ),
        "gate-allow-warning": _bool_str(
            validation_policy.get("gate_allow_warning", True)
        ),
        "allow-unsafe-negative-coverage": _bool_str(
            scenario.get("allow_unsafe_negative_coverage", False)
        ),
        "rollback-serviceability-timeout-sec": str(
            scenario.get(
                "rollback_serviceability_timeout_sec",
                defaults.get("rollback_serviceability_timeout_sec", 900),
            )
        ),
        "rollback-serviceability-interval-sec": str(
            scenario.get(
                "rollback_serviceability_interval_sec",
                defaults.get("rollback_serviceability_interval_sec", 10),
            )
        ),
    }
    return {key: value for key, value in params.items() if value != ""}


def render_submission(
    scenario: dict[str, Any],
    manifest: dict[str, Any],
    *,
    allow_placeholder: bool = False,
) -> dict[str, Any]:
    submission = {
        "scenario_id": scenario["id"],
        "workflow_template": scenario["workflow_template"],
        "parameters": render_argo_parameters(
            scenario, manifest, allow_placeholder=allow_placeholder
        ),
    }
    if scenario.get("submit_generate_name"):
        submission["submit_generate_name"] = scenario["submit_generate_name"]
    return submission


def validate_registered_scenario_parameters(
    manifest: dict[str, Any],
    scenario_id: str,
    runtime_parameters: dict[str, Any],
) -> dict[str, Any] | None:
    if not any(item.get("id") == scenario_id for item in manifest.get("scenarios", [])):
        return None

    missing_phase_parameters = [
        key
        for phase in ("base", "target", "rollback")
        for key in (f"{phase}-milvus-image", f"{phase}-version")
        if not runtime_parameters.get(key)
    ]
    if missing_phase_parameters:
        raise ValueError(
            f"{scenario_id}: runtime parameters are missing phase overrides: "
            f"{', '.join(missing_phase_parameters)}"
        )

    phase_overrides = {
        phase: {
            "image": str(runtime_parameters[f"{phase}-milvus-image"]),
            "version": str(runtime_parameters[f"{phase}-version"]),
        }
        for phase in ("base", "target", "rollback")
    }
    resolved = resolve_gate_scenario(
        manifest,
        scenario_id,
        deploy_profile_override=str(runtime_parameters.get("deploy-profile") or ""),
        phase_overrides=phase_overrides,
    )
    expected = render_argo_parameters(resolved, manifest)
    drift = {}
    actual_workflow_template = str(
        runtime_parameters.get("workflow-template") or "<missing>"
    )
    if actual_workflow_template != resolved["workflow_template"]:
        drift["workflow-template"] = {
            "expected": resolved["workflow_template"],
            "actual": actual_workflow_template,
        }
    for name, expected_value in expected.items():
        if name in REGISTERED_SCENARIO_MUTABLE_PARAMETERS:
            continue
        if name not in runtime_parameters:
            drift[name] = {"expected": expected_value, "actual": "<missing>"}
            continue
        actual_value = str(runtime_parameters[name])
        if actual_value != expected_value:
            drift[name] = {"expected": expected_value, "actual": actual_value}
    if drift:
        details = ", ".join(
            f"{name}: expected {values['expected']!r}, got {values['actual']!r}"
            for name, values in sorted(drift.items())
        )
        raise ValueError(
            f"{scenario_id}: registered scenario protected parameter drift: {details}"
        )
    return resolved


def validate_gate_manifest(
    manifest: dict[str, Any], source: str = "<manifest>"
) -> None:
    if manifest.get("version") != "1":
        raise ValueError(f"{source}: version must be '1'")
    for section in (
        "defaults",
        "workflow_templates",
        "deploy_profiles",
        "schema_matrices",
        "image_aliases",
    ):
        if not isinstance(manifest.get(section), dict) or not manifest[section]:
            raise ValueError(f"{source}: {section} must be a non-empty mapping")
    _require_bool_if_present(
        manifest["defaults"],
        "index_compatibility_validation_enabled",
        source=source,
        scenario_id="defaults",
    )
    _require_bool_if_present(
        manifest["defaults"],
        "phase_dml_dql_validation_enabled",
        source=source,
        scenario_id="defaults",
    )
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{source}: scenarios must be a non-empty list")
    seen: set[str] = set()
    for scenario in scenarios:
        scenario_id = scenario.get("id")
        if not scenario_id:
            raise ValueError(f"{source}: every scenario requires id")
        if scenario_id in seen:
            raise ValueError(f"{source}: duplicate scenario id {scenario_id}")
        seen.add(str(scenario_id))
        for key in ("mode", "classification", "support_status", "collection_prefix"):
            if key not in scenario:
                raise ValueError(f"{source}: scenario {scenario_id} missing {key}")
        _validate_submit_generate_name(
            scenario.get("submit_generate_name"),
            source=source,
            scenario_id=str(scenario_id),
        )
        _validate_scenario_bool_fields(scenario, source=source)
        _validate_topology_requirements(scenario, source=source)
        if (
            scenario.get("allow_unsafe_negative_coverage") is True
            and scenario.get("classification") != "negative"
        ):
            raise ValueError(
                f"{source}: scenario {scenario_id} may enable allow_unsafe_negative_coverage only "
                "when classification is negative"
            )
        for section, logical_name in (
            ("workflow_templates", "workflow_template"),
            ("deploy_profiles", "deploy_profile"),
            ("schema_matrices", "schema_matrix"),
        ):
            _resolve_ref(manifest, section, scenario, logical_name)
        for phase in ("base", "target", "rollback"):
            _resolve_phase(manifest, scenario, phase)


def validate_resolved_gate_scenario(scenario: dict[str, Any]) -> None:
    _validate_scenario_execution_mode(scenario)
    _validate_phase_image_versions(scenario)
    if scenario.get("classification") != "gate":
        return
    base_version = str(scenario["base"]["version"])
    target_version = str(scenario["target"]["version"])
    rollback_version = str(scenario["rollback"]["version"])

    if (
        scenario.get("rollback_enabled", True) is True
        and scenario.get("forward_workload_enabled") is True
        and scenario.get("rollback_forward_validation_enabled") is True
    ):
        forward_schema_matrix = _schema_matrix_path(
            str(scenario.get("forward_schema_matrix") or "")
        )
        incompatible_forward_specs = rollback_incompatible_specs(
            load_schema_matrix(forward_schema_matrix),
            rollback_version,
        )
        if incompatible_forward_specs:
            raise ValueError(
                f"{scenario['id']}: forward schemas cannot be required after rollback "
                f"to {rollback_version}; incompatible schemas: "
                f"{', '.join(spec.name for spec in incompatible_forward_specs)}"
            )

    is_2_6_to_3_0_to_2_6 = (
        base_version.startswith("2.6")
        and target_version.startswith("3.0")
        and rollback_version.startswith("2.6")
    )
    if not is_2_6_to_3_0_to_2_6:
        return

    forbidden = set(scenario.get("forbidden_after_upgrade") or [])
    if not {"storage_v3", "vortex"} <= forbidden:
        raise ValueError(
            f"{scenario['id']}: 2.6 -> 3.0 -> 2.6 gate must forbid storage_v3 and vortex"
        )

    blocked_flags_by_phase = {
        "base": {"storage_v3": "loon_ffi_enabled", "vortex": "vortex_enabled"},
        "target": {"storage_v3": "loon_ffi_enabled", "vortex": "vortex_enabled"},
        "rollback": {"storage_v3": "loon_ffi_enabled", "vortex": "vortex_enabled"},
    }
    enabled = [
        f"{phase}.{logical_flag}({field})"
        for phase, flags in blocked_flags_by_phase.items()
        for logical_flag, field in flags.items()
        if scenario[phase].get(field) is True
    ]
    if enabled:
        raise ValueError(
            f"{scenario['id']}: 2.6 -> 3.0 -> 2.6 gate must keep storage v3/vortex disabled; "
            f"invalid phase flags: {', '.join(enabled)}"
        )


def validate_no_gate_placeholders(
    scenario: dict[str, Any], *, allow_placeholder: bool = False
) -> None:
    if allow_placeholder:
        return
    placeholders = [
        f"{phase}.image={scenario[phase]['image']}"
        for phase in ("base", "target", "rollback")
        if "placeholder" in str(scenario[phase].get("image", ""))
        and not image_is_immutable(str(scenario[phase].get("image", "")))
    ]
    if placeholders:
        raise ValueError(
            f"{scenario['id']}: runnable scenario contains placeholder images: "
            f"{', '.join(placeholders)}; pass --allow-placeholder only for dry-run/review output"
        )
    if scenario.get("classification") == "gate":
        mutable_images = [
            f"{phase}.image={scenario[phase]['image']}"
            for phase in ("base", "target", "rollback")
            if not image_is_immutable(str(scenario[phase].get("image", "")))
        ]
        if mutable_images:
            raise ValueError(
                f"{scenario['id']}: runnable gate contains mutable images: "
                f"{', '.join(mutable_images)}; use concrete build tags or sha256 digests"
            )


def _resolve_phase(
    manifest: dict[str, Any], scenario: dict[str, Any], phase: str
) -> dict[str, Any]:
    payload = deepcopy(scenario.get(phase) or {})
    image_ref = payload.get("image_ref")
    if image_ref:
        aliases = manifest.get("image_aliases") or {}
        if image_ref not in aliases:
            raise ValueError(
                f"{scenario.get('id')}: {phase}.image_ref {image_ref!r} is not defined"
            )
        alias = aliases[image_ref]
        payload["image"] = alias["image"]
        payload["version"] = alias["version"]
    if not payload.get("image") or not payload.get("version"):
        raise ValueError(
            f"{scenario.get('id')}: {phase} requires image_ref or image+version"
        )
    return payload


def _resolve_ref(
    manifest: dict[str, Any],
    section: str,
    scenario: dict[str, Any],
    field: str,
) -> str:
    direct_value = scenario.get(field)
    if direct_value:
        return str(direct_value)
    ref = scenario.get(f"{field}_ref")
    if ref is None:
        raise ValueError(f"{scenario.get('id')}: missing {field} or {field}_ref")
    mapping = manifest.get(section) or {}
    if ref not in mapping:
        raise ValueError(
            f"{scenario.get('id')}: {field}_ref {ref!r} is not defined in {section}"
        )
    return str(mapping[ref])


def _validate_scenario_execution_mode(scenario: dict[str, Any]) -> None:
    scenario_id = str(scenario.get("id") or "<unknown>")
    scenario_mode = str(scenario.get("mode") or "")
    workflow_template = str(scenario.get("workflow_template") or "")
    workflow_mode = WORKFLOW_TEMPLATE_MODES.get(workflow_template)
    if workflow_mode is None:
        raise ValueError(
            f"{scenario_id}: workflow template {workflow_template!r} has no mode mapping"
        )
    if workflow_mode != scenario_mode:
        raise ValueError(
            f"{scenario_id}: mode {scenario_mode} does not match workflow template "
            f"{workflow_template} mode {workflow_mode}"
        )

    profile_path = Path(str(scenario.get("deploy_profile") or ""))
    if not profile_path.is_absolute():
        profile_path = ROOT.parent / profile_path
    if not profile_path.exists():
        raise ValueError(
            f"{scenario_id}: deploy profile does not exist: {profile_path}"
        )
    profile = load_deploy_profile(profile_path)
    profile_mode = str(profile.get("mode") or "")
    if profile_mode != scenario_mode:
        raise ValueError(
            f"{scenario_id}: mode {scenario_mode} does not match deploy profile "
            f"{profile_path} mode {profile_mode}"
        )

    min_replicas = (scenario.get("topology_requirements") or {}).get(
        "min_replicas"
    ) or {}
    insufficient = []
    for component, minimum in min_replicas.items():
        component_spec = profile.get("components", {}).get(component)
        actual = component_spec.get("replicas", 1) if component_spec else 0
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < minimum:
            insufficient.append(f"{component}={actual!r} (required >= {minimum})")
    if insufficient:
        raise ValueError(
            f"{scenario_id}: deploy profile {profile_path} does not satisfy "
            "topology_requirements.min_replicas: " + ", ".join(insufficient)
        )


def _schema_matrix_path(value: str) -> Path:
    matrix_path = Path(value)
    if not matrix_path.is_absolute():
        matrix_path = ROOT.parent / matrix_path
    return matrix_path


def _validate_phase_image_versions(scenario: dict[str, Any]) -> None:
    for phase in ("base", "target", "rollback"):
        phase_payload = scenario[phase]
        declared_family = version_family(str(phase_payload["version"]))
        image_family = image_version_family(str(phase_payload["image"]))
        if image_family is not None and image_family != declared_family:
            raise ValueError(
                f"{scenario['id']}: {phase} image version family {image_family} "
                f"does not match declared version family {declared_family}; "
                f"image={phase_payload['image']} version={phase_payload['version']}"
            )


def _bool_str(value: Any) -> str:
    if not isinstance(value, bool):
        raise TypeError(f"expected YAML boolean, got {type(value).__name__}: {value!r}")
    return "true" if value else "false"


def _validate_scenario_bool_fields(scenario: dict[str, Any], *, source: str) -> None:
    scenario_id = scenario.get("id")
    scenario_bool_fields = {
        "post_upgrade_config_toggle_enabled",
        "post_upgrade_json_shredding_enabled",
        "forward_workload_enabled",
        "rollback_enabled",
        "rollback_forward_validation_enabled",
        "index_compatibility_validation_enabled",
        "phase_dml_dql_validation_enabled",
        "schema_evolution_existing_enabled",
        "schema_evolution_forward_enabled",
        "allow_unsafe_negative_coverage",
    }
    phase_bool_fields = {
        "json_shredding_enabled",
        "loon_ffi_enabled",
        "vortex_enabled",
    }
    phase_index_version_fields = {
        "target_vec_index_version",
        "target_scalar_index_version",
    }
    validation_policy_bool_fields = {
        "pressure_fail_on_error",
        "gate_allow_warning",
    }
    for field in scenario_bool_fields:
        _require_bool_if_present(
            scenario, field, source=source, scenario_id=scenario_id
        )
    for phase in ("base", "target", "rollback"):
        phase_payload = scenario.get(phase) or {}
        for field in phase_bool_fields:
            _require_bool_if_present(
                phase_payload,
                field,
                source=source,
                scenario_id=scenario_id,
                prefix=phase,
            )
        for field in phase_index_version_fields:
            if field not in phase_payload:
                continue
            value = phase_payload[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError(
                    f"{source}: scenario {scenario_id} {phase}.{field} must be an integer >= -1"
                )
    validation_policy = scenario.get("validation_policy") or {}
    for field in validation_policy_bool_fields:
        _require_bool_if_present(
            validation_policy,
            field,
            source=source,
            scenario_id=scenario_id,
            prefix="validation_policy",
        )


def _validate_topology_requirements(scenario: dict[str, Any], *, source: str) -> None:
    requirements = scenario.get("topology_requirements")
    if requirements is None:
        return
    scenario_id = str(scenario.get("id") or "<unknown>")
    if not isinstance(requirements, dict):
        raise ValueError(
            f"{source}: scenario {scenario_id} topology_requirements must be a mapping"
        )
    unknown = sorted(set(requirements) - {"min_replicas"})
    if unknown:
        raise ValueError(
            f"{source}: scenario {scenario_id} topology_requirements has unknown fields: "
            f"{', '.join(unknown)}"
        )
    min_replicas = requirements.get("min_replicas")
    if not isinstance(min_replicas, dict) or not min_replicas:
        raise ValueError(
            f"{source}: scenario {scenario_id} "
            "topology_requirements.min_replicas must be a non-empty mapping"
        )
    for component, minimum in min_replicas.items():
        if not isinstance(component, str) or not component:
            raise ValueError(
                f"{source}: scenario {scenario_id} topology requirement component "
                "must be a non-empty string"
            )
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError(
                f"{source}: scenario {scenario_id} "
                f"topology_requirements.min_replicas.{component} "
                "must be a positive integer"
            )


def _validate_submit_generate_name(
    value: Any, *, source: str, scenario_id: str
) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - manifest validation consistently uses ValueError.
            f"{source}: scenario {scenario_id} submit_generate_name must be a string"
        )
    if not value:
        raise ValueError(
            f"{source}: scenario {scenario_id} submit_generate_name must not be empty"
        )
    if len(value) > 20:
        raise ValueError(
            f"{source}: scenario {scenario_id} submit_generate_name must be at most 20 chars"
        )
    if not value.endswith("-"):
        raise ValueError(
            f"{source}: scenario {scenario_id} submit_generate_name must end with '-'"
        )
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if any(ch not in allowed for ch in value) or value[0] == "-":
        raise ValueError(
            f"{source}: scenario {scenario_id} submit_generate_name must be a DNS-label prefix"
        )


def _require_bool_if_present(
    payload: dict[str, Any],
    field: str,
    *,
    source: str,
    scenario_id: Any,
    prefix: str | None = None,
) -> None:
    if field not in payload:
        return
    value = payload[field]
    if isinstance(value, bool):
        return
    field_name = f"{prefix}.{field}" if prefix else field
    raise ValueError(
        f"{source}: scenario {scenario_id} field {field_name} must be a YAML boolean, "
        f"got {type(value).__name__}: {value!r}"
    )
