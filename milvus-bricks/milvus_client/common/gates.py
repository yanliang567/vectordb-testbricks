from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_MANIFEST = ROOT / "manifests" / "upgrade_rollback_gates.yaml"


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
    if "forward_schema_matrix_ref" in scenario or "forward_schema_matrix" in scenario:
        resolved["forward_schema_matrix"] = _resolve_ref(
            manifest, "schema_matrices", scenario, "forward_schema_matrix"
        )
    else:
        resolved["forward_schema_matrix"] = resolved["schema_matrix"]

    for phase in ("base", "target", "rollback"):
        resolved[phase] = _resolve_phase(manifest, scenario, phase)

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
                defaults.get("phase_new_collection_rows", 1000),
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
    return {
        "scenario_id": scenario["id"],
        "workflow_template": scenario["workflow_template"],
        "parameters": render_argo_parameters(
            scenario, manifest, allow_placeholder=allow_placeholder
        ),
    }


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
        _validate_scenario_bool_fields(scenario, source=source)
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
    if scenario.get("classification") != "gate":
        return
    base_version = str(scenario["base"]["version"])
    target_version = str(scenario["target"]["version"])
    rollback_version = str(scenario["rollback"]["version"])
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
    ]
    if placeholders:
        raise ValueError(
            f"{scenario['id']}: runnable scenario contains placeholder images: "
            f"{', '.join(placeholders)}; pass --allow-placeholder only for dry-run/review output"
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
    validation_policy = scenario.get("validation_policy") or {}
    for field in validation_policy_bool_fields:
        _require_bool_if_present(
            validation_policy,
            field,
            source=source,
            scenario_id=scenario_id,
            prefix="validation_policy",
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
