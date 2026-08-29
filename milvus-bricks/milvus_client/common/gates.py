from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from milvus_client.common.capability import load_capability_catalog
from milvus_client.common.deploy import load_deploy_profile
from milvus_client.common.schema import (
    load_schema_matrix,
    rollback_incompatible_specs,
)
from milvus_client.common.version import (
    image_is_immutable,
    image_version_family,
    version_at_least,
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
STRICT_LIFECYCLE_CLASSIFICATIONS = {"gate", "candidate"}
RELEASE_GATE_SUPPORT_STATUSES = {
    "supported",
    "supported_with_config_constraints",
}
UNREGISTERED_SCENARIO_METADATA = {
    "scenario-classification": "unregistered",
    "scenario-support-status": "unknown",
    "release-gate-eligible": "false",
    "index-engine-contract-mode": "none",
    "index-engine-capability": "none",
    "index-engine-qualification-status": "not_applicable",
}
INDEX_ENGINE_CONTRACT_MODES = {"target_only", "round_trip"}
NO_INDEX_ENGINE_CONTRACT = {
    "mode": "none",
    "capability": "none",
    "qualification_status": "not_applicable",
}
INDEX_ENGINE_DERIVED_FIELDS = {
    "schema_matrix",
    "schema_matrix_ref",
    "forward_schema_matrix",
    "forward_schema_matrix_ref",
    "forward_workload_enabled",
    "rollback_enabled",
    "rollback_forward_validation_enabled",
    "drop_forward_before_rollback_enabled",
}
INDEX_ENGINE_PHASE_DERIVED_FIELDS = {
    "target_vec_index_version",
    "target_scalar_index_version",
}
VORTEX_MIN_SUPPORTED_VERSION = "3.0.1"
VORTEX_CANDIDATE_SUPPORT_STATUS = "pre_release_candidate"
VORTEX_CANDIDATE_COMPATIBILITY = "vortex-0.75+"
CANDIDATE_ALIAS_METADATA_FIELDS = {
    "source_commit",
    "milvus_storage_commit",
    "vortex_compatibility",
}
# `candidate` is reserved for pre-release Vortex candidates (version < 3.0.1)
# whose images are locked and must carry CANDIDATE_ALIAS_METADATA_FIELDS plus
# vortex_compatibility. Same-version LoonFFI/Vortex toggles are positive gates
# because 3.0.x binaries are dual readers (storage v2/v3, parquet/vortex); the
# compatibility boundaries are cross-version only.
FULL_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
DIGEST_PINNED_IMAGE = re.compile(r"@sha256:[0-9a-fA-F]{64}$")
QUALIFICATION_EVIDENCE_URI = re.compile(r"^(?:argo|https)://[^\s/]+/.+$")


def _image_is_digest_pinned(image: str) -> bool:
    return DIGEST_PINNED_IMAGE.search(image) is not None


def _explicit_profile_images(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    if not isinstance(value, dict):
        return images
    repository = value.get("repository")
    tag = value.get("tag")
    if isinstance(repository, str) and isinstance(tag, str):
        images.append((".".join(path), f"{repository}:{tag}"))
    for key, child in value.items():
        images.extend(_explicit_profile_images(child, (*path, str(key))))
    return images


def _qualification_evidence_is_stable(value: Any) -> bool:
    return (
        isinstance(value, str)
        and QUALIFICATION_EVIDENCE_URI.fullmatch(value) is not None
    )


def _validate_raw_index_engine_contract(
    manifest: dict[str, Any], scenario: dict[str, Any], *, source: str
) -> None:
    contract = scenario.get("index_engine_contract")
    if contract is None:
        return
    scenario_id = str(scenario.get("id") or "<unknown>")
    if not isinstance(contract, dict):
        raise ValueError(
            f"{source}: scenario {scenario_id} index_engine_contract must be a mapping"
        )
    allowed = {
        "mode",
        "capability",
        "matrix_ref",
        "rollback_safe_matrix_ref",
        "vector_version",
        "scalar_version",
        "rationale",
    }
    unknown = sorted(set(contract) - allowed)
    if unknown:
        raise ValueError(
            f"{source}: scenario {scenario_id} index_engine_contract has unknown "
            f"fields: {', '.join(unknown)}"
        )
    mode = contract.get("mode")
    if mode not in INDEX_ENGINE_CONTRACT_MODES:
        raise ValueError(
            f"{source}: scenario {scenario_id} index_engine_contract.mode must be "
            f"one of {sorted(INDEX_ENGINE_CONTRACT_MODES)!r}"
        )
    for field in ("capability", "matrix_ref"):
        if not isinstance(contract.get(field), str) or not contract[field]:
            raise ValueError(
                f"{source}: scenario {scenario_id} index_engine_contract.{field} "
                "must be a non-empty string"
            )
    for field in ("vector_version", "scalar_version"):
        value = contract.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{source}: scenario {scenario_id} index_engine_contract.{field} "
                "must be a non-negative integer"
            )
    matrix_ref = str(contract["matrix_ref"])
    if matrix_ref not in manifest.get("schema_matrices", {}):
        raise ValueError(
            f"{source}: scenario {scenario_id} index_engine_contract.matrix_ref "
            f"{matrix_ref!r} is not defined in schema_matrices"
        )
    collisions = sorted(set(scenario) & INDEX_ENGINE_DERIVED_FIELDS)
    for phase in ("base", "target", "rollback"):
        collisions.extend(
            f"{phase}.{field}"
            for field in sorted(
                set(scenario.get(phase) or {}) & INDEX_ENGINE_PHASE_DERIVED_FIELDS
            )
        )
    if collisions:
        raise ValueError(
            f"{source}: scenario {scenario_id} index_engine_contract owns derived "
            f"fields: {', '.join(collisions)}"
        )
    if mode == "target_only":
        safe_ref = contract.get("rollback_safe_matrix_ref")
        if not isinstance(safe_ref, str) or not safe_ref:
            raise ValueError(
                f"{source}: scenario {scenario_id} target_only contract requires "
                "rollback_safe_matrix_ref"
            )
        if safe_ref not in manifest.get("schema_matrices", {}):
            raise ValueError(
                f"{source}: scenario {scenario_id} rollback_safe_matrix_ref "
                f"{safe_ref!r} is not defined in schema_matrices"
            )
        rationale = contract.get("rationale")
        if not isinstance(rationale, dict) or not rationale:
            raise ValueError(
                f"{source}: scenario {scenario_id} target_only contract requires rationale"
            )
        if rationale.get("baseline_support") != "unsupported":
            raise ValueError(
                f"{source}: scenario {scenario_id} target_only rationale must set "
                "baseline_support: unsupported"
            )
    elif "rollback_safe_matrix_ref" in contract:
        raise ValueError(
            f"{source}: scenario {scenario_id} round_trip contract must not set "
            "rollback_safe_matrix_ref"
        )


def _validate_capability_qualifications(
    manifest: dict[str, Any], *, source: str
) -> None:
    qualifications = manifest["capability_qualifications"]
    image_aliases = manifest["image_aliases"]
    catalog = load_capability_catalog(ROOT / "manifests" / "capability_catalog.yaml")
    for image_ref, qualification in qualifications.items():
        if image_ref not in image_aliases:
            raise ValueError(
                f"{source}: capability qualification image_ref {image_ref!r} "
                "is not defined in image_aliases"
            )
        if not isinstance(qualification, dict):
            raise ValueError(
                f"{source}: capability qualification {image_ref!r} must be a mapping"
            )
        immutable_image = qualification.get("immutable_image")
        if (
            not isinstance(immutable_image, str)
            or not _image_is_digest_pinned(immutable_image)
            or immutable_image != image_aliases[image_ref].get("image")
        ):
            raise ValueError(
                f"{source}: capability qualification {image_ref!r} must match its "
                "immutable image alias exactly and be digest-pinned"
            )
        capabilities = qualification.get("capabilities")
        if not isinstance(capabilities, dict) or not capabilities:
            raise ValueError(
                f"{source}: capability qualification {image_ref!r} requires capabilities"
            )
        for capability, result in capabilities.items():
            if capability not in catalog:
                raise ValueError(
                    f"{source}: capability qualification {image_ref!r} references "
                    f"unknown capability {capability!r}"
                )
            if not isinstance(result, dict) or result.get("status") != "passed":
                raise ValueError(
                    f"{source}: capability qualification {image_ref!r} {capability} "
                    "must record status: passed"
                )
            evidence = result.get("evidence")
            if not isinstance(evidence, dict) or not evidence:
                raise ValueError(
                    f"{source}: capability qualification {image_ref!r} {capability} "
                    "requires topology evidence"
                )
            unknown_topologies = sorted(set(evidence) - {"standalone", "cluster"})
            if unknown_topologies or any(
                not _qualification_evidence_is_stable(value)
                for value in evidence.values()
            ):
                raise ValueError(
                    f"{source}: capability qualification {image_ref!r} {capability} "
                    "evidence must use a stable argo:// or https:// URI for "
                    "standalone/cluster references"
                )


def _compile_index_engine_contract(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    resolved = deepcopy(scenario)
    raw_contract = scenario.get("index_engine_contract")
    if raw_contract is None:
        resolved["index_engine_contract"] = deepcopy(NO_INDEX_ENGINE_CONTRACT)
        return resolved

    contract = deepcopy(raw_contract)
    mode = contract["mode"]
    matrix_ref = contract["matrix_ref"]
    vector_version = contract["vector_version"]
    scalar_version = contract["scalar_version"]
    resolved["forward_workload_enabled"] = True
    resolved["rollback_enabled"] = True
    resolved["forward_schema_matrix_ref"] = matrix_ref
    if mode == "target_only":
        resolved["schema_matrix_ref"] = contract["rollback_safe_matrix_ref"]
        resolved["rollback_forward_validation_enabled"] = False
        resolved["drop_forward_before_rollback_enabled"] = True
        version_phases = ("target",)
        contract["qualification_status"] = "unsupported"
    else:
        resolved["schema_matrix_ref"] = matrix_ref
        resolved["rollback_forward_validation_enabled"] = True
        resolved["drop_forward_before_rollback_enabled"] = False
        version_phases = ("base", "target", "rollback")
        contract["qualification_status"] = "pending"
    for phase in version_phases:
        resolved.setdefault(phase, {})["target_vec_index_version"] = vector_version
        resolved[phase]["target_scalar_index_version"] = scalar_version
    resolved["index_engine_contract"] = contract
    return resolved


def _resolve_index_engine_qualification(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> None:
    contract = scenario["index_engine_contract"]
    if contract["mode"] != "round_trip":
        return
    capability = contract["capability"]
    topology = str(scenario["mode"])
    qualifications = manifest.get("capability_qualifications") or {}
    for phase in ("base", "rollback"):
        image_ref = scenario[phase].get("image_ref")
        qualification = qualifications.get(image_ref)
        if not isinstance(qualification, dict):
            raise ValueError(
                f"{scenario['id']}: round_trip {phase} image_ref {image_ref!r} "
                "requires capability qualification"
            )
        qualified_image = str(qualification.get("immutable_image") or "")
        actual_image = str(scenario[phase]["image"])
        if (
            not _image_is_digest_pinned(qualified_image)
            or qualified_image != actual_image
        ):
            raise ValueError(
                f"{scenario['id']}: round_trip {phase} image {actual_image!r} does "
                f"not match qualified immutable image {qualified_image!r}"
            )
        capability_result = (qualification.get("capabilities") or {}).get(capability)
        if (
            not isinstance(capability_result, dict)
            or capability_result.get("status") != "passed"
        ):
            raise ValueError(
                f"{scenario['id']}: {phase} image_ref {image_ref!r} has no passed "
                f"qualification for {capability}"
            )
        evidence = capability_result.get("evidence") or {}
        if not isinstance(evidence, dict) or not _qualification_evidence_is_stable(
            evidence.get(topology)
        ):
            raise ValueError(
                f"{scenario['id']}: {phase} {capability} qualification requires "
                f"{topology} evidence as a stable argo:// or https:// URI"
            )
    contract["qualification_status"] = "passed"


def _validate_resolved_index_engine_contract(scenario: dict[str, Any]) -> None:
    contract = scenario.get("index_engine_contract") or NO_INDEX_ENGINE_CONTRACT
    if contract["mode"] == "none":
        return
    capability = contract["capability"]
    catalog = load_capability_catalog(ROOT / "manifests" / "capability_catalog.yaml")
    if capability not in catalog:
        raise ValueError(f"{scenario['id']}: unknown index capability {capability!r}")

    matrix_specs = load_schema_matrix(
        _schema_matrix_path(scenario["forward_schema_matrix"])
    )
    if not matrix_specs or any(
        capability not in spec.required_capabilities for spec in matrix_specs
    ):
        raise ValueError(
            f"{scenario['id']}: contract matrix schemas must all require {capability}"
        )
    expected_versions = {
        "target_vec_index_version": contract["vector_version"],
        "target_scalar_index_version": contract["scalar_version"],
    }
    for spec in matrix_specs:
        for field, expected in expected_versions.items():
            if spec.validator_params.get(field) != expected:
                raise ValueError(
                    f"{scenario['id']}: schema {spec.name} validator_params.{field} "
                    f"must be {expected}"
                )

    mode = contract["mode"]
    if mode == "target_only":
        safe_specs = load_schema_matrix(_schema_matrix_path(scenario["schema_matrix"]))
        non_safe_specs = [
            spec.name for spec in safe_specs if spec.compat_mode != "rollback_safe"
        ]
        if not safe_specs or non_safe_specs:
            details = ", ".join(non_safe_specs) if non_safe_specs else "<empty matrix>"
            raise ValueError(
                f"{scenario['id']}: rollback-safe matrix must contain only "
                f"rollback_safe schemas; invalid schemas: {details}"
            )
        index_engine_capabilities = {
            capability_id
            for capability_id in catalog
            if capability_id.startswith("IndexEngine")
        }
        unsafe_capabilities = sorted(
            {
                required
                for spec in safe_specs
                for required in spec.required_capabilities
                if required in index_engine_capabilities
            }
        )
        if unsafe_capabilities:
            raise ValueError(
                f"{scenario['id']}: rollback-safe matrix must not require "
                "index-engine capabilities; found: " + ", ".join(unsafe_capabilities)
            )
        expected_phases = {
            "base": (-1, -1),
            "target": (contract["vector_version"], contract["scalar_version"]),
            "rollback": (-1, -1),
        }
        expected_flags = (False, True, "unsupported")
    else:
        if scenario["schema_matrix"] != scenario["forward_schema_matrix"]:
            raise ValueError(
                f"{scenario['id']}: round_trip must use the contract matrix in all phases"
            )
        expected_phases = {
            phase: (contract["vector_version"], contract["scalar_version"])
            for phase in ("base", "target", "rollback")
        }
        expected_flags = (True, False, "passed")
    rollback_validate, drop_forward, qualification_status = expected_flags
    if (
        scenario.get("forward_workload_enabled") is not True
        or scenario.get("rollback_enabled") is not True
        or scenario.get("rollback_forward_validation_enabled") is not rollback_validate
        or scenario.get("drop_forward_before_rollback_enabled") is not drop_forward
        or contract.get("qualification_status") != qualification_status
    ):
        raise ValueError(
            f"{scenario['id']}: compiled index engine contract flags drifted"
        )
    for phase, (vector_version, scalar_version) in expected_phases.items():
        if (
            scenario[phase].get("target_vec_index_version", -1) != vector_version
            or scenario[phase].get("target_scalar_index_version", -1) != scalar_version
        ):
            raise ValueError(
                f"{scenario['id']}: compiled {phase} index engine versions drifted"
            )


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

    resolved = _compile_index_engine_contract(manifest, scenario)
    resolved["workflow_template"] = _resolve_ref(
        manifest, "workflow_templates", resolved, "workflow_template"
    )
    resolved["deploy_profile"] = deploy_profile_override or _resolve_ref(
        manifest, "deploy_profiles", resolved, "deploy_profile"
    )
    resolved["schema_matrix"] = _resolve_ref(
        manifest, "schema_matrices", resolved, "schema_matrix"
    )
    if scenario.get("submit_generate_name") is not None:
        resolved["submit_generate_name"] = str(scenario["submit_generate_name"])
    if "forward_schema_matrix_ref" in resolved or "forward_schema_matrix" in resolved:
        resolved["forward_schema_matrix"] = _resolve_ref(
            manifest, "schema_matrices", resolved, "forward_schema_matrix"
        )
    else:
        resolved["forward_schema_matrix"] = resolved["schema_matrix"]

    unknown_phases = sorted(set(phase_overrides or {}) - {"base", "target", "rollback"})
    if unknown_phases:
        raise ValueError(
            f"{scenario_id}: unsupported phase overrides: {', '.join(unknown_phases)}"
        )

    for phase in ("base", "target", "rollback"):
        resolved[phase] = _resolve_phase(manifest, resolved, phase)
        override = (phase_overrides or {}).get(phase) or {}
        unknown = sorted(set(override) - {"image", "version"})
        if unknown:
            raise ValueError(
                f"{scenario_id}: unsupported {phase} override fields: "
                f"{', '.join(unknown)}"
            )
        for field in ("image", "version"):
            if override.get(field):
                if (
                    resolved.get("classification") == "candidate"
                    and resolved[phase].get("vortex_compatibility")
                    and str(override[field]) != str(resolved[phase][field])
                ):
                    raise ValueError(
                        f"{scenario_id}: {phase} reviewed candidate image is locked; "
                        "update the manifest alias through code review instead of "
                        f"overriding {field}"
                    )
                if (
                    field == "image"
                    and resolved.get("classification")
                    in STRICT_LIFECYCLE_CLASSIFICATIONS
                ):
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

    _resolve_index_engine_qualification(manifest, resolved)

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
        "scenario-classification": str(scenario["classification"]),
        "scenario-support-status": str(scenario["support_status"]),
        "release-gate-eligible": _bool_str(
            scenario.get("classification") == "gate"
            and scenario.get("support_status") in RELEASE_GATE_SUPPORT_STATUSES
        ),
        "index-engine-contract-mode": str(
            scenario.get("index_engine_contract", NO_INDEX_ENGINE_CONTRACT)["mode"]
        ),
        "index-engine-capability": str(
            scenario.get("index_engine_contract", NO_INDEX_ENGINE_CONTRACT)[
                "capability"
            ]
        ),
        "index-engine-qualification-status": str(
            scenario.get("index_engine_contract", NO_INDEX_ENGINE_CONTRACT)[
                "qualification_status"
            ]
        ),
        "deploy-profile": str(scenario["deploy_profile"]),
        "milvus-log-level": str(
            scenario.get("milvus_log_level", defaults.get("milvus_log_level", "debug"))
        ),
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
        "drop-forward-before-rollback-enabled": _bool_str(
            scenario.get(
                "drop_forward_before_rollback_enabled",
                scenario.get("forward_workload_enabled", False)
                and scenario.get("rollback_enabled", True)
                and not scenario.get("rollback_forward_validation_enabled", False),
            )
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
        drift = {
            name: {
                "expected": expected_value,
                "actual": str(runtime_parameters.get(name, "<missing>")),
            }
            for name, expected_value in UNREGISTERED_SCENARIO_METADATA.items()
            if str(runtime_parameters.get(name, "<missing>")) != expected_value
        }
        if drift:
            details = ", ".join(
                f"{name}: expected {values['expected']!r}, got {values['actual']!r}"
                for name, values in sorted(drift.items())
            )
            raise ValueError(
                f"{scenario_id}: unregistered scenario report metadata must remain "
                f"fail-closed: {details}"
            )
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
    if manifest.get("version") != "2":
        raise ValueError(f"{source}: version must be '2'")
    for section in (
        "defaults",
        "workflow_templates",
        "deploy_profiles",
        "schema_matrices",
        "image_aliases",
    ):
        if not isinstance(manifest.get(section), dict) or not manifest[section]:
            raise ValueError(f"{source}: {section} must be a non-empty mapping")
    qualifications = manifest.get("capability_qualifications")
    if not isinstance(qualifications, dict):
        raise ValueError(f"{source}: capability_qualifications must be a mapping")
    _validate_capability_qualifications(manifest, source=source)
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
        _validate_raw_index_engine_contract(manifest, scenario, source=source)
        _validate_topology_requirements(scenario, source=source)
        if (
            scenario.get("allow_unsafe_negative_coverage") is True
            and scenario.get("classification") != "negative"
        ):
            raise ValueError(
                f"{source}: scenario {scenario_id} may enable allow_unsafe_negative_coverage only "
                "when classification is negative"
            )
        classification = str(scenario.get("classification"))
        if classification not in {"gate", "candidate", "negative", "known_limitation"}:
            raise ValueError(
                f"{source}: scenario {scenario_id} has unsupported classification "
                f"{classification!r}"
            )
        if (
            classification == "candidate"
            and scenario.get("support_status") != VORTEX_CANDIDATE_SUPPORT_STATUS
        ):
            raise ValueError(
                f"{source}: candidate scenario {scenario_id} must use support_status "
                f"{VORTEX_CANDIDATE_SUPPORT_STATUS!r}"
            )
        if (
            classification == "gate"
            and scenario.get("support_status") not in RELEASE_GATE_SUPPORT_STATUSES
        ):
            raise ValueError(
                f"{source}: gate scenario {scenario_id} must use a release-eligible "
                f"support_status from {sorted(RELEASE_GATE_SUPPORT_STATUSES)!r}"
            )
        refs = [
            ("workflow_templates", "workflow_template"),
            ("deploy_profiles", "deploy_profile"),
        ]
        if "index_engine_contract" not in scenario:
            refs.append(("schema_matrices", "schema_matrix"))
        for section, logical_name in refs:
            _resolve_ref(manifest, section, scenario, logical_name)
        for phase in ("base", "target", "rollback"):
            _resolve_phase(manifest, scenario, phase)


def validate_resolved_gate_scenario(scenario: dict[str, Any]) -> None:
    _validate_scenario_execution_mode(scenario)
    _validate_phase_image_versions(scenario)
    _validate_resolved_index_engine_contract(scenario)
    if scenario.get("classification") not in STRICT_LIFECYCLE_CLASSIFICATIONS:
        _validate_vortex_loon_dependency(scenario)
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
        _validate_vortex_loon_dependency(scenario)
        _validate_vortex_compatibility_contract(scenario)
        return

    # The 2.6 -> 3.0 -> 2.6 gate path does not call
    # _validate_vortex_loon_dependency above because the blocked-flags check below
    # already rejects storage_v3/vortex in every phase, which implies the V => L
    # dependency holds trivially.
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
    if scenario.get("classification") in STRICT_LIFECYCLE_CLASSIFICATIONS:
        lifecycle_kind = str(scenario.get("classification"))
        mutable_images = [
            f"{phase}.image={scenario[phase]['image']}"
            for phase in ("base", "target", "rollback")
            if not image_is_immutable(str(scenario[phase].get("image", "")))
        ]
        if mutable_images:
            raise ValueError(
                f"{scenario['id']}: runnable {lifecycle_kind} contains mutable images: "
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
        for field in CANDIDATE_ALIAS_METADATA_FIELDS:
            if field in alias:
                payload[field] = alias[field]
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
    if scenario.get("classification") == "gate":
        mutable_dependency_images = [
            f"{path}={image}"
            for path, image in _explicit_profile_images(
                profile.get("helm_values", {}), ("helm_values",)
            )
            if not _image_is_digest_pinned(image)
        ]
        if mutable_dependency_images:
            raise ValueError(
                f"{scenario_id}: release gate deploy profile dependency image must be "
                "digest-pinned: " + ", ".join(mutable_dependency_images)
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


def _has_reviewed_vortex_candidate_metadata(
    scenario: dict[str, Any], phase: str
) -> bool:
    payload = scenario[phase]
    source_commit = str(payload.get("source_commit") or "")
    storage_commit = str(payload.get("milvus_storage_commit") or "")
    image_tag = str(payload.get("image") or "").split("@", 1)[0].rsplit(":", 1)[-1]
    return (
        scenario.get("classification") == "candidate"
        and scenario.get("support_status") == VORTEX_CANDIDATE_SUPPORT_STATUS
        and payload.get("vortex_compatibility") == VORTEX_CANDIDATE_COMPATIBILITY
        and FULL_GIT_SHA.fullmatch(source_commit) is not None
        and FULL_GIT_SHA.fullmatch(storage_commit) is not None
        and image_tag.endswith(f"-{source_commit[:8]}")
        and image_is_immutable(str(payload.get("image") or ""))
    )


def _phase_supports_vortex(scenario: dict[str, Any], phase: str) -> bool:
    return version_at_least(
        str(scenario[phase]["version"]), VORTEX_MIN_SUPPORTED_VERSION
    ) or _has_reviewed_vortex_candidate_metadata(scenario, phase)


def _validate_vortex_compatibility_contract(scenario: dict[str, Any]) -> None:
    for phase in ("base", "target", "rollback"):
        if not scenario[phase].get("vortex_enabled", False):
            continue
        if _phase_supports_vortex(scenario, phase):
            continue
        if scenario.get("classification") == "candidate":
            raise ValueError(
                f"{scenario['id']}: {phase} Vortex requires reviewed Vortex "
                "compatibility metadata for the locked pre-release candidate image"
            )
        raise ValueError(
            f"{scenario['id']}: {phase} Vortex requires Milvus >= "
            f"{VORTEX_MIN_SUPPORTED_VERSION}; got {scenario[phase]['version']}"
        )

    vortex_data_may_exist = any(
        scenario[phase].get("vortex_enabled", False) for phase in ("base", "target")
    )
    if (
        scenario.get("rollback_enabled", True)
        and vortex_data_may_exist
        and not _phase_supports_vortex(scenario, "rollback")
    ):
        raise ValueError(
            f"{scenario['id']}: rollback must support Vortex data written before "
            f"rollback; Milvus >= {VORTEX_MIN_SUPPORTED_VERSION} or a reviewed "
            "pre-release candidate image is required"
        )


def _validate_vortex_loon_dependency(scenario: dict[str, Any]) -> None:
    for phase in ("base", "target", "rollback"):
        if scenario[phase].get("vortex_enabled") and not scenario[phase].get(
            "loon_ffi_enabled"
        ):
            raise ValueError(
                f"{scenario['id']}: {phase} Vortex requires LoonFFI "
                "(vortex_enabled=true implies loon_ffi_enabled=true)"
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
        "drop_forward_before_rollback_enabled",
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
