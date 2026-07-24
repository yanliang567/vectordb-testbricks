from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from milvus_client.common.args import parse_bool


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _required_validation_names(config_matrix: dict[str, Any]) -> list[str]:
    required = ["validate_before_upgrade", "validate_after_upgrade"]
    if config_matrix["forward_workload_enabled"]:
        required.append("validate_forward_after_upgrade")
    if (
        config_matrix["rollback_enabled"]
        and config_matrix["index_compatibility_validation_enabled"]
    ):
        required.append("validate_index_compatibility_after_upgrade")
    if (
        config_matrix["rollback_enabled"]
        and config_matrix["phase_dml_dql_validation_enabled"]
    ):
        required.append("validate_phase_dml_dql_after_upgrade")
    if config_matrix["rollback_enabled"]:
        required.append("validate_after_rollback")
    if (
        config_matrix["rollback_enabled"]
        and config_matrix["index_compatibility_validation_enabled"]
    ):
        required.append("validate_index_compatibility_after_rollback")
    if (
        config_matrix["rollback_enabled"]
        and config_matrix["phase_dml_dql_validation_enabled"]
    ):
        required.append("validate_phase_dml_dql_after_rollback")
    if (
        config_matrix["rollback_enabled"]
        and config_matrix["forward_workload_enabled"]
        and config_matrix["rollback_forward_validation_enabled"]
    ):
        required.append("validate_forward_after_rollback")
    return required


def _required_serviceability_names(config_matrix: dict[str, Any]) -> list[str]:
    if not config_matrix["rollback_enabled"]:
        return []
    required = ["wait_rollback_serviceability"]
    if (
        config_matrix["forward_workload_enabled"]
        and config_matrix["rollback_forward_validation_enabled"]
    ):
        required.append("wait_forward_rollback_serviceability")
    return required


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    k8s_dir = Path(args.k8s_dir)
    pressure = _load_json(Path(args.pressure_summary), {})
    env = _load_json(Path(args.env_snapshot), {})
    flow = _load_json(Path(args.flow_summary), {})
    deploy_topology = (
        _load_json(Path(args.deploy_topology), {}) if args.deploy_topology else {}
    )

    results = {}
    for path in sorted(results_dir.glob("*.json")):
        results[path.stem] = _load_json(
            path, {"status": "unreadable", "file": path.name}
        )

    config_matrix = {
        "base_json_shredding_enabled": parse_bool(args.base_json_shredding_enabled),
        "target_json_shredding_enabled": parse_bool(args.target_json_shredding_enabled),
        "rollback_json_shredding_enabled": parse_bool(
            args.rollback_json_shredding_enabled
        ),
        "base_loon_ffi_enabled": parse_bool(args.base_loon_ffi_enabled),
        "target_loon_ffi_enabled": parse_bool(args.target_loon_ffi_enabled),
        "rollback_loon_ffi_enabled": parse_bool(args.rollback_loon_ffi_enabled),
        "base_vortex_enabled": parse_bool(args.base_vortex_enabled),
        "target_vortex_enabled": parse_bool(args.target_vortex_enabled),
        "rollback_vortex_enabled": parse_bool(args.rollback_vortex_enabled),
        "post_upgrade_config_toggle_enabled": parse_bool(
            args.post_upgrade_config_toggle_enabled
        ),
        "post_upgrade_json_shredding_enabled": parse_bool(
            args.post_upgrade_json_shredding_enabled
        ),
        "forward_workload_enabled": parse_bool(args.forward_workload_enabled),
        "forward_schema_matrix": args.forward_schema_matrix,
        "rollback_enabled": parse_bool(args.rollback_enabled),
        "rollback_forward_validation_enabled": parse_bool(
            args.rollback_forward_validation_enabled
        ),
        "index_compatibility_validation_enabled": parse_bool(
            args.index_compatibility_validation_enabled
        ),
        "phase_dml_dql_validation_enabled": parse_bool(
            args.phase_dml_dql_validation_enabled
        ),
        "phase_new_collection_rows": args.phase_new_collection_rows,
        "phase_existing_dml_rows": args.phase_existing_dml_rows,
        "phase_existing_delete_rows": args.phase_existing_delete_rows,
        "schema_evolution_existing_enabled": parse_bool(
            args.schema_evolution_existing_enabled
        ),
        "schema_evolution_forward_enabled": parse_bool(
            args.schema_evolution_forward_enabled
        ),
    }
    validation = {
        name: payload
        for name, payload in results.items()
        if name.startswith("validate_")
    }
    missing_validations = [
        name
        for name in _required_validation_names(config_matrix)
        if name not in validation
    ]
    for name in missing_validations:
        validation[name] = {
            "status": "missing",
            "failures": [
                {
                    "type": "VALIDATION_RESULT_MISSING",
                    "message": "required validation result json is missing",
                    "validation": name,
                }
            ],
        }
    serviceability = {
        name: payload
        for name, payload in results.items()
        if payload.get("brick") == "wait_data_serviceability"
        or name.endswith("_serviceability")
    }
    missing_serviceability = [
        name
        for name in _required_serviceability_names(config_matrix)
        if name not in serviceability
    ]
    for name in missing_serviceability:
        serviceability[name] = {
            "status": "missing",
            "failures": [
                {
                    "type": "SERVICEABILITY_RESULT_MISSING",
                    "message": "required serviceability wait result json is missing",
                    "serviceability": name,
                }
            ],
        }
    failed_results = {
        name: payload
        for name, payload in {**results, **validation, **serviceability}.items()
        if payload.get("status") not in {"passed", "skipped"}
    }
    validation_passed = bool(validation) and all(
        payload.get("status") in {"passed", "skipped"}
        for payload in validation.values()
    )
    pressure_failed = int(pressure.get("failed", 0) or 0)
    pressure_fail_on_error = parse_bool(args.pressure_fail_on_error)

    status = "passed"
    if (
        failed_results
        or not validation_passed
        or (pressure_fail_on_error and pressure_failed)
    ):
        status = "failed"
    elif pressure_failed:
        status = "warning"

    k8s_files = {
        path.name: str(path) for path in sorted(k8s_dir.glob("*")) if path.is_file()
    }

    return {
        "status": status,
        "workflow": {
            "name": args.workflow_name,
            "uid": args.workflow_uid,
            "namespace": args.workflow_namespace,
        },
        "target": {
            "milvus_release_name": args.milvus_release_name,
            "milvus_namespace": args.milvus_namespace,
            "milvus_host": args.milvus_host,
            "base_milvus_image": args.base_milvus_image,
            "rollback_milvus_image": args.rollback_milvus_image,
            "target_milvus_image": args.target_milvus_image,
            "base_version": args.base_version,
            "rollback_version": args.rollback_version,
            "target_version": args.target_version,
        },
        "parameters": {
            "scenario_id": args.scenario_id,
            "repo_url": args.repo_url,
            "repo_revision": args.repo_revision,
            "deploy_profile": args.deploy_profile,
            "schema_matrix": args.schema_matrix,
            "collection_prefix": args.collection_prefix,
            "forward_collection_prefix": args.forward_collection_prefix,
            "rows_per_collection": args.rows_per_collection,
            "batch_size": args.batch_size,
            "pressure_modules": args.pressure_modules.split(),
            "pressure_fail_on_error": pressure_fail_on_error,
            "observe_before_upgrade_sec": args.observe_before_upgrade_sec,
            "observe_after_upgrade_sec": args.observe_after_upgrade_sec,
            "observe_before_rollback_sec": args.observe_before_rollback_sec,
            "observe_after_rollback_sec": args.observe_after_rollback_sec,
            "rollback_serviceability_timeout_sec": args.rollback_serviceability_timeout_sec,
            "rollback_serviceability_interval_sec": args.rollback_serviceability_interval_sec,
            "config_matrix": config_matrix,
        },
        "validation": {
            "passed": validation_passed,
            "results": {
                name: {
                    "status": payload.get("status"),
                    "failures": payload.get("failures", []),
                    "metrics": payload.get("metrics", {}),
                }
                for name, payload in validation.items()
            },
        },
        "serviceability": {
            "results": {
                name: {
                    "status": payload.get("status"),
                    "failures": payload.get("failures", []),
                    "metrics": payload.get("metrics", {}),
                }
                for name, payload in serviceability.items()
            },
        },
        "pressure": pressure,
        "deploy_topology": deploy_topology,
        "failed_results": {
            name: {
                "status": payload.get("status"),
                "failures": payload.get("failures", []),
                "metrics": payload.get("metrics", {}),
            }
            for name, payload in failed_results.items()
        },
        "k8s_snapshot": k8s_files,
        "env_snapshot": env,
        "flow_summary": flow,
    }


def build_markdown(report: dict[str, Any]) -> str:
    params = report["parameters"]
    target = report["target"]
    workflow = report["workflow"]
    validation = report["validation"]["results"]
    pressure = report.get("pressure", {})
    serviceability = report.get("serviceability", {}).get("results", {})
    config_matrix = params.get("config_matrix", {})

    validation_lines = [
        f"- `{name}`: {payload.get('status')}"
        for name, payload in sorted(validation.items())
    ] or ["- no validation results found"]

    pressure_lines = [
        f"- total result files: {pressure.get('total', 0)}",
        f"- passed result files: {pressure.get('passed', 0)}",
        f"- failed/warning result files: {pressure.get('failed', 0)}",
        f"- maintenance-window excluded failures: {pressure.get('excluded_failed', 0)}",
        f"- fail_on_error: {pressure.get('fail_on_error')}",
    ]
    for failed in pressure.get("failed_results", []):
        pressure_lines.append(
            f"- warning `{failed.get('file')}` `{failed.get('brick')}`: {failed.get('status')}"
        )
    for excluded in pressure.get("excluded_failed_results", []):
        window = excluded.get("maintenance_window", {})
        pressure_lines.append(
            f"- excluded `{excluded.get('file')}` `{excluded.get('brick')}`: "
            f"{excluded.get('status')} during `{window.get('label')}`"
        )
    for window in pressure.get("maintenance_windows", []):
        pressure_lines.append(
            f"- maintenance window `{window.get('label')}`: "
            f"duration_sec=`{window.get('duration_sec')}`"
        )
    serviceability_lines = []
    for name, payload in sorted(serviceability.items()):
        metrics = payload.get("metrics", {})
        serviceability_lines.append(
            f"- `{name}`: {payload.get('status')}, recovered=`{metrics.get('recovered')}`, "
            f"recovery_duration_sec=`{metrics.get('recovery_duration_sec')}`, attempts=`{metrics.get('attempts')}`"
        )
    if not serviceability_lines:
        serviceability_lines = ["- no serviceability wait results found"]
    config_lines = [
        f"- base jsonShredding: `{config_matrix.get('base_json_shredding_enabled')}`",
        f"- target jsonShredding: `{config_matrix.get('target_json_shredding_enabled')}`",
        f"- rollback jsonShredding: `{config_matrix.get('rollback_json_shredding_enabled')}`",
        f"- base LoonFFI/storage v3: `{config_matrix.get('base_loon_ffi_enabled')}`",
        f"- target LoonFFI/storage v3: `{config_matrix.get('target_loon_ffi_enabled')}`",
        f"- rollback LoonFFI/storage v3: `{config_matrix.get('rollback_loon_ffi_enabled')}`",
        f"- base vortex: `{config_matrix.get('base_vortex_enabled')}`",
        f"- target vortex: `{config_matrix.get('target_vortex_enabled')}`",
        f"- rollback vortex: `{config_matrix.get('rollback_vortex_enabled')}`",
        f"- post-upgrade config toggle: `{config_matrix.get('post_upgrade_config_toggle_enabled')}`",
        f"- post-upgrade jsonShredding: `{config_matrix.get('post_upgrade_json_shredding_enabled')}`",
        f"- forward workload: `{config_matrix.get('forward_workload_enabled')}`",
        f"- forward schema matrix: `{config_matrix.get('forward_schema_matrix')}`",
        f"- rollback enabled: `{config_matrix.get('rollback_enabled')}`",
        f"- rollback forward validation: `{config_matrix.get('rollback_forward_validation_enabled')}`",
        f"- index compatibility validation: `{config_matrix.get('index_compatibility_validation_enabled')}`",
        f"- phase DML/DQL validation: `{config_matrix.get('phase_dml_dql_validation_enabled')}`",
        f"- phase new collection rows/schema: `{config_matrix.get('phase_new_collection_rows')}`",
        f"- phase existing DML rows/schema: `{config_matrix.get('phase_existing_dml_rows')}`",
        f"- phase existing delete rows/schema: `{config_matrix.get('phase_existing_delete_rows')}`",
        f"- schema evolution existing: `{config_matrix.get('schema_evolution_existing_enabled')}`",
        f"- schema evolution forward: `{config_matrix.get('schema_evolution_forward_enabled')}`",
        f"- rollback serviceability timeout sec: `{params.get('rollback_serviceability_timeout_sec')}`",
        f"- rollback serviceability interval sec: `{params.get('rollback_serviceability_interval_sec')}`",
    ]

    lines = [
        "# Milvus Standalone Upgrade/Rollback Report",
        "",
        f"- workflow: `{workflow['name']}`",
        f"- status: `{report['status']}`",
        f"- scenario: `{params.get('scenario_id')}`",
        f"- deploy profile: `{params.get('deploy_profile')}`",
        f"- collection prefix: `{params['collection_prefix']}`",
        f"- forward collection prefix: `{params['forward_collection_prefix']}`",
        f"- rows per collection: `{params['rows_per_collection']}`",
        f"- base image: `{target['base_milvus_image']}`",
        f"- base version: `{target['base_version']}`",
        f"- target image: `{target['target_milvus_image']}`",
        f"- target version: `{target['target_version']}`",
        f"- rollback image: `{target['rollback_milvus_image']}`",
        f"- rollback version: `{target['rollback_version']}`",
        "",
        "## Config Matrix",
        *config_lines,
        "",
        "## Validation",
        *validation_lines,
        "",
        "## Serviceability Recovery",
        *serviceability_lines,
        "",
        "## Pressure",
        *pressure_lines,
        "",
        "## Artifacts",
        "- raw brick results: `/tmp/milvus-bricks/results`",
        "- pressure results: `/tmp/milvus-bricks/pressure-results`",
        "- k8s snapshot: `/tmp/milvus-bricks/k8s`",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a merged workflow report")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--pressure-summary", required=True)
    parser.add_argument("--k8s-dir", required=True)
    parser.add_argument("--env-snapshot", required=True)
    parser.add_argument("--flow-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--workflow-uid", required=True)
    parser.add_argument("--workflow-namespace", required=True)
    parser.add_argument("--milvus-release-name", required=True)
    parser.add_argument("--milvus-namespace", required=True)
    parser.add_argument("--milvus-host", required=True)
    parser.add_argument("--base-milvus-image", required=True)
    parser.add_argument("--rollback-milvus-image", required=True)
    parser.add_argument("--target-milvus-image", required=True)
    parser.add_argument("--base-version", required=True)
    parser.add_argument("--rollback-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--deploy-profile", default="")
    parser.add_argument("--deploy-topology", default="")
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--collection-prefix", required=True)
    parser.add_argument("--forward-collection-prefix", required=True)
    parser.add_argument("--rows-per-collection", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--pressure-modules", required=True)
    parser.add_argument("--pressure-fail-on-error", required=True)
    parser.add_argument("--observe-before-upgrade-sec", type=int, required=True)
    parser.add_argument("--observe-after-upgrade-sec", type=int, required=True)
    parser.add_argument("--observe-before-rollback-sec", type=int, required=True)
    parser.add_argument("--observe-after-rollback-sec", type=int, required=True)
    parser.add_argument(
        "--rollback-serviceability-timeout-sec", type=int, required=True
    )
    parser.add_argument(
        "--rollback-serviceability-interval-sec", type=int, required=True
    )
    parser.add_argument("--base-json-shredding-enabled", default="false")
    parser.add_argument("--target-json-shredding-enabled", default="false")
    parser.add_argument("--rollback-json-shredding-enabled", default="false")
    parser.add_argument("--base-loon-ffi-enabled", default="false")
    parser.add_argument("--target-loon-ffi-enabled", default="false")
    parser.add_argument("--rollback-loon-ffi-enabled", default="false")
    parser.add_argument("--base-vortex-enabled", default="false")
    parser.add_argument("--target-vortex-enabled", default="false")
    parser.add_argument("--rollback-vortex-enabled", default="false")
    parser.add_argument("--post-upgrade-config-toggle-enabled", default="false")
    parser.add_argument("--post-upgrade-json-shredding-enabled", default="false")
    parser.add_argument("--forward-workload-enabled", default="false")
    parser.add_argument("--forward-schema-matrix", default="")
    parser.add_argument("--rollback-enabled", default="true")
    parser.add_argument("--rollback-forward-validation-enabled", default="false")
    parser.add_argument("--index-compatibility-validation-enabled", default="false")
    parser.add_argument("--phase-dml-dql-validation-enabled", default="false")
    parser.add_argument("--phase-new-collection-rows", type=int, default=0)
    parser.add_argument("--phase-existing-dml-rows", type=int, default=0)
    parser.add_argument("--phase-existing-delete-rows", type=int, default=0)
    parser.add_argument("--schema-evolution-existing-enabled", default="false")
    parser.add_argument("--schema-evolution-forward-enabled", default="false")
    parser.add_argument(
        "--soft-fail", action="store_true", help="Write failed report status but exit 0"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_report(args)
    _write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(build_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.soft_fail:
        return 0
    return 0 if report["status"] in {"passed", "warning"} else 1


if __name__ == "__main__":
    sys.exit(main())
