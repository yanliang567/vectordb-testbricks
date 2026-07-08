from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from milvus_client.common.args import parse_bool


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _required_validation_names(config_matrix: dict[str, Any]) -> list[str]:
    required = ["validate_before_upgrade", "validate_after_upgrade"]
    if config_matrix["forward_workload_enabled"]:
        required.append("validate_forward_after_upgrade")
    if config_matrix["rollback_enabled"]:
        required.append("validate_after_rollback")
    if (
        config_matrix["rollback_enabled"]
        and config_matrix["forward_workload_enabled"]
        and config_matrix["rollback_forward_validation_enabled"]
    ):
        required.append("validate_forward_after_rollback")
    return required


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    k8s_dir = Path(args.k8s_dir)
    pressure = _load_json(Path(args.pressure_summary), {})
    env = _load_json(Path(args.env_snapshot), {})
    flow = _load_json(Path(args.flow_summary), {})

    results = {}
    for path in sorted(results_dir.glob("*.json")):
        results[path.stem] = _load_json(path, {"status": "unreadable", "file": path.name})

    config_matrix = {
        "base_json_shredding_enabled": parse_bool(args.base_json_shredding_enabled),
        "target_json_shredding_enabled": parse_bool(args.target_json_shredding_enabled),
        "rollback_json_shredding_enabled": parse_bool(args.rollback_json_shredding_enabled),
        "target_loon_ffi_enabled": parse_bool(args.target_loon_ffi_enabled),
        "post_upgrade_config_toggle_enabled": parse_bool(args.post_upgrade_config_toggle_enabled),
        "post_upgrade_json_shredding_enabled": parse_bool(args.post_upgrade_json_shredding_enabled),
        "forward_workload_enabled": parse_bool(args.forward_workload_enabled),
        "forward_schema_matrix": args.forward_schema_matrix,
        "rollback_enabled": parse_bool(args.rollback_enabled),
        "rollback_forward_validation_enabled": parse_bool(args.rollback_forward_validation_enabled),
        "schema_evolution_existing_enabled": parse_bool(args.schema_evolution_existing_enabled),
        "schema_evolution_forward_enabled": parse_bool(args.schema_evolution_forward_enabled),
    }
    validation = {
        name: payload
        for name, payload in results.items()
        if name.startswith("validate_")
    }
    missing_validations = [
        name for name in _required_validation_names(config_matrix) if name not in validation
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
    failed_results = {
        name: payload
        for name, payload in {**results, **validation}.items()
        if payload.get("status") not in {"passed", "skipped"}
    }
    validation_passed = bool(validation) and all(
        payload.get("status") in {"passed", "skipped"} for payload in validation.values()
    )
    pressure_failed = int(pressure.get("failed", 0) or 0)
    pressure_fail_on_error = parse_bool(args.pressure_fail_on_error)

    status = "passed"
    if failed_results or not validation_passed or (pressure_fail_on_error and pressure_failed):
        status = "failed"
    elif pressure_failed:
        status = "warning"

    k8s_files = {
        path.name: str(path)
        for path in sorted(k8s_dir.glob("*"))
        if path.is_file()
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
            "repo_url": args.repo_url,
            "repo_revision": args.repo_revision,
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
        "pressure": pressure,
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
    config_matrix = params.get("config_matrix", {})

    validation_lines = [
        f"- `{name}`: {payload.get('status')}"
        for name, payload in sorted(validation.items())
    ] or ["- no validation results found"]

    pressure_lines = [
        f"- total result files: {pressure.get('total', 0)}",
        f"- passed result files: {pressure.get('passed', 0)}",
        f"- failed/warning result files: {pressure.get('failed', 0)}",
        f"- fail_on_error: {pressure.get('fail_on_error')}",
    ]
    for failed in pressure.get("failed_results", []):
        pressure_lines.append(
            f"- warning `{failed.get('file')}` `{failed.get('brick')}`: {failed.get('status')}"
        )
    config_lines = [
        f"- base jsonShredding: `{config_matrix.get('base_json_shredding_enabled')}`",
        f"- target jsonShredding: `{config_matrix.get('target_json_shredding_enabled')}`",
        f"- rollback jsonShredding: `{config_matrix.get('rollback_json_shredding_enabled')}`",
        f"- target LoonFFI: `{config_matrix.get('target_loon_ffi_enabled')}`",
        f"- post-upgrade config toggle: `{config_matrix.get('post_upgrade_config_toggle_enabled')}`",
        f"- post-upgrade jsonShredding: `{config_matrix.get('post_upgrade_json_shredding_enabled')}`",
        f"- forward workload: `{config_matrix.get('forward_workload_enabled')}`",
        f"- forward schema matrix: `{config_matrix.get('forward_schema_matrix')}`",
        f"- rollback enabled: `{config_matrix.get('rollback_enabled')}`",
        f"- rollback forward validation: `{config_matrix.get('rollback_forward_validation_enabled')}`",
        f"- schema evolution existing: `{config_matrix.get('schema_evolution_existing_enabled')}`",
        f"- schema evolution forward: `{config_matrix.get('schema_evolution_forward_enabled')}`",
    ]

    lines = [
        "# Milvus Standalone Upgrade/Rollback Report",
        "",
        f"- workflow: `{workflow['name']}`",
        f"- status: `{report['status']}`",
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
    parser.add_argument("--base-json-shredding-enabled", default="false")
    parser.add_argument("--target-json-shredding-enabled", default="false")
    parser.add_argument("--rollback-json-shredding-enabled", default="false")
    parser.add_argument("--target-loon-ffi-enabled", default="false")
    parser.add_argument("--post-upgrade-config-toggle-enabled", default="false")
    parser.add_argument("--post-upgrade-json-shredding-enabled", default="false")
    parser.add_argument("--forward-workload-enabled", default="false")
    parser.add_argument("--forward-schema-matrix", default="")
    parser.add_argument("--rollback-enabled", default="true")
    parser.add_argument("--rollback-forward-validation-enabled", default="false")
    parser.add_argument("--schema-evolution-existing-enabled", default="false")
    parser.add_argument("--schema-evolution-forward-enabled", default="false")
    parser.add_argument("--soft-fail", action="store_true", help="Write failed report status but exit 0")
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
