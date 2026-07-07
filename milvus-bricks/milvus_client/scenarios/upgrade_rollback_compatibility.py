from __future__ import annotations

from pathlib import Path
import sys

import yaml

from milvus_client.common.args import build_common_parser
from milvus_client.common.result import result_from_args


def add_args(parser):
    parser.add_argument("--scenario-manifest", default="milvus_client/manifests/scenario_upgrade_rollback.yaml")
    parser.add_argument("--dry-run", action="store_true")


def build_plan(manifest: dict) -> list[dict]:
    cycles = int(manifest.get("cycles", 1))
    steps = [
        {"name": "precheck", "phase": "before_upgrade"},
        {"name": "create_compat_schema", "phase": "before_upgrade"},
        {"name": "seed_compat_data", "phase": "before_upgrade"},
        {"name": "start_mixed_rw_pressure", "phase": "before_upgrade"},
        {"name": "start_validator_loop", "phase": "before_upgrade"},
    ]
    for cycle in range(1, cycles + 1):
        steps.extend(
            [
                {"name": "wait_upgrade", "cycle": cycle, "phase": "before_upgrade"},
                {"name": "observe_after_upgrade", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "create_forward_schema", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "validate_compat_and_forward", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "wait_rollback", "cycle": cycle, "phase": "before_rollback"},
                {"name": "observe_after_rollback", "cycle": cycle, "phase": "after_rollback"},
                {"name": "validate_compat_only", "cycle": cycle, "phase": "after_rollback"},
            ]
        )
    steps.append({"name": "final_validate_compat", "phase": "steady_state"})
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Upgrade/rollback compatibility scenario")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "upgrade_rollback_compatibility")
    manifest = yaml.safe_load(Path(args.scenario_manifest).read_text()) or {}
    plan = build_plan(manifest)
    result.metrics = {
        "scenario": manifest.get("name", "upgrade_rollback_compatibility"),
        "cycles": int(manifest.get("cycles", 1)),
        "planned_steps_total": len(plan),
        "planned_steps": plan,
    }
    if not args.dry_run:
        result.mark_failed(
            "NOT_IMPLEMENTED",
            "non-dry-run scenario execution must be composed by Argo or a future controller",
        )
        result.write(args.output_json)
        return 2
    result.write(args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

