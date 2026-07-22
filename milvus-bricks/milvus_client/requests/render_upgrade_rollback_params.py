from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import yaml

from milvus_client.common.gates import load_gate_manifest, render_submission, resolve_gate_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Argo submit parameters from a code-managed upgrade/rollback gate scenario"
    )
    parser.add_argument("--manifest", default=str(Path("milvus_client/manifests/upgrade_rollback_gates.yaml")))
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--deploy-profile", default=None, help="Override the deploy profile selected by the scenario")
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Allow promoted gate scenarios to render parameters with placeholder images for dry-run/review output",
    )
    parser.add_argument("--format", choices=["json", "yaml", "argo-args"], default="json")
    parser.add_argument("--output", default="-", help="Output path, or '-' for stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_gate_manifest(args.manifest)
        scenario = resolve_gate_scenario(
            manifest,
            args.scenario_id,
            deploy_profile_override=args.deploy_profile,
        )
        submission = render_submission(scenario, manifest, allow_placeholder=args.allow_placeholder)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        output = json.dumps(submission, indent=2, sort_keys=True)
    elif args.format == "yaml":
        output = yaml.safe_dump(submission, sort_keys=False)
    else:
        output = _render_argo_args(submission)

    if args.output == "-":
        print(output)
    else:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n")
    return 0


def _render_argo_args(submission: dict[str, object]) -> str:
    workflow_template = str(submission["workflow_template"])
    parameters = submission["parameters"]
    if not isinstance(parameters, dict):
        raise TypeError("submission.parameters must be a mapping")
    chunks = [f"--from workflowtemplate/{shlex.quote(workflow_template)}"]
    for name in sorted(parameters):
        chunks.append(f"-p {shlex.quote(f'{name}={parameters[name]}')}")
    return " ".join(chunks)


if __name__ == "__main__":
    sys.exit(main())
