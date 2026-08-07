from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

from milvus_client.common.gates import (
    load_gate_manifest,
    render_submission,
    resolve_gate_scenario,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Argo submit parameters from a code-managed upgrade/rollback gate scenario"
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("milvus_client/manifests/upgrade_rollback_gates.yaml")),
    )
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument(
        "--repo-revision",
        default=None,
        help="Override the repository branch, tag, or commit SHA used by the workflow",
    )
    parser.add_argument(
        "--deploy-profile",
        default=None,
        help="Override the deploy profile selected by the scenario",
    )
    for phase in ("base", "target", "rollback"):
        parser.add_argument(
            f"--{phase}-milvus-image",
            default=None,
            help=f"Override the concrete Milvus image for the {phase} phase",
        )
        parser.add_argument(
            f"--{phase}-version",
            default=None,
            help=f"Override the semantic Milvus version for the {phase} phase",
        )
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Allow promoted gate scenarios to render parameters with placeholder images for dry-run/review output",
    )
    parser.add_argument(
        "--format", choices=["json", "yaml", "argo-args"], default="json"
    )
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
            phase_overrides={
                phase: {
                    key: value
                    for key, value in {
                        "image": getattr(args, f"{phase}_milvus_image"),
                        "version": getattr(args, f"{phase}_version"),
                    }.items()
                    if value
                }
                for phase in ("base", "target", "rollback")
            },
        )
        submission = render_submission(
            scenario, manifest, allow_placeholder=args.allow_placeholder
        )
        if args.repo_revision:
            submission["parameters"]["repo-revision"] = args.repo_revision
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
    chunks = []
    submit_generate_name = submission.get("submit_generate_name")
    if submit_generate_name:
        chunks.append(f"--generate-name {shlex.quote(str(submit_generate_name))}")
    chunks.append(f"--from workflowtemplate/{shlex.quote(workflow_template)}")
    for name in sorted(parameters):
        chunks.append(f"-p {shlex.quote(f'{name}={parameters[name]}')}")
    return " ".join(chunks)


if __name__ == "__main__":
    sys.exit(main())
