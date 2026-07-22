from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from milvus_client.common.args import parse_bool
from milvus_client.common.deploy import deploy_topology_summary, dump_yaml, load_deploy_profile, render_milvus_cr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a Milvus Operator CR from a code-managed deploy profile")
    parser.add_argument("--deploy-profile", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--image-update-mode", default="all")
    parser.add_argument("--image-pull-policy", default="IfNotPresent")
    parser.add_argument("--json-shredding-enabled", type=parse_bool, default=False)
    parser.add_argument("--loon-ffi-enabled", type=parse_bool, default=False)
    parser.add_argument("--vortex-enabled", type=parse_bool, default=False)
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--workflow-uid", required=True)
    parser.add_argument("--app-name", default="")
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_name = args.app_name or Path(args.deploy_profile).stem
    profile = load_deploy_profile(args.deploy_profile)
    labels = {
        "app.kubernetes.io/managed-by": "argo-workflow",
        "app.kubernetes.io/name": app_name,
        "zilliz.com/workflow-run-id": args.workflow_uid,
    }
    annotations = {
        "zilliz.com/workflow-name": args.workflow_name,
        "zilliz.com/workflow-uid": args.workflow_uid,
        "zilliz.com/deploy-profile": args.deploy_profile,
    }
    cr = render_milvus_cr(
        profile=profile,
        name=args.name,
        namespace=args.namespace,
        image=args.image,
        version=args.version,
        image_pull_policy=args.image_pull_policy,
        image_update_mode=args.image_update_mode,
        json_shredding_enabled=args.json_shredding_enabled,
        loon_ffi_enabled=args.loon_ffi_enabled,
        vortex_enabled=args.vortex_enabled,
        labels=labels,
        annotations=annotations,
    )
    dump_yaml(cr, args.output_yaml)
    summary = deploy_topology_summary(profile, cr)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
