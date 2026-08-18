from __future__ import annotations

import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client, get_server_version
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.version import (
    is_daily_build_image,
    matching_pinned_image_build_tag,
    version_at_least,
    version_family,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Milvus connection precheck")
    parser.add_argument("--expected-server-version", default="")
    args = parser.parse_args(argv)
    result = result_from_args(args, "precheck")
    try:
        client = create_client(args.uri, args.token, args.db_name)
        collections = client.list_collections()
        server_version = get_server_version(client)
        result.capabilities = {
            "server_version": server_version,
            "effective_server_version": server_version,
            "sdk_version": "unknown",
            "supported": [],
            "unsupported": [],
        }
        try:
            actual_family = version_family(server_version)
        except ValueError as exc:
            matched_build_tag = matching_pinned_image_build_tag(
                args.expected_server_image, server_version
            )
            if (
                args.release_gate_eligible
                or not args.expected_server_version
                or matched_build_tag is None
            ):
                result.mark_failed(
                    "SERVER_VERSION_UNAVAILABLE",
                    "Milvus API did not return a parseable server version",
                    actual_version=server_version,
                    expected_image=args.expected_server_image,
                    release_gate_eligible=args.release_gate_eligible,
                    error=str(exc),
                )
                result.write(args.output_json)
                return 1
            expected_family = version_family(args.expected_server_version)
            result.capabilities["effective_server_version"] = (
                args.expected_server_version
            )
            result.metrics = {
                "collections_total": len(collections),
                "server_version_family": expected_family,
                "expected_server_version_family": expected_family,
                "server_version_validation_mode": "immutable_image_build_identity",
                "matched_server_build_tag": matched_build_tag,
            }
            result.status = PASSED
            result.write(args.output_json)
            return 0
        result.metrics = {
            "collections_total": len(collections),
            "server_version_family": actual_family,
        }
        if args.expected_server_version:
            expected_family = version_family(args.expected_server_version)
            result.metrics["expected_server_version_family"] = expected_family
            if actual_family != expected_family:
                result.mark_failed(
                    "SERVER_VERSION_MISMATCH",
                    "Milvus server version family differs from the expected phase version",
                    expected_version=args.expected_server_version,
                    expected_family=expected_family,
                    actual_version=server_version,
                    actual_family=actual_family,
                )
                result.write(args.output_json)
                return 1
            if not version_at_least(server_version, args.expected_server_version):
                if is_daily_build_image(args.expected_server_image):
                    result.metrics["server_version_validation_mode"] = (
                        "release_candidate_build"
                    )
                    result.metrics["candidate_server_version"] = server_version
                else:
                    result.mark_failed(
                        "SERVER_VERSION_TOO_OLD",
                        "Milvus server version is below the expected phase version",
                        expected_version=args.expected_server_version,
                        actual_version=server_version,
                    )
                    result.write(args.output_json)
                    return 1
        result.status = PASSED
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "ENV_UNAVAILABLE", "failed to connect to Milvus", error=str(exc)
        )
        result.write(args.output_json)
        return 3
    result.write(args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
