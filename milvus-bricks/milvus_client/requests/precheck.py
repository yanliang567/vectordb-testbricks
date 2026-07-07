from __future__ import annotations

import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client, get_server_version
from milvus_client.common.result import FAILED, PASSED, result_from_args


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Milvus connection precheck")
    args = parser.parse_args(argv)
    result = result_from_args(args, "precheck")
    try:
        client = create_client(args.uri, args.token, args.db_name)
        collections = client.list_collections()
        result.capabilities = {
            "server_version": get_server_version(client),
            "sdk_version": "unknown",
            "supported": [],
            "unsupported": [],
        }
        result.metrics = {"collections_total": len(collections)}
        result.status = PASSED
    except Exception as exc:
        result.status = FAILED
        result.mark_failed("ENV_UNAVAILABLE", "failed to connect to Milvus", error=str(exc))
        result.write(args.output_json)
        return 3
    result.write(args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
