from __future__ import annotations

import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import collection_name, load_schema_matrix


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Drop Milvus collections from a schema matrix")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "drop_schema_matrix")

    try:
        client = create_client(args.uri, args.token, args.db_name)
        specs = load_schema_matrix(args.schema_matrix)
        dropped = []
        missing = []
        failed = []
        for spec in specs:
            name = collection_name(args.collection_prefix, spec)
            try:
                if client.has_collection(name):
                    client.drop_collection(name)
                    dropped.append({"schema": spec.name, "collection": name})
                else:
                    missing.append({"schema": spec.name, "collection": name})
            except Exception as exc:
                failed.append({"schema": spec.name, "collection": name, "error": str(exc)})

        result.metrics = {
            "schemas_total": len(specs),
            "dropped_total": len(dropped),
            "missing_total": len(missing),
            "failed_total": len(failed),
            "dropped": dropped,
            "missing": missing,
        }
        if failed:
            result.status = FAILED
            for failure in failed:
                result.mark_failed(
                    "DROP_COLLECTION_FAILED",
                    "failed to drop schema collection",
                    **failure,
                )
        else:
            result.status = PASSED
        result.write(args.output_json)
        return 1 if failed else 0
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "ENV_UNAVAILABLE", "failed to drop collections", error=str(exc)
        )
        result.write(args.output_json)
        return 3


if __name__ == "__main__":
    sys.exit(main())
