from __future__ import annotations

import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.workload import run_operation, run_pressure_workload


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--operation-interval-sec", type=float, default=0.0)
    parser.add_argument("--baseline-start-id", type=int, default=0)
    parser.add_argument("--baseline-rows-per-collection", type=int, default=0)


def _run_operation(client, spec, collection, operation, seed, batch_size, op_index):
    return run_operation(client, spec, collection, operation, seed, batch_size, op_index)


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Run mixed read/write pressure against schema matrix collections")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "mixed_rw_pressure")

    try:
        client = create_client(args.uri, args.token, args.db_name)
        summary = run_pressure_workload(
            client,
            args.schema_matrix,
            args.collection_prefix,
            ["insert", "upsert", "query", "search", "count"],
            args.seed,
            args.duration_sec,
            args.max_workers,
            args.batch_size,
            operation_interval_sec=args.operation_interval_sec,
            baseline_start_id=args.baseline_start_id,
            baseline_rows_per_collection=args.baseline_rows_per_collection,
        )
    except Exception as exc:
        result.status = FAILED
        result.mark_failed("MIXED_RW_FAILED", "mixed read/write operation failed", error=str(exc))
        result.write(args.output_json)
        return 1

    result.status = FAILED if summary.failed else PASSED
    result.metrics = summary.metrics()
    result.write(args.output_json)
    return 1 if result.status == FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
