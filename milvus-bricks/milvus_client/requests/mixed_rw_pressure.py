from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from random import Random
import time
import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.data import first_vector_field, generate_rows, stable_float_vector
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import collection_name, load_schema_matrix


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--operation-interval-sec", type=float, default=0.0)


def _run_operation(client, spec, collection, operation, seed, batch_size, op_index):
    if operation == "insert":
        start_id = 10_000_000 + op_index * batch_size
        rows = generate_rows(spec, start_id=start_id, count=batch_size, seed=seed)
        client.insert(collection_name=collection, data=rows)
        return ("insert", len(rows))
    if operation == "upsert":
        start_id = 20_000_000 + op_index * batch_size
        rows = generate_rows(spec, start_id=start_id, count=batch_size, seed=seed)
        client.upsert(collection_name=collection, data=rows)
        return ("upsert", len(rows))
    if operation == "query":
        client.query(collection_name=collection, filter="id >= 0", output_fields=["id"], limit=10)
        return ("query", 1)
    if operation == "search":
        vector_field = first_vector_field(spec)
        if vector_field is None or vector_field.dtype != "FLOAT_VECTOR":
            return ("search_skipped", 0)
        query_vector = stable_float_vector(seed, op_index, vector_field.dim or 128)
        client.search(
            collection_name=collection,
            data=[query_vector],
            anns_field=vector_field.name,
            limit=5,
            search_params={"metric_type": "COSINE", "params": {}},
        )
        return ("search", 1)
    raise ValueError(f"unknown operation: {operation}")


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Run mixed read/write pressure against schema matrix collections")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "mixed_rw_pressure")
    client = create_client(args.uri, args.token, args.db_name)
    specs = load_schema_matrix(args.schema_matrix)
    rng = Random(args.seed)
    operations = ["insert", "upsert", "query", "search"]
    deadline = time.time() + args.duration_sec if args.duration_sec > 0 else time.time()
    op_index = 0
    counts: dict[str, int] = {}

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = set()

            def collect(done):
                for future in done:
                    op, count = future.result()
                    counts[op] = counts.get(op, 0) + count

            while time.time() <= deadline or (args.duration_sec == 0 and not futures):
                spec = rng.choice(specs)
                collection = collection_name(args.collection_prefix, spec)
                operation = rng.choice(operations)
                futures.add(
                    pool.submit(
                        _run_operation,
                        client,
                        spec,
                        collection,
                        operation,
                        args.seed,
                        args.batch_size,
                        op_index,
                    )
                )
                op_index += 1
                if args.operation_interval_sec > 0:
                    time.sleep(args.operation_interval_sec)
                if len(futures) >= args.max_workers:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    collect(done)
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                collect(done)
    except Exception as exc:
        result.status = FAILED
        result.mark_failed("MIXED_RW_FAILED", "mixed read/write operation failed", error=str(exc))
        result.metrics = counts
        result.write(args.output_json)
        return 1

    result.status = PASSED
    result.metrics = counts
    result.write(args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
