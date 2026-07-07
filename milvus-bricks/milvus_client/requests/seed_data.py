from __future__ import annotations

from pathlib import Path
import json
import sys

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.client import create_client
from milvus_client.common.data import generate_rows, stable_checksum
from milvus_client.common.result import FAILED, result_from_args
from milvus_client.common.schema import collection_name, load_schema_matrix


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--rows-per-collection", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--flush", type=parse_bool, default=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Seed deterministic data into schema matrix collections")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "seed_data")
    client = create_client(args.uri, args.token, args.db_name)
    specs = load_schema_matrix(args.schema_matrix)
    checkpoint = {"collections": {}}
    inserted_total = 0

    for spec in specs:
        name = collection_name(args.collection_prefix, spec)
        if not client.has_collection(name):
            result.mark_failed("COLLECTION_NOT_FOUND", "target collection does not exist", collection=name)
            continue
        collection_rows = []
        for start in range(args.start_id, args.start_id + args.rows_per_collection, args.batch_size):
            count = min(args.batch_size, args.start_id + args.rows_per_collection - start)
            rows = generate_rows(spec, start_id=start, count=count, seed=args.seed)
            if rows:
                client.insert(collection_name=name, data=rows)
                collection_rows.extend(rows)
                inserted_total += len(rows)
        if args.flush:
            try:
                client.flush(collection_name=name)
            except TypeError:
                client.flush(name)
        checkpoint["collections"][name] = {
            "schema_name": spec.name,
            "expected_count": args.rows_per_collection,
            "min_pk": args.start_id,
            "max_pk": args.start_id + args.rows_per_collection - 1,
            "checksum": stable_checksum(collection_rows),
        }

    checkpoint_path = Path(args.checkpoint_dir) / "seed_data.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True))
    result.checkpoint = {"path": str(checkpoint_path), "version": 1}
    result.metrics = {
        "collections_total": len(specs),
        "entities_inserted": inserted_total,
        "checkpoint_path": str(checkpoint_path),
    }
    if result.failures:
        result.status = FAILED
    result.write(args.output_json)
    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())

