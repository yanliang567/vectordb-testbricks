from __future__ import annotations

from pathlib import Path
import json
import sys

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.client import create_client
from milvus_client.common.data import (
    checksum_fields_for_spec,
    generate_rows,
    stable_checksum,
)
from milvus_client.common.result import FAILED, result_from_args
from milvus_client.common.schema import (
    auto_id_enabled,
    collection_name,
    load_schema_matrix,
)


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--rows-per-collection", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--flush", type=parse_bool, default=True)


def _extract_insert_ids(response) -> list:
    if response is None:
        return []
    if isinstance(response, dict):
        for key in ("ids", "primary_keys", "primaryKeys"):
            if key in response:
                return list(response[key])
    for attr in ("ids", "primary_keys", "primaryKeys"):
        if hasattr(response, attr):
            return list(getattr(response, attr))
    return []


def _partition_for_id(partitions: list[str], pk: int) -> str | None:
    if not partitions:
        return None
    return partitions[pk % len(partitions)]


def _insert_rows(
    client, collection_name: str, rows: list[dict], partition_name: str | None = None
):
    if partition_name:
        return client.insert(
            collection_name=collection_name, data=rows, partition_name=partition_name
        )
    return client.insert(collection_name=collection_name, data=rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser(
        "Seed deterministic data into schema matrix collections"
    )
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "seed_data")
    try:
        client = create_client(args.uri, args.token, args.db_name)
        specs = load_schema_matrix(args.schema_matrix)
        checkpoint = {"collections": {}}
        inserted_total = 0

        for spec in specs:
            name = collection_name(args.collection_prefix, spec)
            primary_fields = [field for field in spec.fields if field.primary]
            primary_field = primary_fields[0].name if primary_fields else "id"
            checksum_fields = checksum_fields_for_spec(spec)
            collection_rows = []
            collection_inserted = 0
            collection_ids = []
            uses_auto_id = auto_id_enabled(spec)
            try:
                if not client.has_collection(name):
                    result.mark_failed(
                        "COLLECTION_NOT_FOUND",
                        "target collection does not exist",
                        collection=name,
                    )
                    continue
                for start in range(
                    args.start_id,
                    args.start_id + args.rows_per_collection,
                    args.batch_size,
                ):
                    count = min(
                        args.batch_size,
                        args.start_id + args.rows_per_collection - start,
                    )
                    rows = generate_rows(
                        spec, start_id=start, count=count, seed=args.seed
                    )
                    if rows:
                        if spec.partitions:
                            responses = []
                            partition_rows: dict[str, list[tuple[int, dict]]] = {}
                            for offset, row in enumerate(rows):
                                partition = _partition_for_id(
                                    spec.partitions, start + offset
                                )
                                partition_rows.setdefault(partition or "", []).append(
                                    (offset, row)
                                )
                            for partition, rows_with_offsets in partition_rows.items():
                                response = _insert_rows(
                                    client,
                                    name,
                                    [row for _, row in rows_with_offsets],
                                    partition_name=partition or None,
                                )
                                responses.append((rows_with_offsets, response))
                        else:
                            responses = [
                                (
                                    [(offset, row) for offset, row in enumerate(rows)],
                                    _insert_rows(client, name, rows),
                                )
                            ]
                        if uses_auto_id:
                            for rows_with_offsets, response in responses:
                                ids = _extract_insert_ids(response)
                                if len(ids) != len(rows_with_offsets):
                                    raise RuntimeError(
                                        f"auto_id insert returned {len(ids)} ids for {len(rows_with_offsets)} rows"
                                    )
                                for (_, row), inserted_id in zip(
                                    rows_with_offsets, ids
                                ):
                                    row[primary_field] = inserted_id
                                    collection_ids.append(inserted_id)
                        collection_rows.extend(rows)
                        collection_inserted += len(rows)
                        inserted_total += len(rows)
                if args.flush:
                    try:
                        client.flush(collection_name=name)
                    except TypeError:
                        client.flush(name)
            except Exception as exc:
                result.mark_failed(
                    "SEED_COLLECTION_FAILED",
                    "failed to seed collection",
                    collection=name,
                    schema=spec.name,
                    inserted=collection_inserted,
                    error=str(exc),
                )
                continue
            if uses_auto_id:
                min_pk = min(collection_ids) if collection_ids else None
                max_pk = max(collection_ids) if collection_ids else None
                pk_samples = list(
                    dict.fromkeys(
                        collection_ids[:1]
                        + collection_ids[
                            len(collection_ids) // 2 : len(collection_ids) // 2 + 1
                        ]
                        + collection_ids[-1:]
                    )
                )
                pk_values = collection_ids
            else:
                min_pk = args.start_id
                max_pk = args.start_id + args.rows_per_collection - 1
                pk_samples = []
                pk_values = []
            collection_checkpoint = {
                "schema_name": spec.name,
                "expected_count": args.rows_per_collection,
                "primary_field": primary_field,
                "min_pk": min_pk,
                "max_pk": max_pk,
                "data_min_pk": args.start_id,
                "data_max_pk": args.start_id + args.rows_per_collection - 1,
                "checksum_fields": checksum_fields,
                "checksum": stable_checksum(
                    collection_rows, fields=checksum_fields, primary_field=primary_field
                ),
            }
            if pk_samples:
                collection_checkpoint["pk_samples"] = pk_samples
            if pk_values:
                collection_checkpoint["pk_values"] = pk_values
            if spec.partitions:
                collection_checkpoint["partitions"] = spec.partitions
            checkpoint["collections"][name] = collection_checkpoint

        checkpoint_path = (
            Path(args.checkpoint_file)
            if args.checkpoint_file
            else Path(args.checkpoint_dir) / "seed_data.json"
        )
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
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "SEED_DATA_FAILED", "unexpected error during data seeding", error=str(exc)
        )
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
