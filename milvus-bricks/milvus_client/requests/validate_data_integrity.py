from __future__ import annotations

from pathlib import Path
import json
import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import load_schema_matrix
from milvus_client.common.validators import ValidationReport, validate_collection_count, validate_pk_samples


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", default="")


def _primary_field_by_schema(schema_matrix: str) -> dict[str, str]:
    mapping = {}
    for spec in load_schema_matrix(schema_matrix):
        primary = [field for field in spec.fields if field.primary]
        if primary:
            mapping[spec.name] = primary[0].name
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Validate deterministic data integrity")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "validate_data_integrity")
    checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else Path(args.checkpoint_dir) / "seed_data.json"
    if not checkpoint_file.exists():
        result.status = FAILED
        result.mark_failed("CHECKPOINT_NOT_FOUND", "seed checkpoint file does not exist", path=str(checkpoint_file))
        result.write(args.output_json)
        return 2

    checkpoint = json.loads(checkpoint_file.read_text())
    primary_fields = _primary_field_by_schema(args.schema_matrix)
    client = create_client(args.uri, args.token, args.db_name)
    report = ValidationReport()
    for collection, meta in checkpoint.get("collections", {}).items():
        validate_collection_count(client, collection, int(meta["expected_count"]), report)
        primary_field = primary_fields.get(meta["schema_name"], "id")
        min_pk = int(meta["min_pk"])
        max_pk = int(meta["max_pk"])
        mid_pk = min_pk + (max_pk - min_pk) // 2
        validate_pk_samples(client, collection, primary_field, [min_pk, mid_pk, max_pk], report)

    result.status = PASSED if report.passed else FAILED
    result.failures = report.failures
    result.metrics = report.metrics
    result.write(args.output_json)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

