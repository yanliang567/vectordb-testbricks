from __future__ import annotations

from pathlib import Path
import json
import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.data import checksum_fields_for_spec
from milvus_client.common.schema import SchemaSpec, load_schema_matrix
from milvus_client.common.validators import (
    ValidationReport,
    pk_range_filter,
    validate_collection_count,
    validate_pk_samples,
    validate_scalar_checksum,
)


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--checksum-batch-size", type=int, default=1000)


def _spec_by_schema(schema_matrix: str) -> dict[str, SchemaSpec]:
    return {spec.name: spec for spec in load_schema_matrix(schema_matrix)}


def _primary_field(spec: SchemaSpec) -> str:
    primary = [field for field in spec.fields if field.primary]
    if primary:
        return primary[0].name
    return "id"


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
    specs = _spec_by_schema(args.schema_matrix)
    primary_fields = {name: _primary_field(spec) for name, spec in specs.items()}
    client = create_client(args.uri, args.token, args.db_name)
    report = ValidationReport()
    for collection, meta in checkpoint.get("collections", {}).items():
        schema_name = meta["schema_name"]
        primary_field = meta.get("primary_field") or primary_fields.get(schema_name, "id")
        min_pk = int(meta["min_pk"])
        max_pk = int(meta["max_pk"])
        validate_collection_count(
            client,
            collection,
            int(meta["expected_count"]),
            report,
            filter_expr=pk_range_filter(primary_field, min_pk, max_pk),
            metric_suffix="checkpoint_count",
        )
        mid_pk = min_pk + (max_pk - min_pk) // 2
        validate_pk_samples(client, collection, primary_field, [min_pk, mid_pk, max_pk], report)
        checksum = meta.get("checksum")
        if checksum:
            spec = specs.get(schema_name)
            checksum_fields = meta.get("checksum_fields")
            if checksum_fields is None and spec is not None:
                checksum_fields = checksum_fields_for_spec(spec)
            validate_scalar_checksum(
                client,
                collection,
                primary_field,
                min_pk,
                max_pk,
                checksum,
                list(checksum_fields or [primary_field]),
                report,
                batch_size=args.checksum_batch_size,
            )

    result.status = PASSED if report.passed else FAILED
    result.failures = report.failures
    result.metrics = report.metrics
    result.write(args.output_json)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
