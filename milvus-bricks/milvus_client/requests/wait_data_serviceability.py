from __future__ import annotations

from pathlib import Path
import json
import sys
import time

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.data import generate_primary_key_value
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import FieldSpec, SchemaSpec, load_schema_matrix
from milvus_client.common.validators import (
    SERVICEABILITY_TIMEOUT,
    ValidationReport,
    is_transient_serviceability_failure,
    pk_range_filter,
    validate_collection_count,
    validate_pk_samples,
)


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--interval-sec", type=int, default=10)


def _spec_by_schema(schema_matrix: str) -> dict[str, SchemaSpec]:
    return {spec.name: spec for spec in load_schema_matrix(schema_matrix)}


def _primary_field(spec: SchemaSpec) -> FieldSpec | None:
    primary = [field for field in spec.fields if field.primary]
    if primary:
        return primary[0]
    return None


def _validate_serviceable(client, checkpoint: dict, specs: dict[str, SchemaSpec]) -> ValidationReport:
    report = ValidationReport()
    primary_fields = {name: _primary_field(spec) for name, spec in specs.items()}
    for collection, meta in checkpoint.get("collections", {}).items():
        schema_name = meta["schema_name"]
        primary_spec = primary_fields.get(schema_name)
        primary_field = meta.get("primary_field") or (primary_spec.name if primary_spec is not None else "id")
        pk_values = meta.get("pk_values")
        pk_value_fn = (
            (lambda pk, field=primary_spec: generate_primary_key_value(field, pk))
            if primary_spec is not None and not pk_values
            else (lambda pk: pk)
        )
        min_pk = int(meta["min_pk"])
        max_pk = int(meta["max_pk"])
        validate_collection_count(
            client,
            collection,
            int(meta["expected_count"]),
            report,
            filter_expr=pk_range_filter(primary_field, pk_value_fn(min_pk), pk_value_fn(max_pk)),
            metric_suffix="serviceable_count",
        )
        mid_pk = min_pk + (max_pk - min_pk) // 2
        sample_pks = meta.get("pk_samples") or [pk_value_fn(min_pk), pk_value_fn(mid_pk), pk_value_fn(max_pk)]
        validate_pk_samples(client, collection, primary_field, sample_pks, report)
    return report


def _all_failures_transient(report: ValidationReport) -> bool:
    return bool(report.failures) and all(is_transient_serviceability_failure(failure) for failure in report.failures)


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Wait until checkpoint data is query-serviceable")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "wait_data_serviceability")
    try:
        checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else Path(args.checkpoint_dir) / "seed_data.json"
        if not checkpoint_file.exists():
            result.status = FAILED
            result.mark_failed("CHECKPOINT_NOT_FOUND", "seed checkpoint file does not exist", path=str(checkpoint_file))
            result.write(args.output_json)
            return 2
        if args.timeout_sec < 0:
            result.status = FAILED
            result.mark_failed("INVALID_TIMEOUT", "timeout-sec must be non-negative", timeout_sec=args.timeout_sec)
            result.write(args.output_json)
            return 2
        if args.interval_sec < 0:
            result.status = FAILED
            result.mark_failed("INVALID_INTERVAL", "interval-sec must be non-negative", interval_sec=args.interval_sec)
            result.write(args.output_json)
            return 2

        checkpoint = json.loads(checkpoint_file.read_text())
        specs = _spec_by_schema(args.schema_matrix)
        client = create_client(args.uri, args.token, args.db_name)
        deadline = time.monotonic() + args.timeout_sec
        started = time.monotonic()
        attempts = 0
        transient_failures = 0
        last_report = ValidationReport()

        while True:
            attempts += 1
            report = _validate_serviceable(client, checkpoint, specs)
            last_report = report
            if report.passed:
                elapsed = round(time.monotonic() - started, 3)
                result.status = PASSED
                result.metrics = report.metrics
                result.metrics.update(
                    {
                        "attempts": attempts,
                        "recovered": transient_failures > 0,
                        "recovery_duration_sec": elapsed if transient_failures > 0 else 0,
                        "transient_failure_attempts": transient_failures,
                        "timeout_sec": args.timeout_sec,
                        "interval_sec": args.interval_sec,
                    }
                )
                result.write(args.output_json)
                return 0
            if not _all_failures_transient(report):
                result.status = FAILED
                result.failures = report.failures
                result.metrics = report.metrics
                result.metrics.update(
                    {
                        "attempts": attempts,
                        "recovered": False,
                        "recovery_duration_sec": 0,
                        "transient_failure_attempts": transient_failures,
                    }
                )
                result.write(args.output_json)
                return 1

            transient_failures += 1
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(args.interval_sec, max(0, deadline - now)))

        elapsed = round(time.monotonic() - started, 3)
        result.status = FAILED
        result.failures = last_report.failures
        result.metrics = last_report.metrics
        result.metrics.update(
            {
                "attempts": attempts,
                "recovered": False,
                "recovery_duration_sec": elapsed,
                "transient_failure_attempts": transient_failures,
                "timeout_sec": args.timeout_sec,
                "interval_sec": args.interval_sec,
            }
        )
        result.mark_failed(
            SERVICEABILITY_TIMEOUT,
            "checkpoint data did not become query-serviceable before timeout",
            timeout_sec=args.timeout_sec,
            elapsed_sec=elapsed,
        )
        result.write(args.output_json)
        return 1
    except Exception as exc:
        result.status = FAILED
        result.mark_failed("SERVICEABILITY_WAIT_FAILED", "unexpected error while waiting for data serviceability", error=str(exc))
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
