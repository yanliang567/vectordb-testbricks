from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client
from milvus_client.common.feature_validators import (
    EXTERNAL_VALIDATORS,
    run_feature_validator,
    unknown_validators,
)
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import SchemaSpec, collection_name, load_schema_matrix
from milvus_client.common.validators import ValidationReport


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--runtime-milvus-config", default="")
    parser.add_argument("--runtime-user-config", default="")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    payload = yaml.safe_load(source.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"runtime config must be a mapping: {source}")
    return payload


def _runtime_config(args) -> dict[str, Any]:
    return _deep_merge(
        _load_yaml(args.runtime_milvus_config),
        _load_yaml(args.runtime_user_config),
    )


def _checkpoint_path(args) -> Path:
    if args.checkpoint_file:
        return Path(args.checkpoint_file)
    return Path(args.checkpoint_dir) / "seed_data.json"


def _specs_by_name(schema_matrix: str) -> dict[str, SchemaSpec]:
    return {spec.name: spec for spec in load_schema_matrix(schema_matrix)}


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Validate schema feature semantics")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "validate_schema_features")
    try:
        checkpoint_file = _checkpoint_path(args)
        if not checkpoint_file.exists():
            result.status = FAILED
            result.mark_failed(
                "CHECKPOINT_NOT_FOUND",
                "seed checkpoint file does not exist",
                path=str(checkpoint_file),
            )
            result.write(args.output_json)
            return 2
        checkpoint = json.loads(checkpoint_file.read_text())
        specs = _specs_by_name(args.schema_matrix)
        for spec in specs.values():
            unknown = unknown_validators(spec)
            if unknown:
                result.status = FAILED
                result.mark_failed(
                    "UNKNOWN_SCHEMA_VALIDATOR",
                    "schema matrix declares unsupported validators",
                    schema=spec.name,
                    validators=unknown,
                )
        if result.failures:
            result.write(args.output_json)
            return 2

        actual_collections = set(checkpoint.get("collections", {}))
        expected_collections = {
            collection_name(args.collection_prefix, spec) for spec in specs.values()
        }
        missing_collections = sorted(expected_collections - actual_collections)
        if missing_collections:
            result.status = FAILED
            result.mark_failed(
                "SCHEMA_COLLECTION_MISSING",
                "checkpoint omits collections required by the schema matrix",
                collections=missing_collections,
            )
            result.write(args.output_json)
            return 2

        client = create_client(args.uri, args.token, args.db_name)
        runtime_config = _runtime_config(args)
        report = ValidationReport()
        metrics = {
            "collections_checked": 0,
            "validators_declared": 0,
            "feature_validators_executed": 0,
            "external_validators_skipped": 0,
        }
        for collection, meta in checkpoint.get("collections", {}).items():
            schema_name = str(meta.get("schema_name") or "")
            spec = specs.get(schema_name)
            if spec is None:
                report.fail(
                    "SCHEMA_NOT_FOUND",
                    "checkpoint schema is absent from schema matrix",
                    collection=collection,
                    schema=schema_name,
                )
                continue
            metrics["collections_checked"] += 1
            for validator in spec.validators:
                metrics["validators_declared"] += 1
                if validator in EXTERNAL_VALIDATORS:
                    metrics["external_validators_skipped"] += 1
                    continue
                try:
                    run_feature_validator(
                        validator,
                        client,
                        collection,
                        spec,
                        meta,
                        args.seed,
                        report,
                        runtime_config=runtime_config,
                    )
                    metrics["feature_validators_executed"] += 1
                except Exception as exc:
                    report.fail(
                        "FEATURE_VALIDATION_FAILED",
                        "schema feature validator raised an unexpected error",
                        collection=collection,
                        schema=spec.name,
                        validator=validator,
                        error=str(exc),
                    )
        result.status = PASSED if report.passed else FAILED
        result.failures = report.failures
        result.metrics = {**report.metrics, **metrics}
        result.write(args.output_json)
        return 0 if report.passed else 1
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "FEATURE_VALIDATION_FAILED",
            "unexpected error during schema feature validation",
            error=str(exc),
        )
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
