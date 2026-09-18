from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep
from typing import Any
import json
import sys

from milvus_client.common.args import build_common_parser
from milvus_client.common.client import create_client, get_server_version
from milvus_client.common.data import generate_primary_key_value
from milvus_client.common.result import FAILED, PASSED, result_from_args
from milvus_client.common.schema import auto_id_enabled, load_schema_matrix
from milvus_client.common.validators import ValidationReport
from milvus_client.requests.validate_index_compatibility import (
    _indexed_vector_indexes,
    _validate_index_searches,
)


COMPACTION_FAILED = "STORAGE_V3_COMPACTION_FAILED"
COMPACTION_TIMEOUT = "STORAGE_V3_COMPACTION_TIMEOUT"
STORAGE_VERSION_MISMATCH = "STORAGE_V3_STORAGE_VERSION_MISMATCH"
SERVING_SEGMENT_MISMATCH = "STORAGE_V3_SERVING_SEGMENT_MISMATCH"
DATA_QUERY_FAILED = "STORAGE_V3_DATA_QUERY_FAILED"
DATA_INTEGRITY_FAILED = "STORAGE_V3_DATA_INTEGRITY_FAILED"
COMPACTION_LINEAGE_FAILED = "STORAGE_V3_COMPACTION_LINEAGE_FAILED"

ACTIVE_SEGMENT_STATES = {"Growing", "Flushed", "Sealed"}
STABLE_SEGMENT_STATES = {"Flushed", "Sealed"}
TERMINAL_COMPACTION_STATES = {"Completed", "Cleaned", "Failed", "Timeout"}
DEFAULT_TIMEOUT_SEC = 900.0
DEFAULT_POLL_INTERVAL_SEC = 2.0


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--expected-storage-version", type=int, default=3)
    parser.add_argument("--expected-before-storage-version", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument(
        "--poll-interval-sec", type=float, default=DEFAULT_POLL_INTERVAL_SEC
    )


def _enum_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(getattr(value, "name", value))


def _segment_snapshot(segments: list[Any]) -> dict[int, dict[str, Any]]:
    snapshot = {}
    for segment in segments:
        segment_id = int(segment.segment_id)
        state = getattr(segment, "state_name", None)
        if state is None:
            state = getattr(segment, "state")
        snapshot[segment_id] = {
            "segment_id": segment_id,
            "state": _enum_name(state),
            "num_rows": int(segment.num_rows),
            "is_sorted": bool(segment.is_sorted),
            "storage_version": int(segment.storage_version),
        }
    return snapshot


def _active_segments(snapshot: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        segment_id: segment
        for segment_id, segment in snapshot.items()
        if segment["state"] in ACTIVE_SEGMENT_STATES and segment["num_rows"] > 0
    }


def _checkpoint_snapshot(client: Any, collection: str) -> dict[str, Any]:
    persistent = _segment_snapshot(client.list_persistent_segments(collection))
    loaded = _segment_snapshot(client.list_loaded_segments(collection))
    active = _active_segments(persistent)
    serving = {
        segment_id: segment
        for segment_id, segment in loaded.items()
        if segment["num_rows"] > 0
    }
    return {
        "all": persistent,
        "active": active,
        "serving": serving,
        "storage_versions": sorted(
            {segment["storage_version"] for segment in active.values()}
        ),
        "active_rows": sum(segment["num_rows"] for segment in active.values()),
    }


def _wait_for_compaction(
    client: Any,
    collection: str,
    job_id: int,
    timeout_sec: float,
    poll_interval_sec: float,
) -> tuple[str, list[dict[str, Any]]]:
    deadline = monotonic() + timeout_sec
    last_state = "unknown"
    while monotonic() < deadline:
        state = client.get_compaction_state(job_id, timeout=min(30.0, timeout_sec))
        last_state = _enum_name(state)
        if last_state in TERMINAL_COMPACTION_STATES:
            plans = client.get_compaction_plans(job_id, timeout=min(30.0, timeout_sec))
            plan_items = []
            for plan in getattr(plans, "plans", []) or []:
                plan_items.append(
                    {
                        "sources": [int(source) for source in getattr(plan, "sources", [])],
                        "target": int(getattr(plan, "target", -1)),
                    }
                )
            return last_state, plan_items
        sleep(min(poll_interval_sec, max(0.0, deadline - monotonic())))
    raise TimeoutError(
        f"collection={collection} compaction job={job_id} did not finish; "
        f"last_state={last_state}"
    )


def _wait_for_storage_checkpoint(
    client: Any,
    collection: str,
    expected_rows: int,
    expected_storage_version: int,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = monotonic() + timeout_sec
    last_checkpoint: dict[str, Any] = {}
    stable_polls = 0
    last_signature = None
    while monotonic() < deadline:
        checkpoint = _checkpoint_snapshot(client, collection)
        last_checkpoint = checkpoint
        active = checkpoint["active"]
        serving = checkpoint["serving"]
        active_ids = set(active)
        serving_ids = set(serving)
        ready = bool(active) and checkpoint["active_rows"] == expected_rows
        ready = ready and checkpoint["storage_versions"] == [expected_storage_version]
        ready = ready and active_ids == serving_ids
        ready = ready and all(
            segment["state"] in STABLE_SEGMENT_STATES and segment["is_sorted"]
            for segment in active.values()
        )
        ready = ready and all(
            segment["storage_version"] == expected_storage_version
            for segment in serving.values()
        )
        signature = json.dumps(checkpoint, sort_keys=True)
        if ready and signature == last_signature:
            stable_polls += 1
        elif ready:
            stable_polls = 1
        else:
            stable_polls = 0
        if stable_polls >= 2:
            return checkpoint
        last_signature = signature
        sleep(min(poll_interval_sec, max(0.0, deadline - monotonic())))
    raise TimeoutError(
        f"collection={collection} did not reach storage-v{expected_storage_version} "
        f"serving checkpoint: {last_checkpoint}"
    )


def _query_primary_keys(
    client: Any,
    collection: str,
    spec: Any,
    meta: dict[str, Any],
) -> dict[str, Any]:
    primary = next(field for field in spec.fields if field.primary)
    primary_name = meta.get("primary_field") or primary.name
    iterator = client.query_iterator(
        collection_name=collection,
        batch_size=1000,
        filter="",
        output_fields=[primary_name],
        consistency_level="Strong",
    )
    actual = set()
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            actual.update(row.get(primary_name) for row in rows)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    expected = set()
    if auto_id_enabled(spec):
        expected.update(meta.get("pk_values") or [])
    else:
        data_min = int(meta["data_min_pk"])
        data_max = int(meta["data_max_pk"])
        expected.update(
            generate_primary_key_value(primary, number)
            for number in range(data_min, data_max + 1)
        )
    missing = sorted(expected - actual, key=str)
    unexpected = sorted(actual - expected, key=str)
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_sample": missing[:20],
        "unexpected_sample": unexpected[:20],
    }


def _release_and_load(client: Any, collection: str, timeout_sec: float) -> None:
    client.release_collection(collection_name=collection, timeout=timeout_sec)
    client.load_collection(collection_name=collection, timeout=timeout_sec)


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser(
        "Compact existing storage-v2 segments into storage-v3 and verify serving reads"
    )
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "validate_storage_v3_compaction")
    report = ValidationReport()
    try:
        checkpoint_file = Path(args.checkpoint_file)
        if not checkpoint_file.exists():
            result.mark_failed(
                "CHECKPOINT_NOT_FOUND",
                "seed checkpoint file does not exist",
                path=str(checkpoint_file),
            )
            result.write(args.output_json)
            return 2
        seed_checkpoint = json.loads(checkpoint_file.read_text())
        specs = {spec.name: spec for spec in load_schema_matrix(args.schema_matrix)}
        client = create_client(args.uri, args.token, args.db_name)
        result.capabilities = {"server_version": get_server_version(client)}
        evidence = {"expected_storage_version": args.expected_storage_version, "collections": {}}
        metrics = {
            "collections_checked": 0,
            "compact_jobs": 0,
            "compaction_plans": 0,
            "storage_v3_persistent_collections": 0,
            "storage_v3_loaded_collections": 0,
            "query_collections": 0,
            "searches_total": 0,
        }

        for collection, meta in seed_checkpoint.get("collections", {}).items():
            spec = specs.get(meta.get("schema_name"))
            if spec is None:
                report.fail(
                    "SCHEMA_NOT_FOUND",
                    "checkpoint schema is absent from schema matrix",
                    collection=collection,
                    schema=meta.get("schema_name"),
                )
                continue
            metrics["collections_checked"] += 1
            try:
                before = _checkpoint_snapshot(client, collection)
                if before["storage_versions"] != [args.expected_before_storage_version]:
                    report.fail(
                        STORAGE_VERSION_MISMATCH,
                        "pre-compaction persistent segments are not all storage-v2",
                        collection=collection,
                        expected=[args.expected_before_storage_version],
                        actual=before["storage_versions"],
                    )
                    continue
                job_id = client.compact(collection_name=collection, timeout=args.timeout_sec)
                metrics["compact_jobs"] += 1
                state, plans = _wait_for_compaction(
                    client,
                    collection,
                    int(job_id),
                    args.timeout_sec,
                    args.poll_interval_sec,
                )
                if state not in {"Completed", "Cleaned"}:
                    report.fail(
                        COMPACTION_FAILED,
                        "compact job finished in a failed state",
                        collection=collection,
                        job_id=job_id,
                        state=state,
                    )
                    continue
                metrics["compaction_plans"] += len(plans)
                after_persisted = _checkpoint_snapshot(client, collection)
                before_ids = set(before["active"])
                after_ids = set(after_persisted["active"])
                plan_targets = {plan["target"] for plan in plans}
                plan_sources = {
                    source for plan in plans for source in plan["sources"]
                }
                if not plans or not plan_targets.intersection(after_ids) or not plan_sources.intersection(before_ids):
                    report.fail(
                        COMPACTION_LINEAGE_FAILED,
                        "compact job has no observable source-to-target segment transition",
                        collection=collection,
                        job_id=job_id,
                        plans=plans,
                        before_active=sorted(before_ids),
                        after_active=sorted(after_ids),
                    )
                _release_and_load(client, collection, args.timeout_sec)
                serving = _wait_for_storage_checkpoint(
                    client,
                    collection,
                    int(meta["expected_count"]),
                    args.expected_storage_version,
                    args.timeout_sec,
                    args.poll_interval_sec,
                )
                metrics["storage_v3_persistent_collections"] += 1
                metrics["storage_v3_loaded_collections"] += 1
                query_evidence = _query_primary_keys(client, collection, spec, meta)
                metrics["query_collections"] += 1
                if query_evidence["missing_sample"] or query_evidence["unexpected_sample"] or query_evidence["actual_count"] != query_evidence["expected_count"]:
                    report.fail(
                        DATA_INTEGRITY_FAILED,
                        "query iterator did not return the checkpoint primary-key set",
                        collection=collection,
                        **query_evidence,
                    )
                search_report = ValidationReport()
                search_count = _validate_index_searches(
                    client, collection, spec, meta, args.seed, search_report
                )
                metrics["searches_total"] += search_count
                report.failures.extend(search_report.failures)
                report.passed = report.passed and search_report.passed
                if not search_count and _indexed_vector_indexes(spec):
                    report.fail(
                        DATA_QUERY_FAILED,
                        "no indexed vector search was executed for collection",
                        collection=collection,
                    )
                evidence["collections"][collection] = {
                    "before": before,
                    "job_id": int(job_id),
                    "state": state,
                    "plans": plans,
                    "after_persisted": after_persisted,
                    "after_loaded": serving,
                    "query": query_evidence,
                    "searches": search_count,
                }
            except TimeoutError as exc:
                report.fail(
                    COMPACTION_TIMEOUT,
                    "storage-v3 compaction checkpoint timed out",
                    collection=collection,
                    error=str(exc),
                )
            except Exception as exc:
                report.fail(
                    COMPACTION_FAILED,
                    "storage-v3 compact/serving validation failed",
                    collection=collection,
                    error=str(exc),
                )

        checkpoint_path = Path(args.checkpoint_dir) / "storage_v3_compaction.json"
        checkpoint_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        result.checkpoint = {"path": str(checkpoint_path), "version": 1}
        result.metrics = metrics
        result.metrics["evidence_path"] = str(checkpoint_path)
        result.status = PASSED if report.passed else FAILED
        result.failures = report.failures
        result.write(args.output_json)
        return 0 if report.passed else 1
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            COMPACTION_FAILED,
            "unexpected storage-v3 compaction validation failure",
            error=str(exc),
        )
        result.write(args.output_json)
        return 4


if __name__ == "__main__":
    sys.exit(main())
