from __future__ import annotations

import base64
import gzip
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CONNECTIVITY_PATTERNS = (
    "fail connecting to server",
    "failed to connect",
    "server unavailable",
    "connection refused",
    "connection reset",
    "connection aborted",
    "connection closed",
    "deadline exceeded",
    "temporarily unavailable",
    "transport is closing",
    "timed out",
    "timeout",
    "unavailable",
    "eof",
)

FAILED_METRIC_KEYS = (
    "requests_failed",
    "failed_search",
    "failed_query",
    "failed_insert",
    "failed_upsert",
    "failed_delete",
    "failed_count",
    "failed_query_iterator",
)

ROLLOUT_WINDOW_LABELS = {
    "upgrade-rollout",
    "post-upgrade-config-rollout",
    "rollback-rollout",
}

COLLECTION_RELOAD_WINDOW_KIND = "collection-reload"

COLLECTION_RELOAD_WINDOW_LABELS = {
    "validate_index_compatibility": {
        "index-compatibility-reload-after-upgrade",
        "index-compatibility-reload-after-rollback",
    },
    "validate_phase_dml_dql": {
        "phase-dml-dql-reload-after-upgrade",
        "phase-dml-dql-reload-after-rollback",
        "phase-checkpoint-reload-after-rollback",
    },
}

COLLECTION_RELOAD_UNAVAILABLE_OPERATIONS = {
    "search",
    "query",
    "query_iterator",
    "count",
    "delete",
}


@contextmanager
def record_maintenance_window(
    windows: list[dict[str, Any]],
    *,
    label: str,
    source: str,
    collection: str,
):
    started_at = datetime.now(timezone.utc)
    try:
        yield
    finally:
        finished_at = datetime.now(timezone.utc)
        windows.append(
            {
                "kind": COLLECTION_RELOAD_WINDOW_KIND,
                "label": label,
                "source": source,
                "collection": collection,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_sec": max(0.0, (finished_at - started_at).total_seconds()),
            }
        )


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _result_overlaps_window(result: dict[str, Any], window: dict[str, Any]) -> bool:
    start, end = result_interval(result)
    window_start = parse_time(window.get("started_at") or window.get("started_at_ts"))
    window_end = parse_time(window.get("finished_at") or window.get("finished_at_ts"))
    if start is None or end is None or window_start is None or window_end is None:
        return False
    return start <= window_end and end >= window_start


def _availability_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    operations_total = 0
    requests_failed = 0
    incomplete_sample_count = 0
    failed_sample_count = 0
    impacted_bricks: set[str] = set()
    failure_starts: list[datetime] = []
    failure_ends: list[datetime] = []

    for result in results:
        metrics = result.get("metrics") or {}
        if "operations_total" not in metrics:
            incomplete_sample_count += 1
        operations_total += _non_negative_int(metrics.get("operations_total"))
        result_failed = failed_metric_count(result)
        requests_failed += result_failed
        sample_failed = result_failed > 0 or result.get("status") not in {
            "passed",
            "skipped",
        }
        if not sample_failed:
            continue
        failed_sample_count += 1
        brick = str(result.get("brick") or "")
        if brick:
            impacted_bricks.add(brick)
        failures = result.get("failures") or []
        intervals = [failure_interval(failure, result) for failure in failures]
        if not intervals:
            intervals = [result_interval(result)]
        for start, end in intervals:
            if start is not None:
                failure_starts.append(start)
            if end is not None:
                failure_ends.append(end)

    operations_succeeded = max(0, operations_total - requests_failed)
    success_rate = (
        round(operations_succeeded / operations_total, 6)
        if operations_total > 0
        else None
    )
    first_failure = min(failure_starts) if failure_starts else None
    last_failure = max(failure_ends) if failure_ends else None
    failure_span_sec = (
        max(0.0, (last_failure - first_failure).total_seconds())
        if first_failure is not None and last_failure is not None
        else 0.0
    )
    complete = bool(results) and incomplete_sample_count == 0
    return {
        "sample_count": len(results),
        "incomplete_sample_count": incomplete_sample_count,
        "complete": complete,
        "calibration_eligible": complete and operations_total > 0,
        "operations_total": operations_total,
        "operations_succeeded": operations_succeeded,
        "requests_failed": requests_failed,
        "success_rate": success_rate,
        "failed_sample_count": failed_sample_count,
        "impacted_bricks": sorted(impacted_bricks),
        "first_failure_at": first_failure.isoformat() if first_failure else None,
        "last_failure_at": last_failure.isoformat() if last_failure else None,
        "failure_span_sec": failure_span_sec,
    }


def pressure_availability_samples(
    parsed_results: dict[str, dict[str, Any]],
    attempts: list[dict[str, Any]],
    unreadable_results: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    unreadable_results = unreadable_results or {}
    attempts_by_file = {
        str(attempt.get("result_file")): attempt
        for attempt in attempts
        if attempt.get("result_file")
    }
    samples = list(parsed_results.values())

    for result_file, error in sorted(unreadable_results.items()):
        if result_file in parsed_results:
            continue
        attempt = attempts_by_file.get(result_file, {})
        samples.append(
            {
                "file": result_file,
                "brick": attempt.get("module"),
                "status": "unreadable",
                "metrics": {},
                "failures": [
                    {
                        "type": "PRESSURE_RESULT_UNREADABLE",
                        "message": error,
                    }
                ],
            }
        )

    for attempt in attempts:
        result_file = str(attempt.get("result_file") or "")
        if not result_file:
            continue
        if result_file in parsed_results or result_file in unreadable_results:
            continue
        pending = attempt.get("return_code") == "pending"
        samples.append(
            {
                "file": result_file,
                "brick": attempt.get("module"),
                "status": "pending_result" if pending else "missing_result",
                "metrics": {},
                "failures": [
                    {
                        "type": (
                            "PRESSURE_ATTEMPT_PENDING"
                            if pending
                            else "PRESSURE_RESULT_MISSING"
                        ),
                        "message": (
                            "pressure attempt was recorded but did not complete"
                            if pending
                            else "pressure attempt did not produce a result json"
                        ),
                    }
                ],
            }
        )
    return samples


def build_pressure_availability_summary(
    results: list[dict[str, Any]], maintenance_windows: list[dict[str, Any]]
) -> dict[str, Any]:
    rollout_windows = [
        window
        for window in maintenance_windows
        if str(window.get("label") or "") in ROLLOUT_WINDOW_LABELS
    ]
    steady_state_excluded_windows = [
        window
        for window in maintenance_windows
        if str(window.get("label") or "") in ROLLOUT_WINDOW_LABELS
        or window.get("kind") == COLLECTION_RELOAD_WINDOW_KIND
    ]
    window_summaries = []
    for window in rollout_windows:
        overlapping_results = [
            result for result in results if _result_overlaps_window(result, window)
        ]
        window_summaries.append(
            {
                "label": window.get("label"),
                "started_at": window.get("started_at"),
                "finished_at": window.get("finished_at"),
                "duration_sec": window.get("duration_sec"),
                **_availability_stats(overlapping_results),
            }
        )
    steady_state_results = [
        result
        for result in results
        if all(value is not None for value in result_interval(result))
        and not any(
            _result_overlaps_window(result, window)
            for window in steady_state_excluded_windows
        )
    ]
    unassigned_sample_count = sum(
        1
        for result in results
        if any(value is None for value in result_interval(result))
    )
    return {
        "mode": "observational",
        "gate_enforced": False,
        "measurement": "overlapping_pressure_result_slices",
        "unassigned_sample_count": unassigned_sample_count,
        "overall": _availability_stats(results),
        "steady_state": _availability_stats(steady_state_results),
        "rollout_windows": window_summaries,
    }


def workflow_owned_configmaps(
    payload: dict[str, Any], *, workflow_name: str, workflow_uid: str
) -> list[dict[str, Any]]:
    prefix = f"{workflow_name}-pressure-"
    items = payload.get("items")
    if items is None and payload.get("kind") == "ConfigMap":
        items = [payload]
    matched = []
    for item in items or []:
        metadata = item.get("metadata") or {}
        labels = metadata.get("labels") or {}
        if not str(metadata.get("name") or "").startswith(prefix):
            continue
        if labels.get("zilliz.com/workflow-run-id") != workflow_uid:
            continue
        matched.append(item)
    return matched


def pressure_result_configmaps(
    payload: dict[str, Any], *, workflow_name: str, workflow_uid: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in workflow_owned_configmaps(
            payload, workflow_name=workflow_name, workflow_uid=workflow_uid
        )
        if ((item.get("metadata") or {}).get("labels") or {}).get(
            "zilliz.com/pressure-result"
        )
        == "true"
    ]


def pressure_result_text_from_configmap(item: dict[str, Any]) -> str | None:
    data = item.get("data") or {}
    if "result.json" in data:
        return data["result.json"]

    binary_data = item.get("binaryData") or {}
    encoded = binary_data.get("result.json.gz")
    if encoded is None:
        return None
    compressed = base64.b64decode(encoded, validate=True)
    return gzip.decompress(compressed).decode("utf-8")


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def result_interval(result: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    start = parse_time(result.get("started_at"))
    end = parse_time(result.get("finished_at"))
    if start is None and end is not None:
        start = end
    if end is None and start is not None:
        end = start
    return start, end


def failure_interval(
    failure: dict[str, Any], fallback: dict[str, Any]
) -> tuple[datetime | None, datetime | None]:
    start = parse_time(failure.get("started_at"))
    end = parse_time(failure.get("finished_at"))
    if start is None and end is None:
        return result_interval(fallback)
    if start is None:
        start = end
    if end is None:
        end = start
    return start, end


def overlap_window(
    start: datetime | None,
    end: datetime | None,
    maintenance_windows: list[dict[str, Any]],
    padding_sec: int = 5,
) -> dict[str, Any] | None:
    windows = overlapping_windows(start, end, maintenance_windows, padding_sec)
    return windows[0] if windows else None


def overlapping_windows(
    start: datetime | None,
    end: datetime | None,
    maintenance_windows: list[dict[str, Any]],
    padding_sec: int = 5,
) -> list[dict[str, Any]]:
    if start is None or end is None:
        return []
    windows = []
    for window in maintenance_windows:
        window_start = parse_time(
            window.get("started_at") or window.get("started_at_ts")
        )
        window_end = parse_time(
            window.get("finished_at") or window.get("finished_at_ts")
        )
        if window_start is None or window_end is None:
            continue
        padded_start = window_start - timedelta(seconds=padding_sec)
        padded_end = window_end + timedelta(seconds=padding_sec)
        if start <= padded_end and end >= padded_start:
            windows.append(window)
    return windows


def maintenance_windows_from_workflow_nodes(
    nodes: list[dict[str, Any]],
    *,
    post_upgrade_config_toggle_enabled: bool,
    schema_evolution_existing_enabled: bool,
    schema_evolution_forward_enabled: bool,
) -> list[dict[str, Any]]:
    def node_by_display(display_name: str) -> dict[str, Any] | None:
        return next(
            (node for node in nodes if node.get("displayName") == display_name), None
        )

    windows = []
    for label, start_name, end_name, enabled in (
        ("upgrade-rollout", "patch-upgrade", "wait-upgrade-ready", True),
        (
            "schema-evolution-existing",
            "schema-evolution-existing",
            "schema-evolution-existing",
            schema_evolution_existing_enabled,
        ),
        (
            "post-upgrade-config-rollout",
            "patch-post-upgrade-config",
            "wait-post-upgrade-config-ready",
            post_upgrade_config_toggle_enabled,
        ),
        (
            "schema-evolution-forward",
            "schema-evolution-forward",
            "schema-evolution-forward",
            schema_evolution_forward_enabled,
        ),
        ("rollback-rollout", "patch-rollback", "wait-rollback-ready", True),
    ):
        if not enabled:
            continue
        start_node = node_by_display(start_name)
        end_node = node_by_display(end_name)
        if (start_node or {}).get("phase") != "Succeeded" or (end_node or {}).get(
            "phase"
        ) != "Succeeded":
            continue
        start = parse_time((start_node or {}).get("startedAt"))
        end = parse_time((end_node or {}).get("finishedAt"))
        if start is None or end is None:
            continue
        windows.append(
            {
                "label": label,
                "started_at": start.isoformat(),
                "finished_at": end.isoformat(),
                "duration_sec": max(0.0, (end - start).total_seconds()),
                "started_at_ts": start,
                "finished_at_ts": end,
            }
        )
    return windows


def maintenance_windows_from_brick_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    windows = []
    for result in results:
        source = str(result.get("brick") or "")
        allowed_labels = COLLECTION_RELOAD_WINDOW_LABELS.get(source, set())
        metrics = result.get("metrics") or {}
        for window in metrics.get("maintenance_windows") or []:
            if not isinstance(window, dict):
                continue
            start = parse_time(window.get("started_at"))
            end = parse_time(window.get("finished_at"))
            label = str(window.get("label") or "")
            kind = str(window.get("kind") or "")
            if (
                start is None
                or end is None
                or end < start
                or kind != COLLECTION_RELOAD_WINDOW_KIND
                or label not in allowed_labels
            ):
                continue
            windows.append(
                {
                    "kind": kind,
                    "label": label,
                    "source": source,
                    "collection": str(window.get("collection") or ""),
                    "started_at": start.isoformat(),
                    "finished_at": end.isoformat(),
                    "duration_sec": max(0.0, (end - start).total_seconds()),
                    "started_at_ts": start,
                    "finished_at_ts": end,
                }
            )
    return sorted(
        windows,
        key=lambda window: (
            window["started_at_ts"],
            window["finished_at_ts"],
            window["label"],
            window["collection"],
        ),
    )


def has_failed_metrics(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    return any(int(metrics.get(key, 0) or 0) > 0 for key in FAILED_METRIC_KEYS)


def failed_metric_count(result: dict[str, Any]) -> int:
    metrics = result.get("metrics") or {}
    requests_failed = int(metrics.get("requests_failed", 0) or 0)
    operation_failures = sum(
        int(metrics.get(key, 0) or 0)
        for key in FAILED_METRIC_KEYS
        if key != "requests_failed"
    )
    return max(requests_failed, operation_failures)


def is_connectivity_failure(failure: dict[str, Any]) -> bool:
    if failure.get("connectivity_transient") is True:
        return True
    if failure.get("connectivity_transient") is False:
        return False
    text = json.dumps(failure, sort_keys=True).lower()
    return any(pattern in text for pattern in CONNECTIVITY_PATTERNS)


def is_schema_evolution_schema_mismatch(
    failure: dict[str, Any], window: dict[str, Any]
) -> bool:
    label = str(window.get("label") or "")
    if not label.startswith("schema-evolution-"):
        return False
    error_type = str(failure.get("error_type") or "")
    text = json.dumps(failure, sort_keys=True).lower()
    if error_type != "SchemaMismatchRetryableException" and (
        "schemamismatchretryableexception" not in text
    ):
        return False
    return "schema mismatch" in text


def is_rollout_service_switch_failure(
    failure: dict[str, Any], window: dict[str, Any]
) -> bool:
    label = str(window.get("label") or "")
    if label not in ROLLOUT_WINDOW_LABELS:
        return False

    error_type = str(failure.get("error_type") or "")
    text = json.dumps(failure, sort_keys=True).lower()
    if error_type != "MilvusException" and "milvusexception" not in text:
        return False

    if "channel not available" in text and (
        "channel distribution is not serviceable" in text
        or "no available shard leaders" in text
    ):
        return True
    if "internal count result should only have one column" in text:
        return True
    if (
        "reduce_by_groups" in text
        and "fielddatas length 0" in text
        and "expected 1" in text
    ):
        return True
    return "find no available mixcoord" in text or (
        "empty grpc client" in text and "mixcoord" in text
    )


def is_collection_reload_unavailable_failure(
    failure: dict[str, Any], window: dict[str, Any]
) -> bool:
    if window.get("kind") != COLLECTION_RELOAD_WINDOW_KIND:
        return False
    failure_collection = str(failure.get("collection") or "")
    window_collection = str(window.get("collection") or "")
    if not failure_collection or not window_collection:
        return False
    if failure_collection != window_collection:
        return False
    if str(failure.get("operation") or "") not in (
        COLLECTION_RELOAD_UNAVAILABLE_OPERATIONS
    ):
        return False
    error_type = str(failure.get("error_type") or "")
    text = json.dumps(failure, sort_keys=True).lower()
    if error_type != "MilvusException" and "milvusexception" not in text:
        return False
    return "collection not loaded" in text or "collection is not loaded" in text


def failure_entry(path: Path | str, result: dict[str, Any]) -> dict[str, Any]:
    file_name = path.name if isinstance(path, Path) else str(path)
    return {
        "file": file_name,
        "brick": result.get("brick"),
        "status": result.get("status"),
        "failures": result.get("failures", []),
        "metrics": result.get("metrics", {}),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
    }


def classify_pressure_result(
    path: Path | str,
    result: dict[str, Any],
    maintenance_windows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    if result.get("status") == "passed":
        return ("passed", None)

    failures = result.get("failures") or []
    entry = failure_entry(path, result)
    if not failures:
        if has_failed_metrics(result):
            entry["classification_reason"] = (
                "metrics_only_failure_without_error_details"
            )
        return ("failed", entry)

    metric_failure_count = failed_metric_count(result)
    if metric_failure_count > len(failures):
        entry["classification_reason"] = "failed_metrics_exceed_failure_details"
        entry["failure_detail_count"] = len(failures)
        entry["failed_metric_count"] = metric_failure_count
        return ("failed", entry)

    remaining_failures: list[dict[str, Any]] = []
    excluded_failures: list[dict[str, Any]] = []
    matched_window: dict[str, Any] | None = None

    for failure in failures:
        start, end = failure_interval(failure, result)
        window = overlap_window(start, end, maintenance_windows)
        failure_start = parse_time(failure.get("started_at"))
        failure_end = parse_time(failure.get("finished_at"))
        failure_windows = (
            overlapping_windows(failure_start, failure_end, maintenance_windows)
            if failure_start is not None and failure_end is not None
            else []
        )
        reload_failure_windows = (
            overlapping_windows(
                failure_start,
                failure_end,
                maintenance_windows,
                padding_sec=0,
            )
            if failure_start is not None and failure_end is not None
            else []
        )
        connectivity_window = (
            window
            if window is not None
            and window.get("kind") != COLLECTION_RELOAD_WINDOW_KIND
            and is_connectivity_failure(failure)
            else None
        )
        schema_mismatch_window = next(
            (
                candidate
                for candidate in failure_windows
                if is_schema_evolution_schema_mismatch(failure, candidate)
            ),
            None,
        )
        rollout_service_switch_window = next(
            (
                candidate
                for candidate in failure_windows
                if is_rollout_service_switch_failure(failure, candidate)
            ),
            None,
        )
        collection_reload_window = next(
            (
                candidate
                for candidate in reload_failure_windows
                if is_collection_reload_unavailable_failure(failure, candidate)
            ),
            None,
        )
        if (
            connectivity_window is not None
            or schema_mismatch_window is not None
            or rollout_service_switch_window is not None
            or collection_reload_window is not None
        ):
            matched_window = (
                collection_reload_window
                if collection_reload_window is not None
                else connectivity_window
                if connectivity_window is not None
                else schema_mismatch_window
                if schema_mismatch_window is not None
                else rollout_service_switch_window
            )
            excluded_failures.append(failure)
        else:
            remaining_failures.append(failure)

    if remaining_failures:
        entry["failures"] = remaining_failures
        if excluded_failures:
            entry["excluded_failures"] = excluded_failures
        return ("failed", entry)

    if matched_window is None:
        return ("failed", entry)

    entry["failures"] = excluded_failures
    entry["maintenance_window"] = {
        "kind": matched_window.get("kind"),
        "label": matched_window.get("label"),
        "source": matched_window.get("source"),
        "collection": matched_window.get("collection"),
        "started_at": matched_window.get("started_at"),
        "finished_at": matched_window.get("finished_at"),
    }
    entry["status"] = "maintenance_window_excluded"
    return ("excluded", entry)
