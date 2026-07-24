from __future__ import annotations

import json
from datetime import datetime, timedelta
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
    if start is None or end is None:
        return None
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
            return window
    return None


def has_failed_metrics(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    return any(int(metrics.get(key, 0) or 0) > 0 for key in FAILED_METRIC_KEYS)


def is_connectivity_failure(failure: dict[str, Any]) -> bool:
    if failure.get("connectivity_transient") is True:
        return True
    if failure.get("connectivity_transient") is False:
        return False
    text = json.dumps(failure, sort_keys=True).lower()
    return any(pattern in text for pattern in CONNECTIVITY_PATTERNS)


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

    remaining_failures: list[dict[str, Any]] = []
    excluded_failures: list[dict[str, Any]] = []
    matched_window: dict[str, Any] | None = None

    for failure in failures:
        start, end = failure_interval(failure, result)
        window = overlap_window(start, end, maintenance_windows)
        if window is not None and is_connectivity_failure(failure):
            excluded_failures.append(failure)
            matched_window = window
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
        "label": matched_window.get("label"),
        "started_at": matched_window.get("started_at"),
        "finished_at": matched_window.get("finished_at"),
    }
    entry["status"] = "maintenance_window_excluded"
    return ("excluded", entry)
