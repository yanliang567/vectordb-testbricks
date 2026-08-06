from __future__ import annotations

import base64
import gzip
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

ROLLOUT_WINDOW_LABELS = {
    "upgrade-rollout",
    "post-upgrade-config-rollout",
    "rollback-rollout",
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
    return "find no available mixcoord" in text or (
        "empty grpc client" in text and "mixcoord" in text
    )


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
        connectivity_window = (
            window if window is not None and is_connectivity_failure(failure) else None
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
        if (
            connectivity_window is not None
            or schema_mismatch_window is not None
            or rollout_service_switch_window is not None
        ):
            matched_window = (
                connectivity_window
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
        "label": matched_window.get("label"),
        "started_at": matched_window.get("started_at"),
        "finished_at": matched_window.get("finished_at"),
    }
    entry["status"] = "maintenance_window_excluded"
    return ("excluded", entry)
