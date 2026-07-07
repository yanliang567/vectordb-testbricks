from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


PASSED = "passed"
FAILED = "failed"
WARNING = "warning"
SKIPPED = "skipped"


@dataclass
class BrickResult:
    brick: str
    feature_set: str
    compat_mode: str
    lifecycle_phase: str
    status: str
    target: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    artifacts: list[str] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, failure_type: str, message: str, **details: Any) -> None:
        self.status = FAILED
        failure = {"type": failure_type, "message": message}
        failure.update(details)
        self.failures.append(failure)

    def write(self, path: str | Path) -> None:
        self.finish()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))


def result_from_args(args: Any, brick: str, status: str = PASSED) -> BrickResult:
    return BrickResult(
        brick=brick,
        feature_set=args.feature_set,
        compat_mode=args.compat_mode,
        lifecycle_phase=args.lifecycle_phase,
        status=status,
        target={
            "uri": args.uri,
            "db_name": args.db_name,
            "collection_prefix": args.collection_prefix,
        },
    )
