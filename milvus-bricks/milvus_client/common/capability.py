from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import yaml


@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    detect: dict[str, Any]
    unsupported_behavior: str = "skip"
    requires_cluster_admin: bool = False


def load_capability_catalog(path: str | Path) -> dict[str, CapabilitySpec]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    specs = {}
    for item in payload.get("capabilities", []):
        spec = CapabilitySpec(
            id=item["id"],
            detect=item.get("detect", {}),
            unsupported_behavior=item.get("unsupported_behavior", "skip"),
            requires_cluster_admin=bool(item.get("requires_cluster_admin", False)),
        )
        specs[spec.id] = spec
    return specs


def parse_version(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value or "")
    return tuple(int(p) for p in parts[:3]) if parts else (0,)


def version_at_least(current: str, minimum: str) -> bool:
    current_parts = parse_version(current)
    min_parts = parse_version(minimum)
    width = max(len(current_parts), len(min_parts))
    current_parts += (0,) * (width - len(current_parts))
    min_parts += (0,) * (width - len(min_parts))
    return current_parts >= min_parts


def evaluate_capabilities(
    required: list[str],
    catalog: dict[str, CapabilitySpec],
    server_version: str,
) -> dict[str, Any]:
    supported = []
    unsupported = []
    for capability_id in required:
        spec = catalog.get(capability_id)
        if spec is None:
            unsupported.append(capability_id)
            continue
        minimum = spec.detect.get("server_version_min")
        if minimum and not version_at_least(server_version, str(minimum)):
            unsupported.append(capability_id)
        else:
            supported.append(capability_id)
    return {
        "server_version": server_version,
        "supported": supported,
        "unsupported": unsupported,
    }

