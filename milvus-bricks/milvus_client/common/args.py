from __future__ import annotations

import argparse


COMPAT_MODES = ("rollback_safe", "upgrade_only", "forward_only")
LIFECYCLE_PHASES = (
    "before_upgrade",
    "after_upgrade",
    "before_rollback",
    "after_rollback",
    "steady_state",
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--db-name", default="default")
    parser.add_argument("--collection-prefix", required=True)
    parser.add_argument("--duration-sec", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-set", default="compat_2_6")
    parser.add_argument("--compat-mode", choices=COMPAT_MODES, default="rollback_safe")
    parser.add_argument("--capability-probe", type=parse_bool, default=True)
    parser.add_argument("--skip-unsupported", type=parse_bool, default=True)
    parser.add_argument("--lifecycle-phase", choices=LIFECYCLE_PHASES, default="steady_state")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser
