"""Common helpers for MilvusClient bricks.

This package re-exports the old `common.py` helpers through `common.legacy`
so existing scripts that run from this directory can keep using
`from common import ...` while new bricks use submodules such as
`common.args` and `common.result`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_legacy_module = None


def _load_legacy() -> Any:
    global _legacy_module
    if _legacy_module is None:
        try:
            _legacy_module = import_module(".legacy", __name__)
        except Exception as exc:
            raise ImportError(
                "legacy common helpers could not be imported; install the "
                "legacy script dependencies or import milvus_client.common "
                "submodules directly"
            ) from exc
    return _legacy_module


def __getattr__(name: str) -> Any:
    legacy = _load_legacy()
    try:
        return getattr(legacy, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
