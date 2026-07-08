from __future__ import annotations

import sys

from milvus_client.requests._pressure import run_pressure_brick


def main(argv: list[str] | None = None) -> int:
    return run_pressure_brick(argv, "count_pressure", ["count"])


if __name__ == "__main__":
    sys.exit(main())
