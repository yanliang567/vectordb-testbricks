from __future__ import annotations

import sys

from milvus_client.requests._pressure import run_pressure_brick


def main(argv: list[str] | None = None) -> int:
    return run_pressure_brick(argv, "delete_pressure", ["delete"])


if __name__ == "__main__":
    sys.exit(main())
