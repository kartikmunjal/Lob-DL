from __future__ import annotations

import argparse
import json
from pathlib import Path

from lobdl.selection_lock import build_lock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze all 16 validation selections before test data may be loaded"
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    report = build_lock(args.artifacts)
    path = args.artifacts / "selections" / "FROZEN.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
