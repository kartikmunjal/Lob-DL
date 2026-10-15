from __future__ import annotations

import argparse
import json
from pathlib import Path

from lobdl.data.lobster import compare_snapshots, reconstruct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/lobster.json"))
    args = parser.parse_args()
    result = compare_snapshots(reconstruct(args.messages, args.levels), args.book)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["exact"]:
        raise SystemExit("LOBSTER reconstruction did not match published snapshots")


if __name__ == "__main__":
    main()

