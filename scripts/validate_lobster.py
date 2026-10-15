from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.data.lobster import validate_pure_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure LOBSTER replay from the first supplied snapshot")
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--time-bin-minutes", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/lobster.json"))
    parser.add_argument("--exact-mask-output", type=Path)
    args = parser.parse_args()
    result, mask = validate_pure_replay(args.messages, args.book, args.levels, args.time_bin_minutes)
    result["ticker"] = args.ticker
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.exact_mask_output:
        args.exact_mask_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.exact_mask_output, mask)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
