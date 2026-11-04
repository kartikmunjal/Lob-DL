"""Freeze Part C B2 selections: write the hash manifest and publish copies for commit.

    python scripts/freeze_factorial.py --output OUT --publish reports/factorial_selections

Commit the published files BEFORE running any final fit (see docs/STUDY2_PROTOCOL.md Part C).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lobdl.factorial import build_lock, lock_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    args = parser.parse_args()
    path = lock_file(args.output)
    if path.exists():
        raise SystemExit(f"already frozen: {path}")
    lock = build_lock(args.output)
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    args.publish.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, args.publish / "FROZEN.json")
    for entry in lock["selections"].values():
        source = args.output / entry["path"]
        target = args.publish / Path(entry["path"]).relative_to("factorial_selections")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(json.dumps({"frozen": len(lock["selections"]), "manifest": str(path)}, indent=2))


if __name__ == "__main__":
    main()
