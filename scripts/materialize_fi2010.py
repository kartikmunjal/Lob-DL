from __future__ import annotations

import argparse
import json
from pathlib import Path

from lobdl.data.materialize import materialize_days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_days(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()

