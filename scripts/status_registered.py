from __future__ import annotations

import argparse
import json
from pathlib import Path

from lobdl.search import SELECTION_SEEDS
from lobdl.selection_lock import HORIZONS, MODELS, selection_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Result-blind registered-run progress")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    selections = {}
    for model in MODELS:
        expected = 12 if model == "B1" else 12 * len(SELECTION_SEEDS)
        final_path = selection_path(args.artifacts, model, HORIZONS[0]).parent
        for horizon in HORIZONS:
            key = f"{model}/h{horizon}"
            complete = (final_path / f"h{horizon}.json").exists()
            progress_path = (
                args.artifacts
                / "selection_runs"
                / model
                / f"h{horizon}"
                / "progress.json"
            )
            completed_fits = 0
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                completed_fits = len(progress.get("completed_fits", {}))
            selections[key] = {
                "selection_complete": complete,
                "completed_fits": expected if complete else completed_fits,
                "expected_fits": expected,
            }
    report = {
        "selections": selections,
        "completed_selection_files": sum(
            item["selection_complete"] for item in selections.values()
        ),
        "expected_selection_files": len(selections),
        "frozen": (args.artifacts / "selections" / "FROZEN.json").exists(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
