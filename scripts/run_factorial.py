"""Orchestrate Part C. Every step is idempotent, so reruns resume safely.

Stages: ``select`` (B2 validation searches, CPU), ``final`` (needs the frozen manifest).
Phase A (horizons 100, 10) runs before Phase B (50, 20). Failures stop the run.

    python scripts/run_factorial.py --stage select --data D --output O --study1 S
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from lobdl.factorial import NEW_B2_REPRESENTATIONS, NEW_NEURAL_CELLS, PHASE_A, PHASE_B
from lobdl.search import FINAL_SEEDS

SCRIPTS = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print(time.strftime("%H:%M:%S"), " ".join(command), flush=True)
    subprocess.run([sys.executable, *command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("select", "final"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study1", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--only", choices=("gbm", "neural"), default=None)
    args = parser.parse_args()
    common = ["--data", str(args.data), "--output", str(args.output), "--cache", str(args.cache)]
    horizons = (*PHASE_A, *PHASE_B)
    if args.stage == "select":
        for horizon in horizons:
            for representation in NEW_B2_REPRESENTATIONS:
                run([
                    str(SCRIPTS / "factorial_gbm.py"), *common, "--representation", representation,
                    "--phase", "select", "--horizon", str(horizon),
                ])
        return
    for horizon in horizons:
        if args.only in (None, "gbm"):
            for representation in NEW_B2_REPRESENTATIONS:
                for seed in FINAL_SEEDS:
                    run([
                        str(SCRIPTS / "factorial_gbm.py"), *common, "--representation",
                        representation, "--phase", "final", "--horizon", str(horizon),
                        "--seed", str(seed),
                    ])
        if args.only in (None, "neural"):
            for model, representation in NEW_NEURAL_CELLS:
                for seed in FINAL_SEEDS:
                    run([
                        str(SCRIPTS / "factorial_neural.py"), *common, "--study1", str(args.study1),
                        "--model", model, "--representation", representation,
                        "--horizon", str(horizon), "--seed", str(seed),
                    ])


if __name__ == "__main__":
    main()
