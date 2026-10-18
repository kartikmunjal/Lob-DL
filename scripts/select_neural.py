from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from lobdl.models import count_parameters
from lobdl.neural import (
    build_model,
    build_optimizer_and_scheduler,
    make_loaders,
    make_train_validation_datasets,
)
from lobdl.search import (
    SELECTION_SEEDS,
    candidate_grid,
    load_progress,
    save_progress,
    select_candidate,
    summarize_candidate,
)
from lobdl.selection_lock import (
    assert_not_frozen,
    completed_selection,
    verify_presearch_timing,
)
from lobdl.training import fit_classifier, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only registered neural search")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=("M1", "M2"), required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--timing-report", type=Path, default=Path("reports/phase1_timing.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for registered neural search")

    verify_presearch_timing(args.timing_report)
    assert_not_frozen(args.output)
    existing = completed_selection(args.output, args.model, args.horizon)
    if existing is not None:
        print(json.dumps({**existing, "candidates": "already_complete"}, indent=2))
        return
    device = torch.device("cuda")
    train, validation = make_train_validation_datasets(args.data, args.horizon)
    candidates = []
    search_root = args.output / "selection_runs" / args.model / f"h{args.horizon}"
    progress_path = search_root / "progress.json"
    progress = load_progress(progress_path, args.model, args.horizon)
    for trial, parameters in enumerate(candidate_grid(args.model), start=1):
        scores, best_epochs, parameter_counts = [], [], []
        for seed in SELECTION_SEEDS:
            key = f"trial{trial:02d}_seed{seed}"
            if key in progress["completed_fits"]:
                completed = progress["completed_fits"][key]
                if completed["parameters"] != parameters:
                    raise ValueError(f"registered parameters changed for {key}")
                scores.append(completed["validation_macro_f1"])
                best_epochs.append(completed["best_epoch"])
                parameter_counts.append(completed["parameter_count"])
                continue
            seed_everything(seed)
            train_loader, validation_loader, _ = make_loaders(
                train,
                validation,
                None,
                seed,
                args.batch_size,
                args.workers,
            )
            model = build_model(args.model, parameters).to(device)
            parameter_counts.append(count_parameters(model))
            optimizer, scheduler, clip = build_optimizer_and_scheduler(
                args.model,
                model,
                parameters,
                args.max_epochs,
                len(train_loader),
            )
            checkpoint = search_root / f"trial{trial:02d}_seed{seed}.pt"
            fit = fit_classifier(
                model,
                train_loader,
                validation_loader,
                optimizer,
                device,
                checkpoint,
                args.max_epochs,
                args.patience,
                clip,
                scheduler,
            )
            scores.append(fit.best_macro_f1)
            best_epochs.append(fit.best_epoch)
            checkpoint.unlink(missing_ok=True)
            progress["completed_fits"][key] = {
                "parameters": parameters,
                "validation_macro_f1": fit.best_macro_f1,
                "best_epoch": fit.best_epoch,
                "parameter_count": parameter_counts[-1],
            }
            save_progress(progress_path, progress)
            # On Windows, persistent DataLoader workers otherwise survive long
            # enough to overlap the next fit and can terminate the subprocess
            # after one candidate. Release every fit before constructing another.
            del scheduler, optimizer, model, train_loader, validation_loader
            gc.collect()
            torch.cuda.empty_cache()
        candidates.append(
            summarize_candidate(
                trial,
                parameters,
                scores,
                max(parameter_counts),
                best_epochs,
            )
        )
        print(json.dumps(candidates[-1]), flush=True)
    selected = select_candidate(candidates)
    report = {
        "model": args.model,
        "horizon": args.horizon,
        "selection_N_trials": 12,
        "selection_seeds": list(SELECTION_SEEDS),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "test_data_loaded": False,
        "selected": selected,
        "candidates": candidates,
    }
    output = args.output / "selections" / args.model / f"h{args.horizon}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "candidates": "written_to_file"}, indent=2))


if __name__ == "__main__":
    main()
