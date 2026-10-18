import json

from lobdl.search import candidate_grid, load_progress, save_progress, select_candidate
from lobdl.selection_lock import HORIZONS, MODELS, build_lock
from scripts.publish_frozen_selections import publish


def test_every_registered_grid_has_exactly_twelve_candidates():
    for model in ("B1", "B2", "M1", "M2"):
        assert len(candidate_grid(model)) == 12


def test_selection_uses_registered_tie_break_order():
    candidates = [
        {
            "trial": 1,
            "parameters": {"learning_rate": 0.001},
            "mean_validation_macro_f1": 0.5,
            "sd_validation_macro_f1": 0.02,
            "parameter_count": 10,
        },
        {
            "trial": 2,
            "parameters": {"learning_rate": 0.01},
            "mean_validation_macro_f1": 0.5,
            "sd_validation_macro_f1": 0.01,
            "parameter_count": 20,
        },
    ]
    assert select_candidate(candidates)["trial"] == 2


def test_selection_progress_round_trip(tmp_path):
    path = tmp_path / "progress.json"
    progress = load_progress(path, "M1", 10)
    progress["completed_fits"]["trial01_seed5"] = {"score": 0.5}
    save_progress(path, progress)
    assert load_progress(path, "M1", 10) == progress


def test_publish_frozen_selections_verifies_and_copies_lock(tmp_path):
    artifacts = tmp_path / "artifacts"
    for model in MODELS:
        candidates = [
            {
                "trial": index,
                "parameters": parameters,
                "mean_validation_macro_f1": 0.0,
                "sd_validation_macro_f1": 0.0,
                "parameter_count": 1,
            }
            for index, parameters in enumerate(candidate_grid(model), start=1)
        ]
        for horizon in HORIZONS:
            path = artifacts / "selections" / model / f"h{horizon}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "model": model,
                        "horizon": horizon,
                        "selection_N_trials": 12,
                        "test_data_loaded": False,
                        "selected": candidates[0],
                        "candidates": candidates,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    frozen = build_lock(artifacts)
    lock = artifacts / "selections" / "FROZEN.json"
    lock.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "published"
    report = publish(artifacts, output)
    assert report["status"] == "validation_selections_published_before_test"
    assert len(report["files"]) == 16
    assert (output / "FROZEN.json").read_bytes() == lock.read_bytes()
    assert publish(artifacts, output) == report
