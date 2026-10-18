from lobdl.search import candidate_grid, load_progress, save_progress, select_candidate


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
