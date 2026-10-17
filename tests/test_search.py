from lobdl.search import candidate_grid, select_candidate


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
