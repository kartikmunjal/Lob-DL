import numpy as np

from scripts.analyze_factorial import (
    MODELS,
    analyze,
    contrast_definitions,
    prediction_path,
)

CLUSTERS = np.repeat([[8, 0], [8, 1], [9, 0], [9, 1]], 120, axis=0)
Y = np.random.default_rng(0).integers(0, 3, len(CLUSTERS))


def _probabilities(accuracy: float, seed: int):
    rng = np.random.default_rng(seed)
    pred = np.where(rng.random(len(Y)) < accuracy, Y, rng.integers(0, 3, len(Y)))
    return np.eye(3)[pred] * 0.9 + 0.033


def _loader(accuracy_by_cell, absent=()):
    def load(model, representation, horizon, seed):
        if (model, representation) in absent:
            raise FileNotFoundError
        accuracy = accuracy_by_cell[(model, representation)]
        return Y, _probabilities(accuracy, seed + horizon), CLUSTERS

    return load


def test_registered_family_sizes():
    defs = [d for h in (10, 20, 50, 100) for d in contrast_definitions(h)]
    assert sum(d["family"] == "primary" for d in defs) == 8
    assert sum(d["family"] == "secondary" for d in defs) == 72
    interaction = contrast_definitions(10)[0]["terms"]
    assert sum(interaction.values()) == 0  # a difference of differences has zero net weight


def test_interaction_detects_narrowed_gap_and_is_zero_without_one():
    base = {(m, r): 0.5 for m in MODELS for r in ("raw", "eng", "both")}
    base[("B2", "raw")] = base[("B2", "eng")] = base[("B2", "both")] = 0.9
    narrowed = {**base, ("M2", "eng"): 0.9}  # transformer catches up with engineered input
    result = analyze(_loader(narrowed), horizons=(10,), replicates=200)
    h1 = next(r for r in result["contrasts"] if r["contrast"].startswith("H1"))
    assert h1["mean_cluster_difference"] > 0.1
    assert h1["clusters_positive"] == 4
    flat = analyze(_loader(base), horizons=(10,), replicates=200)
    h1_flat = next(r for r in flat["contrasts"] if r["contrast"].startswith("H1"))
    assert abs(h1_flat["mean_cluster_difference"]) < 0.12  # sampling noise only


def test_missing_cells_are_withheld_and_holm_keeps_registered_size():
    accuracy = {(m, r): 0.6 for m in MODELS for r in ("raw", "eng", "both")}
    result = analyze(
        _loader(accuracy, absent={("M2", "eng"), ("M2", "both")}), horizons=(10,), replicates=100
    )
    withheld = [r for r in result["contrasts"] if "withheld" in r]
    assert any(r["contrast"].startswith("H1") for r in withheld)
    assert all("M2" in r["withheld"] for r in withheld)
    computed = [r for r in result["contrasts"] if "withheld" not in r]
    assert computed and all("holm_p" in r for r in computed)
    assert len(result["cells"]) == 7


def test_paths_route_study1_cells_and_new_cells(tmp_path):
    study1, partc = tmp_path / "s1", tmp_path / "pc"
    assert prediction_path(study1, partc, "B2", "eng", 10, 11) == (
        study1 / "B2" / "h10" / "seed11" / "predictions.npz"
    )
    assert prediction_path(study1, partc, "M1", "raw", 50, 23).parts[-4] == "M1"
    assert prediction_path(study1, partc, "M2", "both", 100, 47) == (
        partc / "F_M2_both" / "h100" / "seed47" / "predictions.npz"
    )
