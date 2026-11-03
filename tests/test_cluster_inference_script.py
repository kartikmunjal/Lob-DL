import json

import numpy as np
import pytest

from scripts.cluster_inference import analyse, flatten, published_family


def _write_report(root, horizon, a, b, variant=None):
    report = {
        "system_a": a,
        "system_b": b,
        "horizon": horizon,
        "metric_differences": {
            "macro_f1": {"difference": 0.01, "lower": 0.009, "upper": 0.011}
        },
    }
    if variant:
        report["label_variant"] = variant
    path = root / f"h{horizon}" / f"{a}_minus_{b}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_family_is_read_from_registered_reports(tmp_path):
    _write_report(tmp_path, 10, "B1", "B2")
    _write_report(tmp_path, 100, "B2", "M1")
    family = published_family(tmp_path)
    assert [(e["horizon"], e["system_a"], e["system_b"]) for e in family] == [
        (10, "B1", "B2"),
        (100, "B2", "M1"),
    ]
    with pytest.raises(FileNotFoundError):
        published_family(tmp_path / "missing")


def test_analyse_applies_holm_over_whole_family(tmp_path):
    rng = np.random.default_rng(0)
    rows = 1500
    y = rng.integers(0, 3, rows)
    good = np.eye(3)[y] * 0.9 + 0.1 / 3
    noise = rng.dirichlet(np.ones(3), rows)
    groups = np.column_stack(
        (np.repeat(np.arange(15) // 5 + 8, 100), np.repeat(np.arange(15) % 5, 100), np.ones(rows))
    )
    family = [
        {"system_a": "A", "system_b": "B", "horizon": 10, "label_variant": None,
         "registered_report": "x", "registered_macro_f1": None},
        {"system_a": "A", "system_b": "A", "horizon": 10, "label_variant": None,
         "registered_report": "y", "registered_macro_f1": None},
    ]
    pairs = {("A", "B"): (good, noise), ("A", "A"): (good, good)}

    def loader(_, entry):
        a, b = pairs[(entry["system_a"], entry["system_b"])]
        return a, b, y, groups

    result = analyse(family, loader, tmp_path, replicates=500)
    strong = result[0]["cluster_inference"]["macro_f1"]
    assert strong["sign_flip"]["p_value"] == pytest.approx(2 / 2**15)
    assert strong["holm_p_value"] == pytest.approx(2 * 2 / 2**15)
    assert result[1]["cluster_inference"]["macro_f1"]["holm_p_value"] == pytest.approx(1.0)
    table = flatten(result, "FI2010")
    assert len(table) == 4 and {row["metric"] for row in table} == {"macro_f1", "mcc"}
    assert all(row["status"].startswith("post_hoc") for row in table)
