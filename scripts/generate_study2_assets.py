"""Build Study 2 figures and a markdown summary from the A1/A2 outputs.

Inputs are the JSON/CSV files written by ``cluster_inference.py`` and
``return_eval.py``. Everything produced is labeled post-hoc exploratory and is kept
apart from the registered Study 1 assets in ``reports/generated``.

    python scripts/generate_study2_assets.py --input reports/study2 --output reports/study2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HORIZONS = (10, 20, 50, 100)
PALETTE = {"cluster": "#1f5fa8", "block": "#c8741b", "zero": "#7a7a7a"}
KEY_PAIRS = (
    ("B2", "M1"),
    ("B2", "M2"),
    ("B0:persistence", "B2"),
    ("M1", "M2"),
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def select(rows: list[dict], **conditions: str) -> list[dict]:
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in conditions.items())]


def significance_summary(rows: list[dict], metric: str = "macro_f1") -> list[dict]:
    """Per dataset/label family: how many registered-significant comparisons survive."""
    families: dict[tuple[str, str], list[dict]] = {}
    for row in select(rows, metric=metric):
        families.setdefault((row["dataset"], row["label_variant"]), []).append(row)
    output = []
    for (dataset, label), members in sorted(families.items()):
        block_sig = [
            r for r in members
            if r["registered_block_ci95_lower"] != ""
            and (_float(r["registered_block_ci95_lower"]) > 0 or _float(r["registered_block_ci95_upper"]) < 0)
        ]
        survive = [r for r in block_sig if _float(r["holm_p"]) < 0.05]
        output.append(
            {
                "dataset": dataset,
                "label_variant": label or "primary",
                "comparisons": len(members),
                "clusters": int(float(members[0]["clusters"])),
                "minimum_attainable_p": _float(members[0]["minimum_attainable_p"]),
                "block_ci_excludes_zero": len(block_sig),
                "also_holm_significant": len(survive),
            }
        )
    return output


def plot_forest(rows: list[dict], path: Path) -> None:
    """Cluster CI versus registered block CI for the headline FI-2010 contrasts."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(13, 3.6), sharey=True)
    labels = [f"{a} − {b}" for a, b in KEY_PAIRS]
    for axis, horizon in zip(axes, HORIZONS, strict=True):
        for index, (a, b) in enumerate(KEY_PAIRS):
            match = select(
                rows, dataset="FI2010", horizon=horizon, system_a=a, system_b=b, metric="macro_f1"
            )
            if not match:
                continue
            row = match[0]
            y = len(KEY_PAIRS) - 1 - index
            axis.plot(
                [_float(row["cluster_ci95_lower"]), _float(row["cluster_ci95_upper"])],
                [y + 0.12] * 2, color=PALETTE["cluster"], lw=3, solid_capstyle="butt",
            )
            axis.plot(
                [_float(row["registered_block_ci95_lower"]), _float(row["registered_block_ci95_upper"])],
                [y - 0.12] * 2, color=PALETTE["block"], lw=3, solid_capstyle="butt",
            )
            axis.plot(_float(row["mean_cluster_difference"]), y + 0.12, "o", color=PALETTE["cluster"], ms=5)
        axis.axvline(0, color=PALETTE["zero"], lw=1, ls="--")
        axis.set_title(f"horizon {horizon}", fontsize=10)
        axis.set_yticks(range(len(KEY_PAIRS)))
        axis.set_yticklabels(list(reversed(labels)), fontsize=9)
        axis.set_xlabel("macro-F1 difference")
        axis.grid(axis="x", alpha=0.25)
    handles = [
        axes[0].plot([], [], color=PALETTE["cluster"], lw=3, label="15 stock-day clusters (this study)")[0],
        axes[0].plot([], [], color=PALETTE["block"], lw=3, label="registered window bootstrap")[0],
    ]
    fig.suptitle("Interval width depends on the unit of resampling (post-hoc exploratory)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_markout(rows: list[dict], path: Path) -> None:
    """Top-decile realized move versus half-spread, per system, FI-2010 and LOBSTER."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    systems = ["B0 persistence", "B1 logistic", "B2 LightGBM", "M1 DeepLOB", "M2 transformer"]
    colors = ["#7a7a7a", "#8c6bb1", "#1f5fa8", "#c8741b", "#2a8f5c"]
    for axis, dataset in zip(axes, ("FI2010", "LOBSTER"), strict=True):
        for system, color in zip(systems, colors, strict=True):
            points = [
                (h, _float(select(rows, dataset=dataset, system=system, horizon=h)[0]["top_decile_move_bps"]))
                for h in HORIZONS
                if select(rows, dataset=dataset, system=system, horizon=h)
            ]
            if points:
                axis.plot(*zip(*points, strict=True), marker="o", color=color, label=system, lw=1.6)
        spreads = [
            _float(select(rows, dataset=dataset, system="B2 LightGBM", horizon=h)[0]["top_decile_half_spread_bps"])
            for h in HORIZONS
            if select(rows, dataset=dataset, system="B2 LightGBM", horizon=h)
        ]
        if spreads:
            axis.plot(HORIZONS[: len(spreads)], spreads, color="black", ls="--", lw=1.4, label="half-spread (cost of crossing)")
        axis.set_xscale("log")
        axis.set_xticks(HORIZONS)
        axis.set_xticklabels([str(h) for h in HORIZONS])
        axis.set_xlabel("horizon (underlying events)")
        axis.set_title(dataset, fontsize=10)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("top signal-decile mean forward move (bps)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Is the predicted move larger than the cost of crossing? (post-hoc exploratory)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown(significance: list[dict], return_rows: list[dict]) -> str:
    lines = [
        "# Study 2 results (post-hoc exploratory)",
        "",
        "Generated by `scripts/generate_study2_assets.py`. These re-analyze frozen Study 1",
        "predictions and carry no confirmatory status.",
        "",
        "## A1: do registered comparisons survive stock-day inference?",
        "",
        "| Dataset | Labels | Comparisons | Clusters | Min. attainable p | Block CI excludes 0 | Also Holm-significant |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in significance:
        lines.append(
            f"| {row['dataset']} | {row['label_variant']} | {row['comparisons']} | {row['clusters']} "
            f"| {row['minimum_attainable_p']:.2g} | {row['block_ci_excludes_zero']} | {row['also_holm_significant']} |"
        )
    lines += [
        "",
        "![Cluster versus block intervals](figures/cluster_vs_block_ci.png)",
        "",
        "## A2: return-unit evaluation",
        "",
        "**Read FI-2010 rows with care.** The shipped FI-2010 labels agree with the",
        "public decimal-precision price formula on only 64% (h10), 54% (h20), 37% (h50)",
        "and 25% (h100) of rows (`reports/generated/tables/fi_label_formula_agreement.csv`).",
        "At h10, 62% of rows labeled *up* have exactly zero realized public-price move, and",
        "last-move persistence scores 0.80 macro-F1 on labels while its signal has IC −0.09",
        "against realized public-price moves. The labels are therefore only weakly tied to",
        "moves observable in the public book, so FI-2010 IC, R² and mark-outs measure",
        "agreement with *public-price* moves, not with the labels the models were trained on.",
        "LOBSTER labels and returns both come from the same vendor book and are internally",
        "consistent, so LOBSTER rows are the cleaner return-unit evidence. LOBSTER has five",
        "clusters (tickers); its cluster intervals are indicative only.",
        "",
        "| Dataset | System | Horizon | Mean cluster IC | OOS R² | Top-decile move (bps) | Half-spread (bps) | Taker edge (bps) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in return_rows:
        if row["system"] in {"B2 LightGBM", "M1 DeepLOB"}:
            lines.append(
                f"| {row['dataset']} | {row['system']} | {row['horizon']} | {_float(row['mean_cluster_ic']):.3f} "
                f"| {_float(row['oos_r2_bps']):.4f} | {_float(row['top_decile_move_bps']):.2f} "
                f"| {_float(row['top_decile_half_spread_bps']):.2f} | {_float(row['top_decile_taker_edge_bps']):.2f} |"
            )
    lines += ["", "![Top-decile move versus half-spread](figures/markout_vs_half_spread.png)", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("reports/study2"))
    parser.add_argument("--output", type=Path, default=Path("reports/study2"))
    args = parser.parse_args()
    cluster_rows = read_rows(args.input / "cluster_inference.csv")
    return_rows = read_rows(args.input / "return_eval.csv")
    (args.output / "figures").mkdir(parents=True, exist_ok=True)
    plot_forest(cluster_rows, args.output / "figures" / "cluster_vs_block_ci.png")
    plot_markout(return_rows, args.output / "figures" / "markout_vs_half_spread.png")
    significance = significance_summary(cluster_rows)
    (args.output / "RESULTS.md").write_text(markdown(significance, return_rows), encoding="utf-8")
    print(markdown(significance, return_rows))


if __name__ == "__main__":
    main()
