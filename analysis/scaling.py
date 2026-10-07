"""Scaling-law ablation: paired error bars and scaling plot.

Usage:
    uv run analysis/scaling.py --method conv

Writes to `analysis/`:
    - `results_table.csv` / `results_table.md`: per-run statistics.
    - `scaling.png`: baseline vs method scaling curves, with paired,
      ESS-corrected 95% confidence intervals on the method.
    - `scaling_ci_methods.png`: the same plot with alternative (incorrect)
      error bars, for the analysis questions.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from nrdk import tss

SPLITS = {"p10": 0.1, "p20": 0.2, "p50": 0.5, "p100": 1.0}
Z95 = 1.959964

# Reference categorical palette (slots 1, 2) and text/grid tokens.
COLORS = {"baseline": "#2a78d6", "method": "#eb6834"}
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def pooled_mean(index: dict, experiment: str, key: str = "loss") -> float:
    """Pooled mean over every sample of every trace (not mean of means)."""
    losses = [
        np.load(path)[key].reshape(-1)
        for path in index[experiment].values()]
    return float(np.concatenate(losses).mean())


def load(results: str, baseline: str, method: str) -> pd.DataFrame:
    """Compute paired statistics for every run against its split baseline."""
    # Follow symlinks, so that runs from multiple results directories can be
    # combined into one directory of links.
    index = tss.index(
        results, r"^(?P<experiment>.*)/eval/(?P<trace>.*)/metrics\.npz$",
        follow_symlinks=True)
    experiments = [
        f"{name}/{split}" for name in (baseline, method) for split in SPLITS
        if f"{name}/{split}" in index]
    missing = sorted(
        {f"{n}/{s}" for n in (baseline, method) for s in SPLITS}
        - set(experiments))
    if missing:
        print(f"Warning: missing evaluations for {missing}")

    split = tss.Control.from_index_rule(
        "split", experiments, lambda e: f"{baseline}/{e.rsplit('/', 1)[1]}")
    df = tss.dataframe_from_index(
        index, key="loss", experiments=experiments,
        baseline=f"{baseline}/p100", controls=[split])

    df = df.reset_index() if "name" not in df.columns else df
    df["method"] = df["name"].str.rsplit("/", n=1).str[0]
    df["split"] = df["name"].str.rsplit("/", n=1).str[1]
    df["size"] = df["split"].map(SPLITS)
    df["pooled_mean"] = [pooled_mean(index, e) for e in df["name"]]
    df["ci95_split"] = Z95 * df["rel_split/stderr"]
    # `pct_split/*` is normalized by the global baseline; normalize by the
    # same-split baseline instead.
    base_mean = df.set_index("name")["pooled_mean"]
    split_base = df["split"].map(lambda s: base_mean[f"{baseline}/{s}"])
    df["pct_vs_split"] = 100 * df["rel_split/mean"] / split_base
    df["pct_ci95_vs_split"] = 100 * df["ci95_split"] / split_base
    # Naive (iid) paired standard error, ignoring temporal correlation.
    df["naive_stderr_split"] = (
        df["rel_split/std"] / np.sqrt(df["rel_split/n"]))
    return df.sort_values(["method", "size"])


def write_table(df: pd.DataFrame, out: str) -> None:
    """Write the full statistics table and a compact markdown summary."""
    df.to_csv(os.path.join(out, "results_table.csv"), index=False)

    rows = [
        "| Method | Split | Test loss (pooled mean) | Δ vs baseline "
        "(paired) | 95% CI | Δ % | ESS / n | Significant (p<0.05) |",
        "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        is_base = r["rel_split/mean"] == 0 and r["rel_split/std"] == 0
        delta = "—" if is_base else f"{r['rel_split/mean']:+.5f}"
        ci = "—" if is_base else f"±{r['ci95_split']:.5f}"
        pct = "—" if is_base else (
            f"{r['pct_vs_split']:+.2f}% ± {r['pct_ci95_vs_split']:.2f}%")
        ess = (
            "—" if is_base
            else f"{r['rel_split/ess']:.0f} / {r['rel_split/n']:.0f}")
        sig = "—" if is_base else str(bool(r["p0.05_split"]))
        rows.append(
            f"| {r['method']} | {r['split']} | {r['pooled_mean']:.5f} | "
            f"{delta} | {ci} | {pct} | {ess} | {sig} |")
    with open(os.path.join(out, "results_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def _style(ax: Axes, title: str) -> None:
    """Log-log axes with recessive grid and text-token labels."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor(SURFACE)
    ax.set_xticks(list(SPLITS.values()))
    ax.set_xticklabels([f"{v:g}" for v in SPLITS.values()])
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlabel("Relative training set size", color=TEXT_SECONDARY)
    ax.set_ylabel("Test loss", color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")

    # Narrow log ranges have no decade ticks; place ~5 labeled ticks.
    lo, hi = ax.get_ylim()
    ticks = np.geomspace(lo, hi, 7)[1:-1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.3g}" for t in ticks])
    ax.set_ylim(lo, hi)


def _curves(
    ax: Axes, base: pd.DataFrame, meth: pd.DataFrame, labels: tuple,
    yerr_method: np.ndarray | None = None,
    yerr_base: np.ndarray | None = None, direct_labels: bool = True,
) -> None:
    """Draw baseline and method curves with optional error bars."""
    for df, color, marker, label, yerr in [
        (base, COLORS["baseline"], "o", labels[0], yerr_base),
        (meth, COLORS["method"], "s", labels[1], yerr_method),
    ]:
        if len(df) == 0:
            continue
        ax.errorbar(
            df["size"], df["pooled_mean"], yerr=yerr, color=color,
            marker=marker, markersize=7, linewidth=2, capsize=4,
            elinewidth=1.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=label, zorder=3)
        if direct_labels:
            last = df.iloc[-1]
            ax.annotate(
                label, (last["size"], last["pooled_mean"]),
                xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=9, color=TEXT)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)


def _col(df: pd.DataFrame, key: str) -> np.ndarray:
    """Get a column as a float array."""
    return df[key].to_numpy(dtype=float)


def plot(df: pd.DataFrame, baseline: str, method: str, out: str) -> None:
    """Main scaling plot plus a comparison of error-bar methods."""
    base = df[df["method"] == baseline]
    meth = df[df["method"] == method]
    labels = ("Baseline (linear patch)", f"{method.capitalize()} tokenizer")

    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    # Legend only: the curves converge at p100, so direct labels collide.
    _curves(
        ax, base, meth, labels, yerr_method=_col(meth, "ci95_split"),
        direct_labels=False)
    _style(ax, "Test loss vs training set size (95% paired CI)")
    ax.set_xlim(0.08, 1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scaling.png"), dpi=200)
    plt.close(fig)

    variants = [
        ("Paired, ESS-corrected (correct)",
         Z95 * _col(meth, "rel_split/stderr"), None),
        ("Paired, naive std/√n",
         Z95 * _col(meth, "naive_stderr_split"), None),
        ("Unpaired, ESS-corrected",
         Z95 * _col(meth, "abs/stderr"), Z95 * _col(base, "abs/stderr")),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.4), facecolor=SURFACE, sharey=True)
    # Draw every panel before styling, so that the shared y limits include
    # the widest error bars.
    for ax, (_, ym, yb) in zip(axes, variants):
        _curves(
            ax, base, meth, labels, yerr_method=ym, yerr_base=yb,
            direct_labels=False)
    for ax, (title, _, _) in zip(axes, variants):
        _style(ax, title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scaling_ci_methods.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--method", default="conv")
    parser.add_argument(
        "--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    df = load(args.results, args.baseline, args.method)
    write_table(df, args.out)
    plot(df, args.baseline, args.method, args.out)


if __name__ == "__main__":
    main()
