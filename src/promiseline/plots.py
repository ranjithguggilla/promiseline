"""README and dossier charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SPLITS

INK = "#1f2933"
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#9aa5b1"
GREEN = "#2f855a"
RED = "#c53030"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
})


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def monthly_late_rate(df: pd.DataFrame, path: Path) -> Path:
    m = df.groupby(df["purchased_at"].dt.to_period("M")).agg(
        rate=("late", "mean"), n=("late", "size"))
    m = m.loc[m["n"] >= 200]
    x = m.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axvspan(SPLITS.train_start, SPLITS.valid_start, color=BLUE, alpha=0.07)
    ax.axvspan(SPLITS.valid_start, SPLITS.test_start, color=ORANGE, alpha=0.10)
    ax.axvspan(SPLITS.test_start, SPLITS.test_end, color=GREEN, alpha=0.08)
    ax.plot(x, m["rate"] * 100, marker="o", color=INK, lw=1.6)
    for label, at in (("train", SPLITS.train_start), ("validation", SPLITS.valid_start),
                      ("test", SPLITS.test_start)):
        ax.text(at + pd.Timedelta(days=6), ax.get_ylim()[1] * 0.92, label, fontsize=9, color=INK)
    ax.set_ylabel("orders late (%)")
    ax.set_title("Share of orders that missed the promised date, by purchase month", loc="left")
    return _save(fig, path)


def model_comparison(table: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    y = np.arange(len(table))
    for ax, split, color in ((axes[0], "valid", ORANGE), (axes[1], "test", GREEN)):
        mid = table[f"{split}_ap"]
        lo = mid - table[f"{split}_ap_lo"]
        hi = table[f"{split}_ap_hi"] - mid
        ax.barh(y, mid, color=color, alpha=0.75, xerr=[lo, hi], capsize=3)
        ax.axvline(table[f"{split}_base"].iloc[0], color=GREY, ls="--", lw=1)
        ax.set_title(f"{'validation' if split == 'valid' else 'test'} average precision",
                     loc="left")
    axes[0].set_yticks(y, table["label"])
    axes[1].set_xlabel("dashed line: late base rate", color=GREY, fontsize=8)
    return _save(fig, path)


def capture_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [GREY, BLUE, ORANGE, GREEN, RED]
    for (name, (x, yv)), c in zip(curves.items(), colors, strict=False):
        ax.plot(x * 100, yv * 100, label=name, color=c, lw=1.8)
    ax.plot([0, 100], [0, 100], color=GREY, ls=":", lw=1)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 85)
    ax.axvline(10, color=GREY, lw=0.8)
    ax.set_xlabel("orders flagged, highest risk first (%)")
    ax.set_ylabel("late orders caught (%)")
    ax.set_title("Test window: how many late orders a review list catches", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def calibration(before: pd.DataFrame, after: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    lim = max(before["predicted"].max(), after["predicted"].max(), before["observed"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, ls=":")
    ax.plot(before["predicted"], before["observed"], marker="o", color=ORANGE, label="as trained")
    ax.plot(after["predicted"], after["observed"], marker="o", color=BLUE,
            label="monthly intercept update")
    ax.set_xlabel("predicted late probability")
    ax.set_ylabel("observed late share")
    ax.set_title("Calibration on the test window", loc="left")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def ale_panel(curves: dict[str, pd.DataFrame], titles: dict[str, str], path: Path) -> Path:
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0))
    for ax, (f, d) in zip(np.atleast_1d(axes), curves.items(), strict=True):
        ax.plot(d.iloc[:, 0], d["eff"] * 100, color=BLUE, lw=1.8)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_title(titles.get(f, f), fontsize=9, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("effect on late risk (pts)")
    return _save(fig, path)


def promise_tradeoff(curve: pd.DataFrame, olist_point: tuple[float, float],
                     chosen: dict[str, tuple[float, float]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for rule, color in (("shifted Olist quote", ORANGE), ("quantile model", BLUE)):
        d = curve.loc[curve["rule"] == rule].sort_values("mean_days")
        ax.plot(d["mean_days"], d["on_time"] * 100, color=color, lw=1.8, label=rule)
    ax.scatter(*[olist_point[0]], [olist_point[1] * 100], color=INK, zorder=3,
               label="Olist quote as is")
    for name, (mx, ot) in chosen.items():
        ax.scatter([mx], [ot * 100], marker="D", s=40, color=RED, zorder=4)
        ax.annotate(name, (mx, ot * 100), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("mean quoted delivery days")
    ax.set_ylabel("orders delivered by the quote (%)")
    ax.set_ylim(60, 100)
    ax.set_title("Test window: on-time rate against quoted days", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def causal_bars(effects: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(effects), figsize=(4.2 * len(effects), 3.2))
    for ax, e in zip(np.atleast_1d(axes), effects, strict=True):
        scale = 100 if e["outcome"] == "low" else 1
        vals = [e["naive"] * scale, e["dml"] * scale]
        err = [[0, (e["dml"] - e["dml_ci"][0]) * scale], [0, (e["dml_ci"][1] - e["dml"]) * scale]]
        ax.bar(["naive gap", "double ML"], vals, color=[GREY, BLUE], yerr=err, capsize=4)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title("review score (stars)" if e["outcome"] == "score"
                     else "1-2 star reviews (pts)", loc="left", fontsize=10)
    return _save(fig, path)


def audit_bars(rows: list[tuple[str, float, float]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(rows))
    ax.barh(y - 0.18, [r[1] for r in rows], height=0.36, color=GREY, label="honest setup")
    ax.barh(y + 0.18, [r[2] for r in rows], height=0.36, color=RED, label="flawed setup")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("ROC AUC")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("What each shortcut would have claimed", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)


def sensitivity_heatmap(grid: pd.DataFrame, path: Path) -> Path:
    pv = grid.pivot(index="late_order_cost", columns="efficacy", values="net_per_1000")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    im = ax.imshow(pv.values, cmap="RdBu", vmin=-abs(pv.values).max(), vmax=abs(pv.values).max(),
                   origin="lower")
    ax.set_xticks(range(pv.shape[1]), [f"{c:.0%}" for c in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [f"${c:.0f}" for c in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            val = pv.values[i, j]
            txt = f"-${abs(val):,.0f}" if val < 0 else f"${val:,.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_xlabel("share of flagged late orders rescued")
    ax.set_ylabel("cost of one late order")
    ax.set_title("Net value per 1,000 orders, threshold policy", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, path)


def policy_bars(econ: dict[str, dict], path: Path) -> Path:
    names = list(econ)
    vals = [econ[k]["net_per_1000"] for k in names]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=[GREEN if v > 0 else RED for v in vals], alpha=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    for yi, v, k in zip(y, vals, names, strict=True):
        money = f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
        ax.text(v, yi, f"  {money}  ({econ[k]['flagged']:,} flagged)  ", va="center", fontsize=8,
                ha="left" if v >= 0 else "right")
    ax.set_xlabel("net value per 1,000 orders (USD, stated assumptions)")
    ax.set_title("Test window: which orders to touch", loc="left")
    lo, hi = min(vals + [0]), max(vals + [0])
    ax.set_xlim(lo - (hi - lo) * 0.6 - 5, hi + (hi - lo) * 0.6 + 5)
    return _save(fig, path)
