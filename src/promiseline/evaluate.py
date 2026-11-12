"""Metrics, purchase-day bootstrap intervals and the one-standard-error choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import SEED


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return float(err / len(y))


def capture_at(y: np.ndarray, p: np.ndarray, share: float) -> float:
    """Share of all late orders found in the top ``share`` of orders by risk."""
    y = np.asarray(y)
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def precision_at(y: np.ndarray, p: np.ndarray, share: float) -> float:
    y = np.asarray(y)
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(y[top].mean())


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "capture_top10": capture_at(y, p, 0.10),
        "precision_top10": precision_at(y, p, 0.10),
        "base_rate": float(np.mean(y)),
        "n": int(len(y)),
    }


def _day_groups(days: pd.Series) -> list[np.ndarray]:
    codes = pd.factorize(pd.to_datetime(days).dt.normalize())[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds)


def bootstrap(y, preds: dict[str, np.ndarray], days: pd.Series, metric: str = "ap",
              n_boot: int = 300, seed: int = SEED) -> dict[str, np.ndarray]:
    """Resample whole purchase days; same draw for every model so differences pair up.

    Orders placed on the same day share carriers, weather and backlog, so
    treating them as independent would make the intervals look tighter than
    they are.
    """
    fn = {"ap": average_precision_score, "auc": roc_auc_score,
          "log_loss": lambda a, b: log_loss(a, np.clip(b, 1e-6, 1 - 1e-6), labels=[0, 1])}[metric]
    y = np.asarray(y)
    groups = _day_groups(days)
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() == y[idx].max():
            for k in preds:
                out[k][b] = np.nan
            continue
        for k, p in preds.items():
            out[k][b] = fn(y[idx], np.asarray(p)[idx])
    return out


def interval(samples: np.ndarray) -> list[float]:
    s = samples[~np.isnan(samples)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def one_se_choice(point: dict[str, float], boots: dict[str, np.ndarray], order: list[str],
                  higher_is_better: bool = True) -> tuple[str, dict]:
    """Simplest model whose score is within one bootstrap SE of the best one.

    The SE is that of the paired difference to the best model, which is the
    quantity the decision actually turns on.
    """
    sign = 1.0 if higher_is_better else -1.0
    best = max(point, key=lambda k: sign * point[k])
    detail = {}
    for name in order:
        if name not in point:
            continue
        diff = boots[best] - boots[name]
        se = float(np.nanstd(diff, ddof=1))
        gap = sign * (point[best] - point[name])
        detail[name] = {"score": point[name], "gap_to_best": gap, "se_of_gap": se}
    for name in order:
        if name in detail and detail[name]["gap_to_best"] <= detail[name]["se_of_gap"]:
            return name, {"best": best, "models": detail}
    return best, {"best": best, "models": detail}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({"predicted": p[chunk].mean(), "observed": y[chunk].mean(), "n": len(chunk)})
    return pd.DataFrame(rows)
