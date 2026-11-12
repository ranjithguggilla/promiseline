"""Keep probabilities honest when the late rate moves.

The ranking a model learned survives a regime change far better than its
probability scale does. Each month I shift the model's log-odds by one
constant, chosen so the mean prediction matches the late rate among recent
orders whose outcome is already known on the first day of that month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit


def intercept_shift(p: np.ndarray, target_rate: float) -> float:
    """Log-odds offset that moves the mean of ``p`` onto ``target_rate``."""
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    target = float(np.clip(target_rate, 1e-4, 1 - 1e-4))
    return float(brentq(lambda b: expit(z + b).mean() - target, -15, 15))


def apply_shift(p: np.ndarray, shift: float) -> np.ndarray:
    return expit(logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)) + shift)


def matured_window(history: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 45,
                   min_age_days: int = 15) -> pd.DataFrame:
    """Recent orders whose late/on-time outcome was already known at ``as_of``.

    Orders younger than ``min_age_days`` are skipped entirely: among them only
    the fast, on-time ones have resolved, which would bias the rate downward.
    """
    t = history["purchased_at"]
    recent = history.loc[(t >= as_of - pd.Timedelta(days=lookback_days))
                         & (t < as_of - pd.Timedelta(days=min_age_days))]
    return recent.loc[recent["label_known_at"] < as_of]


def monthly_recalibration(frame: pd.DataFrame, p: np.ndarray, history: pd.DataFrame,
                          history_p: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Shift each month's predictions using only labels known at that month's start."""
    out = np.asarray(p, float).copy()
    hp = pd.Series(np.asarray(history_p, float), index=history.index)
    month = frame["purchased_at"].dt.to_period("M")
    rows = []
    for m in sorted(month.unique()):
        start = m.start_time
        win = matured_window(history, start)
        mask = (month == m).to_numpy()
        if len(win) < 200:
            rows.append({"month": str(m), "shift": 0.0, "window_n": len(win)})
            continue
        shift = intercept_shift(hp.loc[win.index].to_numpy(), win["late"].mean())
        out[mask] = apply_shift(out[mask], shift)
        rows.append({"month": str(m), "shift": shift, "window_n": int(len(win)),
                     "window_rate": float(win["late"].mean()),
                     "predicted_rate": float(out[mask].mean()),
                     "actual_rate": float(frame.loc[mask, "late"].mean())})
    return out, pd.DataFrame(rows)
