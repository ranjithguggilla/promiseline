"""Promise dates: how many days to quote at checkout.

Olist's own estimate is very padded: the median order arrives about two
weeks before the quoted date. Here a quantile model of delivery time is
conformalised on held-out orders (split conformal on the quantile residual)
and compared, at matched on-time rates, with the cheapest possible
alternative: Olist's own quote shifted by a constant number of days.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from .config import SEED
from .features import CHECKOUT_CATEGORICAL, CHECKOUT_NUMERIC


def _as_str(x):
    return x.astype(str)


def make_quantile_model(alpha: float = 0.9) -> Pipeline:
    ordinal = make_pipeline(
        FunctionTransformer(_as_str),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=100,
                       encoded_missing_value=-1),
    )
    pre = ColumnTransformer([("num", "passthrough", CHECKOUT_NUMERIC),
                             ("cat", ordinal, CHECKOUT_CATEGORICAL)])
    reg = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=400,
                            learning_rate=0.05, num_leaves=31, min_child_samples=50,
                            random_state=SEED, verbose=-1)
    return Pipeline([("pre", pre), ("reg", reg)])


def realised_days(df: pd.DataFrame) -> np.ndarray:
    """Calendar days from purchase day to arrival day; never-arrived orders get +inf."""
    d = (df["delivered_at"].dt.normalize() - df["purchased_at"].dt.normalize()).dt.days
    return d.astype(float).fillna(np.inf).to_numpy()


def on_time_rate(promise_days: np.ndarray, actual_days: np.ndarray) -> float:
    return float(np.mean(actual_days <= promise_days))


def conformal_offset(pred: np.ndarray, actual: np.ndarray, target: float) -> float:
    """Smallest offset so that ``pred + offset`` covers ``target`` of calibration orders."""
    scores = np.sort(actual - pred)
    n = len(scores)
    k = min(n - 1, int(np.ceil((n + 1) * target)) - 1)
    return float(scores[k])


def constant_shift(olist_days: np.ndarray, actual: np.ndarray, target: float) -> int:
    """Largest whole-day cut to Olist's quote that still meets ``target`` on calibration orders."""
    best = 0
    for cut in range(0, 40):
        if on_time_rate(olist_days - cut, actual) >= target:
            best = cut
        else:
            break
    return best


def quote(pred: np.ndarray, offset: float, floor: int = 2) -> np.ndarray:
    return np.maximum(np.ceil(pred + offset), floor)


def tradeoff_curve(olist_days: np.ndarray, model_pred: np.ndarray, actual: np.ndarray,
                   offsets: np.ndarray) -> pd.DataFrame:
    """On-time rate against mean quoted days for both quoting rules."""
    rows = []
    for off in offsets:
        q = quote(model_pred, off)
        rows.append({"rule": "quantile model", "mean_days": float(q.mean()),
                     "on_time": on_time_rate(q, actual)})
    for cut in range(-5, 25):
        q = np.maximum(olist_days - cut, 2)
        rows.append({"rule": "shifted Olist quote", "mean_days": float(q.mean()),
                     "on_time": on_time_rate(q, actual)})
    return pd.DataFrame(rows)
