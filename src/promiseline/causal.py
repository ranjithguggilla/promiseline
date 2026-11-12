"""What a missed promise does to the review, with checkout-time confounders removed.

Late orders differ from on-time ones before anything goes wrong: they travel
further, cost more, ship from busier sellers and land in different states.
The naive gap in review scores mixes all of that with the effect of the delay
itself. Double machine learning (partially linear model, cross-fitted
boosting for both nuisance functions) strips out what the checkout features
can explain and leaves the part attributable to lateness.
"""

from __future__ import annotations

import doubleml as dml
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import SEED
from .features import CHECKOUT_CATEGORICAL, CHECKOUT_NUMERIC


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric confounders plus one-hot states and categories; rows with a review only."""
    d = df.loc[df["review_score"].notna()].copy()
    x = d[CHECKOUT_NUMERIC].astype(float)
    x = x.fillna(x.median())
    dummies = pd.get_dummies(d[CHECKOUT_CATEGORICAL].astype(str), drop_first=True, dtype=float)
    keep = dummies.columns[dummies.sum() >= 200]
    out = pd.concat([x, dummies[keep]], axis=1)
    out.columns = [f"x{i}" for i in range(out.shape[1])]
    out["late"] = d["late"].astype(float).to_numpy()
    out["score"] = d["review_score"].astype(float).to_numpy()
    out["low"] = (d["review_score"] <= 2).astype(float).to_numpy()
    return out


def _learners(n_estimators: int):
    reg = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                            min_child_samples=50, random_state=SEED, verbose=-1)
    clf = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                             min_child_samples=50, random_state=SEED, verbose=-1)
    return reg, clf


def effect(frame: pd.DataFrame, outcome: str, n_folds: int = 5,
           n_estimators: int = 300) -> dict:
    """Naive difference and DML estimate of the late-delivery effect on ``outcome``."""
    xcols = [c for c in frame.columns if c.startswith("x")]
    data = dml.DoubleMLData(frame[xcols + [outcome, "late"]], y_col=outcome, d_cols="late",
                            x_cols=xcols)
    reg, clf = _learners(n_estimators)
    np.random.seed(SEED)
    model = dml.DoubleMLPLR(data, ml_l=reg, ml_m=clf, n_folds=n_folds)
    model.fit()
    ci = model.confint(level=0.95).iloc[0].tolist()
    late = frame["late"] == 1
    naive = float(frame.loc[late, outcome].mean() - frame.loc[~late, outcome].mean())
    return {"outcome": outcome, "naive": naive, "dml": float(model.coef[0]),
            "dml_se": float(model.se[0]), "dml_ci": [float(ci[0]), float(ci[1])],
            "n": int(len(frame)), "late_share": float(late.mean())}
