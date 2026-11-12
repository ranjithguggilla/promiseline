"""Three ways this project could have fooled itself, each measured directly."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED
from .features import CHECKOUT_CATEGORICAL, CHECKOUT_NUMERIC
from .models import feature_frame, make_model


def _score(model_name, train, test, numeric=None) -> dict:
    m = make_model(model_name, numeric=numeric).fit(feature_frame(train, numeric), train["late"])
    p = m.predict_proba(feature_frame(test, numeric))[:, 1]
    return {"auc": float(roc_auc_score(test["late"], p)),
            "ap": float(average_precision_score(test["late"], p))}


def outcome_leak(train: pd.DataFrame, test: pd.DataFrame, model_name: str = "logistic") -> dict:
    """Add columns that only exist after the parcel arrives and watch the score explode.

    ``review_lag`` is the gap between the promise and the day Olist sent the
    review request. The request goes out on delivery, so the column is the
    answer in disguise. It sits in the same export as everything else.
    """
    def leaky(d):
        d = d.copy()
        d["review_lag"] = (d["review_creation_date"] - d["promised_date"]).dt.days
        d["review_score_f"] = d["review_score"]
        return d
    numeric = CHECKOUT_NUMERIC + ["review_lag", "review_score_f"]
    return {"honest": _score(model_name, train, test),
            "with_outcome_columns": _score(model_name, leaky(train), leaky(test), numeric)}


def whole_period_history(train: pd.DataFrame, test: pd.DataFrame, full: pd.DataFrame,
                         model_name: str = "logistic") -> dict:
    """Seller late rates computed over the whole export instead of as-of each order."""
    rate = full.groupby("seller_id")["late"].mean()
    def swap(d):
        d = d.copy()
        d["seller_late_rate"] = d["seller_id"].map(rate).fillna(full["late"].mean())
        return d
    return {"as_of": _score(model_name, train, test),
            "whole_period": _score(model_name, swap(train), swap(test))}


def random_vs_forward(train: pd.DataFrame, test: pd.DataFrame, model_name: str = "logistic",
                      folds: int = 5) -> dict:
    """Shuffled k-fold on the development window against the true forward test."""
    X = feature_frame(train)
    y = train["late"].to_numpy()
    aucs, aps = [], []
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        m = make_model(model_name).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
        aps.append(average_precision_score(y[va], p))
    fwd = _score(model_name, train, test)
    return {"random_kfold": {"auc": float(np.mean(aucs)), "ap": float(np.mean(aps)),
                             "base_rate": float(y.mean())},
            "forward": {**fwd, "base_rate": float(test["late"].mean())}}


def feature_inventory() -> pd.DataFrame:
    """Which columns are allowed where; written to the outputs for review."""
    rows = [{"column": c, "known_at": "checkout"} for c in CHECKOUT_NUMERIC + CHECKOUT_CATEGORICAL]
    rows += [{"column": c, "known_at": "carrier handoff"} for c in
             ["handling_days", "approval_hours", "missed_ship_limit", "days_left_at_handoff"]]
    rows += [{"column": c, "known_at": "after outcome (never a feature)"} for c in
             ["delivered_at", "delivery_days", "review_score", "review_creation_date"]]
    return pd.DataFrame(rows)
