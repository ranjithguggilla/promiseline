"""Candidate models. Every preprocessing step lives inside the sklearn pipeline,
so fitting on one window never sees statistics from another."""

from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklego.model_selection import TimeGapSplit

from .config import SEED
from .features import CHECKOUT_CATEGORICAL, CHECKOUT_NUMERIC

# Simplest first: the one-standard-error rule walks this list in order.
MODEL_ORDER = ["promise_only", "logistic", "hist_gbm", "lightgbm"]


def _as_str(x):
    return x.astype(str)


def _onehot(min_frequency: int = 200) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                         sparse_output=False)


def make_model(name: str, numeric: list[str] | None = None,
               categorical: list[str] | None = None, **params) -> Pipeline:
    numeric = list(numeric or CHECKOUT_NUMERIC)
    categorical = list(categorical or CHECKOUT_CATEGORICAL)
    if name == "promise_only":
        # What the platform already "knows": its own promise plus how late things
        # have been running lately.
        cols = ["promise_days", "recent_late_rate"]
        pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"),
                                                       StandardScaler()), cols)])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    if name == "logistic":
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(FunctionTransformer(_as_str), _onehot()), categorical),
        ])
        return Pipeline([("pre", pre),
                         ("clf", LogisticRegression(C=params.get("C", 0.3), max_iter=4000))])
    ordinal = make_pipeline(
        FunctionTransformer(_as_str),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                       min_frequency=100, encoded_missing_value=-1),
    )
    pre = ColumnTransformer([("num", "passthrough", numeric), ("cat", ordinal, categorical)],
                            verbose_feature_names_out=False)
    cat_idx = list(range(len(numeric), len(numeric) + len(categorical)))
    if name == "hist_gbm":
        clf = HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.05), max_iter=params.get("max_iter", 400),
            max_leaf_nodes=params.get("max_leaf_nodes", 31), l2_regularization=1.0,
            categorical_features=cat_idx, early_stopping=False, random_state=SEED)
        return Pipeline([("pre", pre), ("clf", clf)])
    if name == "lightgbm":
        clf = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.03),
            num_leaves=params.get("num_leaves", 31), min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, verbose=-1)
        return Pipeline([("pre", pre), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}")


def feature_frame(df: pd.DataFrame, numeric: list[str] | None = None,
                  categorical: list[str] | None = None) -> pd.DataFrame:
    return df[list(numeric or CHECKOUT_NUMERIC) + list(categorical or CHECKOUT_CATEGORICAL)]


def tune_lightgbm(train: pd.DataFrame, grid: list[dict], n_splits: int = 3) -> tuple[dict, list]:
    """Pick LightGBM settings with time-ordered folds and a 30-day gap.

    The gap matters: labels need up to a month to mature, so a model trained
    through day t is in practice scored on orders from t + 30 onward.
    """
    dates = train["purchased_at"].reset_index(drop=True)
    X = feature_frame(train).reset_index(drop=True)
    y = train["late"].reset_index(drop=True)
    cv = TimeGapSplit(date_serie=dates, valid_duration=dt.timedelta(days=60),
                      gap_duration=dt.timedelta(days=30), n_splits=n_splits)
    results = []
    for params in grid:
        scores = []
        for tr, va in cv.split(X):
            m = make_model("lightgbm", **params).fit(X.iloc[tr], y.iloc[tr])
            scores.append(average_precision_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        results.append({"params": params, "ap_mean": float(np.mean(scores)),
                        "ap_folds": [float(s) for s in scores]})
    best = max(results, key=lambda r: r["ap_mean"])
    return best["params"], results
