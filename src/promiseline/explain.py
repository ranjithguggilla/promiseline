"""Global shape of the risk (ALE curves) and per-order reasons in plain words."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from PyALE import ale


class ProbaModel:
    """PyALE calls ``predict``; route it to the late-class probability."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, X):
        return self.pipeline.predict_proba(X)[:, 1]


def ale_curves(pipeline, X: pd.DataFrame, features: list[str], grid_size: int = 15,
               sample: int = 6000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Accumulated local effects. Unlike partial dependence they stay honest when
    features are correlated, which distance, promise length and lane history are."""
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    model = ProbaModel(pipeline)
    out = {}
    for f in features:
        eff = ale(X=Xs, model=model, feature=[f], feature_type="continuous",
                  grid_size=grid_size, include_CI=False, plot=False)
        out[f] = eff.reset_index().rename(columns={eff.index.name or "index": "value"})
    return out


LABELS = {
    "promise_days": "the delivery promise is {v:.0f} days",
    "distance_km": "the parcel travels about {v:,.0f} km",
    "expected_slack": "the promise leaves {v:.1f} days over this lane's usual delivery time",
    "recent_slack": "the promise leaves {v:.1f} days over this lane's last 30 days",
    "route_mean_days": "this lane usually takes {v:.1f} days",
    "route_recent_days": "this lane has taken {v:.1f} days lately",
    "seller_late_rate": "this seller has missed {v:.0%} of past promises",
    "route_late_rate": "this lane has missed {v:.0%} of past promises",
    "state_late_rate": "orders to this state missed {v:.0%} of promises",
    "zone_late_rate": "orders to this zip area missed {v:.0%} of promises",
    "recent_late_rate": "{v:.0%} of orders resolved in the last two weeks were late",
    "seller_handling_days": "this seller usually takes {v:.1f} days to hand over",
    "freight_total": "freight is {v:.2f}",
    "log_weight": "the order is heavy",
    "platform_7d_orders": "the platform took {v:,.0f} orders in the past week",
    "seller_30d_orders": "the seller took {v:,.0f} orders in the past month",
    "handling_days": "the seller took {v:.1f} days to hand over",
    "days_left_at_handoff": "{v:.1f} days were left on the promise at handoff",
    "missed_ship_limit": "the seller missed its shipping deadline",
    "approval_hours": "payment approval took {v:.0f} hours",
    "customer_state": "the customer is in {v}",
    "seller_state": "the seller is in {v}",
}


def describe(feature: str, value) -> str:
    if feature in ("expected_slack", "recent_slack") and float(value) < 0:
        window = "usually takes" if feature == "expected_slack" else "has taken lately"
        return f"the promise is {abs(float(value)):.1f} days shorter than this lane {window}"
    if feature == "days_left_at_handoff" and float(value) < 0:
        days = abs(float(value))
        return f"the parcel reached the carrier {days:.1f} days after the promised date"
    template = LABELS.get(feature)
    if template is None:
        return f"{feature.replace('_', ' ')} is {value}"
    try:
        return template.format(v=value)
    except (ValueError, TypeError):
        return f"{feature.replace('_', ' ')} is {value}"


def reasons(pipeline, row: pd.DataFrame, reference: dict, top: int = 3,
            min_delta: float = 0.002) -> list[dict]:
    """Swap each input to a typical value and record how much the risk moves.

    Positive ``delta`` means this input pushes this order's risk up compared
    with a typical order. Model-agnostic, so it works for whichever model wins.
    """
    base = float(pipeline.predict_proba(row)[:, 1][0])
    probes = []
    for col, ref in reference.items():
        if col not in row.columns:
            continue
        alt = row.copy()
        alt[col] = ref
        probes.append((col, alt))
    batch = pd.concat([a for _, a in probes], ignore_index=True)
    alt_p = pipeline.predict_proba(batch)[:, 1]
    rows = []
    for (col, _), p in zip(probes, alt_p, strict=True):
        v = row[col].iloc[0]
        if isinstance(v, float) and np.isnan(v):
            continue
        rows.append({"feature": col, "value": v, "delta": base - float(p),
                     "text": describe(col, v)})
    rows.sort(key=lambda r: -r["delta"])
    return [r for r in rows if r["delta"] > min_delta][:top]
