"""Score one order from a plain dictionary. The Gradio app is a thin wrapper
around ``score_order`` so the logic can be unit-tested without a browser."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibrate import apply_shift
from .explain import reasons
from .features import CHECKOUT_CATEGORICAL, CHECKOUT_NUMERIC, HANDOFF_NUMERIC

BANDS = (("high", 0.90), ("elevated", 0.75))


def reference_row(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    ref = {c: float(frame[c].median()) for c in numeric}
    ref.update({c: str(frame[c].mode().iloc[0]) for c in categorical})
    return ref


def build_row(order: dict, reference: dict) -> pd.DataFrame:
    """Fill gaps from the reference order and recompute the derived columns."""
    row = dict(reference)
    row.update({k: v for k, v in order.items() if v is not None})
    if "price_total" in order:
        row["log_price"] = float(np.log1p(order["price_total"]))
    if "weight_g" in order:
        row["log_weight"] = float(np.log1p(order["weight_g"]))
    if "price_total" in order or "freight_total" in order:
        price = order.get("price_total", np.expm1(row["log_price"]))
        row["freight_ratio"] = float(row["freight_total"]) / max(float(price), 1.0)
    row["same_state"] = int(row["customer_state"] == row["seller_state"])
    row["expected_slack"] = float(row["promise_days"]) - float(row["route_mean_days"])
    row["recent_slack"] = float(row["promise_days"]) - float(row["route_recent_days"])
    if "days_left_at_handoff" in reference and "handling_days" in order:
        row["days_left_at_handoff"] = float(row["promise_days"]) - float(row["handling_days"])
    return pd.DataFrame([row])


def band(risk: float, cutoffs: dict) -> str:
    for name, q in BANDS:
        if risk >= cutoffs[str(q)]:
            return name
    return "normal"


def score_order(bundle: dict, order: dict, stage: str = "checkout") -> dict:
    """Late-delivery risk, band and up to three reasons for one order."""
    if stage not in ("checkout", "handoff"):
        raise ValueError("stage must be 'checkout' or 'handoff'")
    model = bundle[f"{stage}_model"]
    reference = bundle[f"{stage}_reference"]
    numeric = CHECKOUT_NUMERIC + (HANDOFF_NUMERIC if stage == "handoff" else [])
    row = build_row(order, reference)[numeric + CHECKOUT_CATEGORICAL]
    raw = float(model.predict_proba(row)[:, 1][0])
    risk = float(apply_shift(np.array([raw]), bundle.get(f"{stage}_shift", 0.0))[0])
    why = reasons(model, row, {k: reference[k] for k in numeric + CHECKOUT_CATEGORICAL})
    return {"stage": stage, "risk": risk, "band": band(raw, bundle[f"{stage}_cutoffs"]),
            "reasons": [r["text"] for r in why]}
