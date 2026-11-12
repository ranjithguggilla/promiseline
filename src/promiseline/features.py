"""Feature construction for the two decision points.

Checkout features use only what exists when the customer clicks buy. Handoff
features add what the platform learns when the seller passes the parcel to
the carrier. Nothing here reads delivery dates or reviews.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .history import asof_sum_count, shrunk_rate, window_count

CHECKOUT_NUMERIC = [
    "promise_days", "distance_km", "log_price", "freight_total", "freight_ratio",
    "log_weight", "log_volume", "n_items", "n_sellers", "photos", "desc_len",
    "installments", "voucher", "purchase_hour", "purchase_dow", "same_state",
    "platform_7d_orders", "seller_30d_orders", "seller_history_n", "seller_late_rate",
    "route_late_rate", "state_late_rate", "recent_late_rate", "seller_handling_days",
    "customer_prior_orders", "route_mean_days", "route_recent_days", "zone_mean_days",
    "zone_late_rate", "expected_slack", "recent_slack",
]
CHECKOUT_CATEGORICAL = ["customer_state", "seller_state", "payment_type", "main_category"]
HANDOFF_NUMERIC = ["handling_days", "approval_hours", "missed_ship_limit", "days_left_at_handoff"]

PRIOR_STRENGTH = 20.0


def _platform_recent_rate(df: pd.DataFrame, days: int = 14) -> np.ndarray:
    """Late share among outcomes that became known in the last ``days`` days."""
    known = df[["label_known_at", "late"]].sort_values("label_known_at")
    t = known["label_known_at"].astype("datetime64[s]").astype(np.int64).to_numpy()
    c = np.concatenate([[0.0], np.cumsum(known["late"].to_numpy(float))])
    q = df["purchased_at"].astype("datetime64[s]").astype(np.int64).to_numpy()
    hi = np.searchsorted(t, q, side="left")
    lo = np.searchsorted(t, q - days * 86400, side="left")
    n = hi - lo
    rate = np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan)
    return rate


def _asof_mean(ek, et, ev, qk, qt) -> np.ndarray:
    t, n = asof_sum_count(ek, et, ev, qk, qt)
    return np.where(n > 0, t / np.maximum(n, 1), 12.0)


def add_checkout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every checkout-time feature attached."""
    out = df.copy()
    out["log_price"] = np.log1p(out["price_total"])
    out["freight_ratio"] = out["freight_total"] / out["price_total"].clip(lower=1.0)
    out["log_weight"] = np.log1p(out["weight_g"])
    out["log_volume"] = np.log1p(out["max_volume_cm3"])
    out["purchase_hour"] = out["purchased_at"].dt.hour
    out["purchase_dow"] = out["purchased_at"].dt.dayofweek
    out["same_state"] = (out["customer_state"] == out["seller_state"]).astype(int)

    everyone = pd.Series("all", index=out.index)
    out["platform_7d_orders"] = window_count(everyone, out["purchased_at"], everyone,
                                             out["purchased_at"], 7)
    out["seller_30d_orders"] = window_count(out["seller_id"], out["purchased_at"],
                                            out["seller_id"], out["purchased_at"], 30)

    # Platform-wide as-of late rate is the prior every narrower rate shrinks toward.
    tot, cnt = asof_sum_count(everyone, out["label_known_at"], out["late"], everyone,
                              out["purchased_at"])
    prior = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.07)

    def rate(keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        t, n = asof_sum_count(keys, out["label_known_at"], out["late"], keys, out["purchased_at"])
        return shrunk_rate(t, n, prior, PRIOR_STRENGTH), n

    out["seller_late_rate"], out["seller_history_n"] = rate(out["seller_id"])
    route = out["seller_state"].astype(str) + ">" + out["customer_state"].astype(str)
    out["route_late_rate"], _ = rate(route)
    out["state_late_rate"], _ = rate(out["customer_state"])
    out["recent_late_rate"] = _platform_recent_rate(out)

    # How long deliveries on this lane have actually been taking, as known at checkout.
    days = (out["delivered_at"] - out["purchased_at"]).dt.total_seconds() / 86400
    zone = out["customer_zip_code_prefix"].astype(str).str[:3]
    prior_days = _asof_mean(everyone, out["delivered_at"], days, everyone, out["purchased_at"])
    t, n = asof_sum_count(route, out["delivered_at"], days, route, out["purchased_at"])
    out["route_mean_days"] = (t + prior_days * 10) / (n + 10)
    t, n = asof_sum_count(route, out["delivered_at"], days, route, out["purchased_at"],
                          window_days=30)
    out["route_recent_days"] = np.where(n >= 5, t / np.maximum(n, 1), out["route_mean_days"])
    t, n = asof_sum_count(zone, out["delivered_at"], days, zone, out["purchased_at"])
    out["zone_mean_days"] = (t + prior_days * 10) / (n + 10)
    out["zone_late_rate"], _ = rate(zone)
    out["expected_slack"] = out["promise_days"] - out["route_mean_days"]
    out["recent_slack"] = out["promise_days"] - out["route_recent_days"]

    handling = ((out["carrier_at"] - out["purchased_at"]).dt.total_seconds() / 86400).clip(0, 60)
    tot_h, cnt_h = asof_sum_count(out["seller_id"], out["carrier_at"], handling,
                                  out["seller_id"], out["purchased_at"])
    out["seller_handling_days"] = np.where(cnt_h > 0, tot_h / np.maximum(cnt_h, 1), np.nan)

    out["customer_prior_orders"] = window_count(out["customer_unique_id"], out["purchased_at"],
                                                out["customer_unique_id"], out["purchased_at"],
                                                3650)
    return out


def add_handoff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["handling_days"] = (out["carrier_at"] - out["purchased_at"]).dt.total_seconds() / 86400
    out["approval_hours"] = (out["approved_at"] - out["purchased_at"]).dt.total_seconds() / 3600
    out["missed_ship_limit"] = (out["carrier_at"] > out["shipping_limit"]).astype(int)
    out["days_left_at_handoff"] = ((out["promised_date"] - out["carrier_at"])
                                   .dt.total_seconds() / 86400)
    return out
