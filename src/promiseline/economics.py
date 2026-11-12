"""Turn a ranking into money under openly stated assumptions.

Touching an order (a nudge to the seller, a carrier check, a note to the
customer) costs ``action_cost``; touching an order that was going to be late
rescues it with probability ``action_efficacy``; each late order that is not
rescued costs ``late_order_cost``. All three numbers are assumptions, so the
module also reports a sensitivity grid.

Two ways to decide what to touch are compared. A fixed list flags the top
share of each day's orders. A threshold policy flags an order when its
expected saving beats the action cost, p * efficacy * late_cost > action_cost,
which only works if p is a calibrated probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COSTS, CostAssumptions


def daily_topk(frame: pd.DataFrame, p: np.ndarray, share: float) -> np.ndarray:
    """Flag the top ``share`` of each purchase day's orders by risk."""
    day = frame["purchased_at"].dt.normalize().to_numpy()
    p = np.asarray(p, float)
    flag = np.zeros(len(p), dtype=bool)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        k = int(round(share * len(idx)))
        if k:
            flag[idx[np.argsort(-p[idx], kind="stable")[:k]]] = True
    return flag


def value_per_thousand(y: np.ndarray, flag: np.ndarray, costs: CostAssumptions = COSTS) -> dict:
    y = np.asarray(y, bool)
    caught = int((y & flag).sum())
    saved = caught * costs.action_efficacy * costs.late_order_cost
    spent = int(flag.sum()) * costs.action_cost
    n = len(y)
    return {"flagged": int(flag.sum()), "late_caught": caught, "late_total": int(y.sum()),
            "net_per_1000": float((saved - spent) / n * 1000),
            "saved_per_1000": float(saved / n * 1000), "spent_per_1000": float(spent / n * 1000)}


def threshold(costs: CostAssumptions = COSTS) -> float:
    """Risk above which touching an order has positive expected value."""
    return costs.action_cost / (costs.action_efficacy * costs.late_order_cost)


def threshold_flags(p: np.ndarray, costs: CostAssumptions = COSTS) -> np.ndarray:
    return np.asarray(p, float) >= threshold(costs)


def sensitivity(y: np.ndarray, p: np.ndarray, late_costs=(8.0, 14.0, 20.0),
                efficacies=(0.2, 0.35, 0.5), costs: CostAssumptions = COSTS) -> pd.DataFrame:
    """Net value of the threshold policy as the two softest assumptions move."""
    rows = []
    for lc in late_costs:
        for ef in efficacies:
            c = CostAssumptions(late_order_cost=lc, action_cost=costs.action_cost,
                                action_efficacy=ef, budget_share=costs.budget_share)
            flag = threshold_flags(p, c)
            v = value_per_thousand(y, flag, c)
            rows.append({"late_order_cost": lc, "efficacy": ef, "threshold": threshold(c),
                         "flagged_share": float(flag.mean()), "net_per_1000": v["net_per_1000"]})
    return pd.DataFrame(rows)


def break_even_cost(y: np.ndarray, flag: np.ndarray, costs: CostAssumptions = COSTS) -> float:
    """Late-order cost at which flagging exactly pays for itself."""
    caught = (np.asarray(y, bool) & flag).sum()
    if caught == 0:
        return float("inf")
    return float(flag.sum() * costs.action_cost / (caught * costs.action_efficacy))
