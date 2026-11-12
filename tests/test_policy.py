import numpy as np
import pandas as pd

from promiseline.calibrate import apply_shift, intercept_shift, matured_window
from promiseline.config import CostAssumptions
from promiseline.economics import daily_topk, threshold, threshold_flags, value_per_thousand
from promiseline.promise import conformal_offset, constant_shift, on_time_rate, quote


def test_intercept_shift_hits_target_and_keeps_order():
    p = np.random.default_rng(0).uniform(0.05, 0.6, 500)
    shift = intercept_shift(p, 0.05)
    q = apply_shift(p, shift)
    assert abs(q.mean() - 0.05) < 1e-6
    assert (np.argsort(q) == np.argsort(p)).all()


def test_matured_window_only_uses_known_outcomes():
    h = pd.DataFrame({
        "purchased_at": pd.to_datetime(["2018-01-01", "2018-01-10", "2018-01-25"]),
        "label_known_at": pd.to_datetime(["2018-01-15", "2018-02-05", "2018-01-30"]),
        "late": [0, 1, 0],
    })
    w = matured_window(h, pd.Timestamp("2018-02-01"))
    assert w.index.tolist() == [0]


def test_threshold_policy_math():
    c = CostAssumptions(late_order_cost=10, action_cost=2, action_efficacy=0.5)
    assert threshold(c) == 0.4
    assert threshold_flags(np.array([0.39, 0.4, 0.9]), c).tolist() == [False, True, True]
    v = value_per_thousand(np.array([1, 0, 1, 0]), np.array([True, True, False, False]), c)
    assert v["late_caught"] == 1
    assert v["net_per_1000"] == (1 * 0.5 * 10 - 2 * 2) / 4 * 1000


def test_daily_topk_respects_each_day():
    f = pd.DataFrame({"purchased_at": pd.to_datetime(["2018-01-01"] * 4 + ["2018-01-02"] * 4)})
    p = np.array([0.9, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.5])
    flag = daily_topk(f, p, 0.25)
    assert flag.tolist() == [True, False, False, False, False, False, False, True]


def test_conformal_offset_reaches_coverage():
    rng = np.random.default_rng(1)
    pred = rng.uniform(5, 15, 2000)
    actual = pred + rng.normal(0, 3, 2000)
    off = conformal_offset(pred, actual, 0.9)
    assert on_time_rate(pred + off, actual) >= 0.9
    q = quote(np.array([0.2, 10.4]), 0.0)
    assert q.tolist() == [2.0, 11.0]


def test_constant_shift_stops_at_target():
    olist = np.full(100, 20.0)
    actual = np.arange(100) / 10.0  # 0 to 9.9 days
    assert constant_shift(olist, actual, 0.9) == 11
    assert on_time_rate(olist - 11, actual) >= 0.9
