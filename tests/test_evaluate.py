import numpy as np
import pandas as pd

from promiseline.evaluate import bootstrap, capture_at, ece, interval, metrics, one_se_choice


def test_ece_zero_for_matching_bins():
    p = np.repeat([0.1, 0.5], 100)
    y = np.r_[np.r_[np.ones(10), np.zeros(90)], np.r_[np.ones(50), np.zeros(50)]]
    assert ece(y, p, bins=2) < 1e-12


def test_capture_and_metrics():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    assert capture_at(y, p, 0.2) == 1.0
    m = metrics(y, p)
    assert m["auc"] == 1.0 and m["base_rate"] == 0.2


def test_one_se_prefers_the_simpler_model_when_close():
    point = {"simple": 0.30, "fancy": 0.31}
    boots = {"simple": np.array([0.29, 0.31, 0.30, 0.28]),
             "fancy": np.array([0.31, 0.30, 0.33, 0.30])}
    chosen, detail = one_se_choice(point, boots, ["simple", "fancy"])
    assert chosen == "simple" and detail["best"] == "fancy"
    far = {"simple": 0.10, "fancy": 0.31}
    assert one_se_choice(far, boots, ["simple", "fancy"])[0] == "fancy"


def test_day_bootstrap_is_paired_and_finite():
    rng = np.random.default_rng(0)
    offsets = pd.to_timedelta(np.repeat(np.arange(30), 20), "D")
    days = pd.Series(pd.Timestamp("2018-01-01") + offsets)
    y = rng.integers(0, 2, len(days))
    p = {"a": rng.random(len(days)), "b": rng.random(len(days))}
    b = bootstrap(y, p, days, "ap", n_boot=50)
    lo, hi = interval(b["a"])
    assert 0 <= lo <= hi <= 1
    assert len(b["a"]) == len(b["b"]) == 50
