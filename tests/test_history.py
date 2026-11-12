import numpy as np
import pandas as pd

from promiseline.history import asof_sum_count, shrunk_rate, window_count


def brute(ek, et, ev, qk, qt, window=None):
    tot, cnt = [], []
    for k, t in zip(qk, qt, strict=True):
        m = (ek == k) & (et < t)
        if window is not None:
            m &= et >= t - pd.Timedelta(days=window)
        tot.append(ev[m].sum())
        cnt.append(m.sum())
    return np.array(tot, float), np.array(cnt, float)


def test_asof_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 400
    ek = pd.Series(rng.choice(list("abcde"), n))
    et = pd.Series(pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.uniform(0, 300, n), unit="D"))
    ev = pd.Series(rng.integers(0, 2, n).astype(float))
    qk = pd.Series(rng.choice(list("abcdef"), 150))
    qt = pd.Series(pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.uniform(0, 320, 150), unit="D"))
    tot, cnt = asof_sum_count(ek, et, ev, qk, qt)
    bt, bc = brute(ek.to_numpy(), et.to_numpy(), ev.to_numpy(), qk.to_numpy(), qt.to_numpy())
    assert np.allclose(tot, bt) and np.allclose(cnt, bc)
    tot, cnt = asof_sum_count(ek, et, ev, qk, qt, window_days=30)
    bt, bc = brute(ek.to_numpy(), et.to_numpy(), ev.to_numpy(), qk.to_numpy(), qt.to_numpy(), 30)
    assert np.allclose(tot, bt) and np.allclose(cnt, bc)


def test_missing_values_and_times_are_ignored():
    ek = pd.Series(["a", "a", "a"])
    et = pd.Series([pd.Timestamp("2017-01-01"), pd.NaT, pd.Timestamp("2017-01-03")])
    ev = pd.Series([1.0, 1.0, np.nan])
    tot, cnt = asof_sum_count(ek, et, ev, pd.Series(["a"]), pd.Series([pd.Timestamp("2018-01-01")]))
    assert tot[0] == 1.0 and cnt[0] == 1.0


def test_window_count_excludes_the_query_instant():
    t = pd.Series(pd.to_datetime(["2017-01-01", "2017-01-05", "2017-01-10"]))
    k = pd.Series(["x", "x", "x"])
    counts = window_count(k, t, k, t, days=7)
    assert counts.tolist() == [0.0, 1.0, 1.0]


def test_shrunk_rate_moves_toward_prior_with_little_data():
    assert shrunk_rate(np.array([1.0]), np.array([1.0]), 0.1, 9.0)[0] == 0.19
    big = shrunk_rate(np.array([500.0]), np.array([1000.0]), 0.1, 9.0)[0]
    assert abs(big - 0.5) < 0.01
