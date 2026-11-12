"""As-of aggregates: what the platform knew about a seller, route or customer
at the instant a new order was placed.

The core trick is one sort and one binary search. Events and queries are
mapped onto a composite integer (group code, seconds since epoch); after
sorting the events, ``np.searchsorted`` tells each query how many of its own
group's events happened strictly before it, and a cumulative sum turns that
position into a running total. That is O((n + m) log n) for the whole table
instead of a Python loop per order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SPAN = np.int64(10**10)  # seconds; comfortably wider than the 2016-2018 window


def _codes(event_keys: pd.Series, query_keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cats = pd.Index(pd.unique(pd.concat([event_keys, query_keys], ignore_index=True)
                              .astype(str)))
    return (cats.get_indexer(event_keys.astype(str)).astype(np.int64),
            cats.get_indexer(query_keys.astype(str)).astype(np.int64))


def _seconds(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(pd.Series(ts)).fillna(pd.Timestamp("1970-01-01"))
    return t.astype("datetime64[s]").astype(np.int64).to_numpy()


def asof_sum_count(event_keys: pd.Series, event_time: pd.Series, event_value: pd.Series,
                   query_keys: pd.Series, query_time: pd.Series,
                   window_days: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of event values in the query's group known strictly before query time.

    With ``window_days`` only events from the trailing window [t - window, t) count.
    """
    ek, qk = _codes(event_keys, query_keys)
    et, qt = _seconds(event_time), _seconds(query_time)
    val = np.asarray(event_value, dtype=float)
    keep = ~np.isnan(val) & (ek >= 0) & pd.to_datetime(pd.Series(event_time)).notna().to_numpy()
    ek, et, val = ek[keep], et[keep], val[keep]
    comp_e = ek * _SPAN + et
    order = np.argsort(comp_e, kind="stable")
    comp_e, val = comp_e[order], val[order]
    csum = np.concatenate([[0.0], np.cumsum(val)])
    hi = np.searchsorted(comp_e, qk * _SPAN + qt, side="left")
    if window_days is None:
        lo = np.searchsorted(comp_e, qk * _SPAN, side="left")
    else:
        start = np.maximum(qt - np.int64(window_days * 86400), 0)
        lo = np.searchsorted(comp_e, qk * _SPAN + start, side="left")
    return csum[hi] - csum[lo], (hi - lo).astype(float)


def window_count(event_keys: pd.Series, event_time: pd.Series, query_keys: pd.Series,
                 query_time: pd.Series, days: float) -> np.ndarray:
    """Number of group events in the half-open window [t - days, t)."""
    ek, qk = _codes(event_keys, query_keys)
    et, qt = _seconds(event_time), _seconds(query_time)
    comp_e = np.sort(ek * _SPAN + et)
    q = qk * _SPAN + qt
    width = np.int64(days * 86400)
    return (np.searchsorted(comp_e, q, side="left")
            - np.searchsorted(comp_e, q - width, side="left")).astype(float)


def shrunk_rate(total: np.ndarray, count: np.ndarray, prior: float, strength: float) -> np.ndarray:
    """Empirical-Bayes style shrinkage toward a prior rate with pseudo-count ``strength``."""
    return (total + prior * strength) / (count + strength)
