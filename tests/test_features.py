import numpy as np
import pandas as pd

from promiseline.features import (
    CHECKOUT_NUMERIC,
    HANDOFF_NUMERIC,
    add_checkout_features,
    add_handoff_features,
)


def test_checkout_features_ignore_the_future(orders):
    """Rewriting outcomes after a cut-off must not change any feature before it."""
    cut = orders["purchased_at"].quantile(0.6)
    base = add_checkout_features(orders)
    changed = orders.copy()
    future = changed["label_known_at"] >= cut
    changed.loc[future, "late"] = 1 - changed.loc[future, "late"]
    later = changed["delivered_at"] >= cut
    changed.loc[later, "delivered_at"] = changed.loc[later, "delivered_at"] + pd.Timedelta(days=9)
    after = add_checkout_features(changed)
    before = orders["purchased_at"] < cut
    a = base.loc[before, CHECKOUT_NUMERIC].to_numpy(float)
    b = after.loc[before, CHECKOUT_NUMERIC].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


def test_checkout_columns_exist_and_rates_are_bounded(featured):
    for col in CHECKOUT_NUMERIC + HANDOFF_NUMERIC:
        assert col in featured.columns
    for col in ["seller_late_rate", "route_late_rate", "state_late_rate", "zone_late_rate"]:
        assert featured[col].between(0, 1).all()


def test_first_order_of_a_seller_has_no_history(featured):
    first = featured.sort_values("purchased_at").drop_duplicates("seller_id")
    assert (first["seller_history_n"] == 0).all()


def test_handoff_features_arithmetic(orders):
    row = add_handoff_features(orders.head(1)).iloc[0]
    expected = (row["carrier_at"] - row["purchased_at"]).total_seconds() / 86400
    assert abs(row["handling_days"] - expected) < 1e-9
    assert row["missed_ship_limit"] == int(row["carrier_at"] > row["shipping_limit"])
