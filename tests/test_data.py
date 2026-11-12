import pandas as pd

from promiseline.data import label_orders, split_frame, zip_centroids


def _orders(**cols):
    base = {
        "order_id": ["a"], "customer_id": ["c"], "order_status": ["delivered"],
        "order_purchase_timestamp": [pd.Timestamp("2017-05-01 10:00")],
        "order_approved_at": [pd.Timestamp("2017-05-01 11:00")],
        "order_delivered_carrier_date": [pd.Timestamp("2017-05-02")],
        "order_delivered_customer_date": [pd.Timestamp("2017-05-10 23:00")],
        "order_estimated_delivery_date": [pd.Timestamp("2017-05-10")],
    }
    base.update({k: [v] for k, v in cols.items()})
    return pd.DataFrame(base)


def test_same_day_arrival_is_on_time_and_known_at_delivery():
    o = label_orders(_orders())
    assert o["late"].iloc[0] == 0
    assert o["label_known_at"].iloc[0] == pd.Timestamp("2017-05-10 23:00")


def test_next_day_arrival_is_late_and_known_after_the_promise():
    o = label_orders(_orders(order_delivered_customer_date=pd.Timestamp("2017-05-11 08:00")))
    assert o["late"].iloc[0] == 1
    assert o["label_known_at"].iloc[0] == pd.Timestamp("2017-05-11")


def test_never_arrived_counts_as_late_and_cancelled_is_dropped():
    lost = label_orders(_orders(order_status="shipped", order_delivered_customer_date=pd.NaT))
    assert lost["late"].iloc[0] == 1 and lost["never_arrived"].iloc[0] == 1
    assert label_orders(_orders(order_status="canceled")).empty


def test_unexpired_undelivered_orders_have_no_label():
    o = _orders(order_status="shipped", order_delivered_customer_date=pd.NaT,
                order_estimated_delivery_date=pd.Timestamp("2018-12-01"))
    assert label_orders(o).empty


def test_centroids_drop_points_outside_brazil(raw_dir):
    geo = pd.read_csv(raw_dir / "olist_geolocation_dataset.csv",
                      dtype={"geolocation_zip_code_prefix": str})
    cent = zip_centroids(geo)
    assert "99999" not in cent.index
    assert cent["lat"].between(-34, 6).all()


def test_build_orders_shape(orders):
    assert orders["order_id"].is_unique
    assert set(orders["late"].unique()) <= {0, 1}
    assert orders["distance_km"].notna().all()
    assert (orders["promise_days"] > 0).all()
    assert orders["purchased_at"].is_monotonic_increasing


def test_splits_are_disjoint_and_ordered(featured):
    sp = split_frame(featured)
    assert sp["train"]["purchased_at"].max() < sp["valid"]["purchased_at"].min()
    assert sp["valid"]["purchased_at"].max() < sp["test"]["purchased_at"].min()
