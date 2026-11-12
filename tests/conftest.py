"""A small synthetic Olist-shaped export so the tests never need the download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from promiseline.data import build_orders, read_raw
from promiseline.features import add_checkout_features, add_handoff_features

STATES = {"SP": (-23.5, -46.6), "RJ": (-22.9, -43.2), "MG": (-19.9, -43.9), "BA": (-12.9, -38.5)}


def write_raw(path, n_orders: int = 900, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    zips = {s: [f"{i}{j:04d}" for j in range(3)] for i, s in enumerate(STATES, start=1)}
    geo = []
    for s, (lat, lng) in STATES.items():
        for z in zips[s]:
            for _ in range(3):
                geo.append([z, lat + rng.normal(0, 0.1), lng + rng.normal(0, 0.1), "x", s])
    geo.append(["99999", 48.8, 2.3, "paris", "SP"])  # a stray point outside Brazil
    pd.DataFrame(geo, columns=["geolocation_zip_code_prefix", "geolocation_lat",
                               "geolocation_lng", "geolocation_city", "geolocation_state"]
                 ).to_csv(path / "olist_geolocation_dataset.csv", index=False)

    sellers = [(f"s{i}", rng.choice(zips["SP"] + zips["MG"]), "c") for i in range(12)]
    sel = pd.DataFrame(sellers, columns=["seller_id", "seller_zip_code_prefix", "seller_city"])
    sel["seller_state"] = sel["seller_zip_code_prefix"].map(
        {z: s for s, zs in zips.items() for z in zs})
    sel.to_csv(path / "olist_sellers_dataset.csv", index=False)

    cats = ["bed_bath_table", "health_beauty", "toys"]
    prods = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(30)],
        "product_category_name": [cats[i % 3] for i in range(30)],
        "product_name_lenght": 40, "product_description_lenght": rng.integers(50, 900, 30),
        "product_photos_qty": rng.integers(1, 5, 30),
        "product_weight_g": rng.integers(100, 9000, 30),
        "product_length_cm": 20, "product_height_cm": 10, "product_width_cm": 15,
    })
    prods.to_csv(path / "olist_products_dataset.csv", index=False)
    pd.DataFrame({"product_category_name": cats, "product_category_name_english": cats}
                 ).to_csv(path / "product_category_name_translation.csv", index=False)

    start = pd.Timestamp("2017-01-02")
    purchase = start + pd.to_timedelta(np.sort(rng.uniform(0, 600, n_orders)), unit="D")
    cust_state = rng.choice(list(STATES), n_orders)
    cust = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n_orders)],
        "customer_unique_id": [f"u{i % (n_orders - 50)}" for i in range(n_orders)],
        "customer_zip_code_prefix": [rng.choice(zips[s]) for s in cust_state],
        "customer_city": "x", "customer_state": cust_state,
    })
    cust.to_csv(path / "olist_customers_dataset.csv", index=False)

    seller_pick = rng.integers(0, 12, n_orders)
    far = (cust_state == "BA").astype(float)
    handling = rng.gamma(2, 1.2, n_orders)
    transit = rng.gamma(3, 2.5, n_orders) + 6 * far + (seller_pick % 4 == 0) * 4
    promise = np.round(transit.mean() + 8 + 6 * far + rng.normal(0, 2, n_orders))
    approved = purchase + pd.to_timedelta(rng.uniform(0.01, 1, n_orders), unit="D")
    carrier = purchase + pd.to_timedelta(handling, unit="D")
    delivered = carrier + pd.to_timedelta(transit, unit="D")
    status = np.array(["delivered"] * n_orders, dtype=object)
    status[rng.choice(n_orders, 10, replace=False)] = "canceled"
    lost = rng.choice(np.flatnonzero(status == "delivered"), 8, replace=False)
    status[lost] = "shipped"
    delivered = pd.Series(delivered).where(status == "delivered")
    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n_orders)], "customer_id": cust["customer_id"],
        "order_status": status, "order_purchase_timestamp": purchase,
        "order_approved_at": approved, "order_delivered_carrier_date": carrier,
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": purchase.normalize() + pd.to_timedelta(promise, unit="D"),
    })
    orders.to_csv(path / "olist_orders_dataset.csv", index=False)

    items = pd.DataFrame({
        "order_id": orders["order_id"], "order_item_id": 1,
        "product_id": [f"p{i}" for i in rng.integers(0, 30, n_orders)],
        "seller_id": [f"s{i}" for i in seller_pick],
        "shipping_limit_date": purchase + pd.Timedelta(days=3),
        "price": rng.gamma(2, 50, n_orders).round(2),
        "freight_value": rng.gamma(2, 8, n_orders).round(2),
    })
    items.to_csv(path / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame({"order_id": orders["order_id"], "payment_sequential": 1,
                  "payment_type": rng.choice(["credit_card", "boleto"], n_orders),
                  "payment_installments": rng.integers(1, 6, n_orders),
                  "payment_value": items["price"] + items["freight_value"]}
                 ).to_csv(path / "olist_order_payments_dataset.csv", index=False)
    arrived = pd.Series(delivered).dt.normalize()
    late = (arrived > orders["order_estimated_delivery_date"]).to_numpy()
    score = np.where(late, rng.integers(1, 3, n_orders), rng.integers(4, 6, n_orders))
    pd.DataFrame({"review_id": [f"r{i}" for i in range(n_orders)], "order_id": orders["order_id"],
                  "review_score": score, "review_comment_title": "", "review_comment_message": "",
                  "review_creation_date": arrived + pd.Timedelta(days=1),
                  "review_answer_timestamp": pd.Series(delivered) + pd.Timedelta(days=2)}
                 ).to_csv(path / "olist_order_reviews_dataset.csv", index=False)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw")
    write_raw(path)
    return path


@pytest.fixture(scope="session")
def orders(raw_dir):
    return build_orders(read_raw(raw_dir))


@pytest.fixture(scope="session")
def featured(orders):
    return add_handoff_features(add_checkout_features(orders))
