"""Load the Olist tables and collapse them to one row per order.

Every column produced here is tagged by when it becomes known: at checkout,
at carrier handoff, or only after the outcome. The feature module reads only
from the first two groups; the audit module deliberately reaches into the
third to show what leakage would look like.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector

from .config import RAW_DIR, SPLITS

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Rough bounding box for Brazil; the geolocation table has a few points in
# Europe and the Atlantic that would otherwise produce 9,000 km "routes".
LAT_RANGE = (-34.0, 5.5)
LNG_RANGE = (-74.0, -34.0)


def read_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    zip_dtype = {"customer_zip_code_prefix": str, "seller_zip_code_prefix": str,
                 "geolocation_zip_code_prefix": str}
    tables = {
        "orders": pd.read_csv(raw_dir / "olist_orders_dataset.csv", parse_dates=DATE_COLS),
        "items": pd.read_csv(raw_dir / "olist_order_items_dataset.csv",
                             parse_dates=["shipping_limit_date"]),
        "payments": pd.read_csv(raw_dir / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(raw_dir / "olist_order_reviews_dataset.csv",
                               parse_dates=["review_creation_date", "review_answer_timestamp"]),
        "customers": pd.read_csv(raw_dir / "olist_customers_dataset.csv", dtype=zip_dtype),
        "sellers": pd.read_csv(raw_dir / "olist_sellers_dataset.csv", dtype=zip_dtype),
        "products": pd.read_csv(raw_dir / "olist_products_dataset.csv"),
        "geo": pd.read_csv(raw_dir / "olist_geolocation_dataset.csv", dtype=zip_dtype),
        "categories": pd.read_csv(raw_dir / "product_category_name_translation.csv"),
    }
    return tables


def zip_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    """Median coordinate per 5-digit zip prefix, after dropping points outside Brazil."""
    g = geo.rename(columns={"geolocation_zip_code_prefix": "zip", "geolocation_lat": "lat",
                            "geolocation_lng": "lng", "geolocation_state": "state"})
    inside = g["lat"].between(*LAT_RANGE) & g["lng"].between(*LNG_RANGE)
    g = g.loc[inside]
    cent = g.groupby("zip").agg(lat=("lat", "median"), lng=("lng", "median"))
    return cent


def state_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    g = geo.rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng",
                            "geolocation_state": "state"})
    inside = g["lat"].between(*LAT_RANGE) & g["lng"].between(*LNG_RANGE)
    return g.loc[inside].groupby("state").agg(lat=("lat", "median"), lng=("lng", "median"))


def locate(zips: pd.Series, states: pd.Series, cent: pd.DataFrame,
           scent: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Zip-prefix coordinates with a state-centroid fallback for unknown prefixes."""
    lat = zips.map(cent["lat"])
    lng = zips.map(cent["lng"])
    lat = lat.fillna(states.map(scent["lat"]))
    lng = lng.fillna(states.map(scent["lng"]))
    return lat.to_numpy(float), lng.to_numpy(float)


def label_orders(orders: pd.DataFrame, snapshot: pd.Timestamp = SPLITS.snapshot) -> pd.DataFrame:
    """Attach the late label and the moment that label became knowable.

    An order is late when it reaches the customer on a calendar day after the
    promised date, or when it never arrived and the promise expired before the
    export was taken. Cancelled and unavailable orders carry no label.
    """
    o = orders.loc[~orders["order_status"].isin(["canceled", "unavailable"])].copy()
    delivered = o["order_delivered_customer_date"].notna()
    # "delivered" status with no delivery timestamp: outcome unknown, drop.
    o = o.loc[delivered | (o["order_status"] != "delivered")].copy()
    delivered = o["order_delivered_customer_date"].notna()
    est = o["order_estimated_delivery_date"]
    expired = est + pd.Timedelta(days=1) <= snapshot
    o = o.loc[delivered | expired].copy()
    delivered = o["order_delivered_customer_date"].notna()
    est = o["order_estimated_delivery_date"]
    late_delivered = o["order_delivered_customer_date"].dt.normalize() > est
    o["late"] = np.where(delivered, late_delivered, True).astype(int)
    o["never_arrived"] = (~delivered).astype(int)
    # On-time orders are known at delivery; late ones the morning after the promise.
    o["label_known_at"] = np.where(
        o["late"].astype(bool), est + pd.Timedelta(days=1), o["order_delivered_customer_date"]
    )
    o["label_known_at"] = pd.to_datetime(o["label_known_at"])
    return o


def build_orders(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per labelled order with checkout, handoff and outcome columns."""
    o = label_orders(tables["orders"])
    items = tables["items"].merge(tables["products"], on="product_id", how="left")
    items = items.merge(tables["categories"], on="product_category_name", how="left")
    items["category"] = items["product_category_name_english"].fillna(
        items["product_category_name"]).fillna("unknown")
    items["volume_cm3"] = (items["product_length_cm"] * items["product_height_cm"]
                           * items["product_width_cm"])

    items = items.merge(tables["sellers"], on="seller_id", how="left")
    cust = tables["customers"]
    items = items.merge(o[["order_id", "customer_id"]], on="order_id", how="inner")
    items = items.merge(cust[["customer_id", "customer_zip_code_prefix", "customer_state"]],
                        on="customer_id", how="left")

    cent = zip_centroids(tables["geo"])
    scent = state_centroids(tables["geo"])
    s_lat, s_lng = locate(items["seller_zip_code_prefix"], items["seller_state"], cent, scent)
    c_lat, c_lng = locate(items["customer_zip_code_prefix"], items["customer_state"], cent, scent)
    ok = ~(np.isnan(s_lat) | np.isnan(c_lat))
    dist = np.full(len(items), np.nan)
    dist[ok] = haversine_vector(np.c_[s_lat[ok], s_lng[ok]], np.c_[c_lat[ok], c_lng[ok]],
                                unit=Unit.KILOMETERS)
    items["distance_km"] = dist

    # The highest-value line decides the "main" seller and category of an order.
    items = items.sort_values(["order_id", "price"], ascending=[True, False])
    main = items.drop_duplicates("order_id")[
        ["order_id", "seller_id", "seller_state", "seller_zip_code_prefix", "category"]
    ].rename(columns={"category": "main_category"})
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        n_products=("product_id", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"),
        max_volume_cm3=("volume_cm3", "max"),
        photos=("product_photos_qty", "mean"),
        desc_len=("product_description_lenght", "mean"),
        distance_km=("distance_km", "max"),
        shipping_limit=("shipping_limit_date", "min"),
    ).reset_index()

    pay = tables["payments"].sort_values(["order_id", "payment_value"], ascending=[True, False])
    pay_main = pay.drop_duplicates("order_id")[["order_id", "payment_type"]]
    pay_agg = pay.groupby("order_id").agg(
        installments=("payment_installments", "max"),
        n_payments=("payment_sequential", "size"),
        voucher=("payment_type", lambda s: int((s == "voucher").any())),
    ).reset_index()

    rev = tables["reviews"].sort_values("review_answer_timestamp").drop_duplicates("order_id")
    rev = rev[["order_id", "review_score", "review_creation_date"]]

    df = (o.merge(cust, on="customer_id", how="left")
          .merge(agg, on="order_id", how="inner")
          .merge(main, on="order_id", how="left")
          .merge(pay_main, on="order_id", how="left")
          .merge(pay_agg, on="order_id", how="left")
          .merge(rev, on="order_id", how="left"))

    df = df.rename(columns={
        "order_purchase_timestamp": "purchased_at",
        "order_approved_at": "approved_at",
        "order_delivered_carrier_date": "carrier_at",
        "order_delivered_customer_date": "delivered_at",
        "order_estimated_delivery_date": "promised_date",
    })
    df["promise_days"] = (df["promised_date"] - df["purchased_at"].dt.normalize()).dt.days
    df["delivery_days"] = (df["delivered_at"] - df["purchased_at"]).dt.total_seconds() / 86400
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["installments"] = df["installments"].fillna(1)
    df["n_payments"] = df["n_payments"].fillna(1)
    df["voucher"] = df["voucher"].fillna(0).astype(int)
    df = df.sort_values("purchased_at").reset_index(drop=True)
    return df


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train, validation and test windows by purchase date."""
    t = df["purchased_at"]
    return {
        "train": df.loc[(t >= SPLITS.train_start) & (t < SPLITS.valid_start)],
        "valid": df.loc[(t >= SPLITS.valid_start) & (t < SPLITS.test_start)],
        "test": df.loc[(t >= SPLITS.test_start) & (t < SPLITS.test_end)],
    }
