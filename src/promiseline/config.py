"""Paths, split dates and constants shared across the package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("PROMISELINE_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
ASSET_DIR = ROOT / "assets"

# The Olist tables are mirrored in a public GitHub repo; pin the commit so the
# numbers in the README can be reproduced byte for byte.
SOURCE_REPO = "https://github.com/tunguyenn99/ecommerce-data-modeling.git"
SOURCE_COMMIT = "63401c2739ae42dd9bbfa7e20227fe7b7de725a3"
SOURCE_SUBDIR = "dataset"
RAW_FILES = (
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
)

SEED = 7


@dataclass(frozen=True)
class Splits:
    """Forward-in-time windows, all keyed on the purchase timestamp."""

    train_start: pd.Timestamp = pd.Timestamp("2017-01-01")
    valid_start: pd.Timestamp = pd.Timestamp("2018-01-01")
    test_start: pd.Timestamp = pd.Timestamp("2018-04-01")
    test_end: pd.Timestamp = pd.Timestamp("2018-09-01")
    # Last timestamp anywhere in the export. Orders that were never delivered
    # but whose promise date passed before this moment count as missed.
    snapshot: pd.Timestamp = pd.Timestamp("2018-10-17 17:30:18")


@dataclass(frozen=True)
class CostAssumptions:
    """Stated, adjustable assumptions behind the money figures (USD)."""

    late_order_cost: float = 14.0  # support contact, goodwill voucher, lost repeat margin
    action_cost: float = 1.5  # a few minutes of ops time per flagged order
    action_efficacy: float = 0.35  # share of flagged late orders the action rescues
    budget_share: float = 0.10  # fixed-list comparison: share of each day's orders touched


SPLITS = Splits()
COSTS = CostAssumptions()
