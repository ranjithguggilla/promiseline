import numpy as np
import pandas as pd
import pytest

from promiseline.app import format_result
from promiseline.audit import feature_inventory, random_vs_forward
from promiseline.features import CHECKOUT_CATEGORICAL, CHECKOUT_NUMERIC, HANDOFF_NUMERIC
from promiseline.models import MODEL_ORDER, feature_frame, make_model
from promiseline.score import build_row, reference_row, score_order


@pytest.mark.parametrize("name", MODEL_ORDER)
def test_every_candidate_fits_and_handles_unseen_categories(featured, name):
    model = make_model(name, n_estimators=30, max_iter=30).fit(feature_frame(featured),
                                                               featured["late"])
    probe = feature_frame(featured.head(5)).copy()
    probe["customer_state"] = "ZZ"
    p = model.predict_proba(probe)[:, 1]
    assert np.all((p >= 0) & (p <= 1))


@pytest.fixture(scope="module")
def bundle(featured):
    num_h = CHECKOUT_NUMERIC + HANDOFF_NUMERIC
    ck = make_model("logistic").fit(feature_frame(featured), featured["late"])
    ho = make_model("lightgbm", numeric=num_h, n_estimators=40).fit(
        feature_frame(featured, num_h), featured["late"])
    return {"checkout_model": ck, "handoff_model": ho,
            "checkout_reference": reference_row(featured, CHECKOUT_NUMERIC, CHECKOUT_CATEGORICAL),
            "handoff_reference": reference_row(featured, num_h, CHECKOUT_CATEGORICAL),
            "checkout_shift": -0.5, "handoff_shift": 0.0,
            "checkout_cutoffs": {"0.75": 0.1, "0.9": 0.2},
            "handoff_cutoffs": {"0.75": 0.1, "0.9": 0.2}, "states": ["BA", "SP"]}


def test_build_row_recomputes_derived_columns(bundle):
    row = build_row({"promise_days": 10, "route_mean_days": 14, "route_recent_days": 12,
                     "customer_state": "BA", "seller_state": "BA", "price_total": 100.0,
                     "freight_total": 20.0}, bundle["checkout_reference"])
    assert row["expected_slack"].iloc[0] == -4
    assert row["recent_slack"].iloc[0] == -2
    assert row["same_state"].iloc[0] == 1
    assert abs(row["freight_ratio"].iloc[0] - 0.2) < 1e-12


def test_score_order_both_stages(bundle):
    order = {"customer_state": "BA", "seller_state": "SP", "promise_days": 8,
             "route_mean_days": 18, "route_recent_days": 18, "distance_km": 1900}
    out = score_order(bundle, order)
    assert 0 <= out["risk"] <= 1 and out["band"] in {"normal", "elevated", "high"}
    assert len(out["reasons"]) <= 3
    late = score_order(bundle, {**order, "handling_days": 12.0, "missed_ship_limit": 1}, "handoff")
    calm = score_order(bundle, {**order, "handling_days": 0.5, "missed_ship_limit": 0}, "handoff")
    assert late["risk"] >= calm["risk"]
    assert "risk" in format_result(late)
    with pytest.raises(ValueError):
        score_order(bundle, order, stage="delivery")


def test_gradio_app_builds(bundle):
    from promiseline.app import build

    demo = build(bundle)
    assert type(demo).__name__ == "Blocks"


def test_reason_text_for_late_handoff():
    from promiseline.explain import describe

    assert "after the promised date" in describe("days_left_at_handoff", -3.2)
    assert "shorter than this lane" in describe("expected_slack", -4.0)


def test_shift_lowers_checkout_risk(bundle):
    order = {"customer_state": "BA", "seller_state": "SP"}
    shifted = score_order(bundle, order)["risk"]
    raw = score_order({**bundle, "checkout_shift": 0.0}, order)["risk"]
    assert shifted < raw


def test_audit_helpers_run(featured):
    half = featured["purchased_at"].quantile(0.6)
    res = random_vs_forward(featured.loc[featured["purchased_at"] < half],
                            featured.loc[featured["purchased_at"] >= half], folds=3)
    assert 0 <= res["forward"]["auc"] <= 1
    inv = feature_inventory()
    assert set(inv["known_at"]) >= {"checkout", "carrier handoff"}
    assert isinstance(inv, pd.DataFrame)
