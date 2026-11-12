"""End-to-end run: data, both decision points, calibration, promise quotes,
review impact, audits, economics, charts and the scoring bundle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import audit, causal, economics, evaluate, explain, plots, promise
from .calibrate import apply_shift, intercept_shift, matured_window, monthly_recalibration
from .config import ASSET_DIR, COSTS, OUTPUT_DIR, RAW_DIR, SPLITS
from .data import build_orders, read_raw, split_frame
from .features import (
    CHECKOUT_CATEGORICAL,
    CHECKOUT_NUMERIC,
    HANDOFF_NUMERIC,
    add_checkout_features,
    add_handoff_features,
)
from .fetch import fetch
from .models import MODEL_ORDER, feature_frame, make_model, tune_lightgbm
from .score import reference_row

LGBM_GRID = [
    {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 400},
    {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 500},
    {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 300},
]
PROMISE_TARGET = 0.92
LABELS = {"promise_only": "promise + recent rate", "logistic": "logistic regression",
          "hist_gbm": "hist gradient boosting", "lightgbm": "LightGBM"}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def capture_curve(y, p, points: int = 81) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="stable")
    cum = np.concatenate([[0], np.cumsum(y[order])]) / max(y.sum(), 1)
    x = np.linspace(0, 1, points)
    return x, cum[(x * len(y)).astype(int)]


def compare_stage(name: str, train, valid, test, numeric, order, lgbm_params, log) -> dict:
    """Fit candidates on train, choose on validation, refit on train+valid, score test."""
    dev = pd.concat([train, valid])
    val_p, test_p, fitted = {}, {}, {}
    for m in order:
        params = lgbm_params if m == "lightgbm" else {}
        model = make_model(m, numeric=numeric, **params)
        model.fit(feature_frame(train, numeric), train["late"])
        val_p[m] = model.predict_proba(feature_frame(valid, numeric))[:, 1]
        final = make_model(m, numeric=numeric, **params)
        final.fit(feature_frame(dev, numeric), dev["late"])
        test_p[m] = final.predict_proba(feature_frame(test, numeric))[:, 1]
        fitted[m] = final
        log(f"  {name}/{m}: valid AP {evaluate.metrics(valid['late'], val_p[m])['ap']:.4f}")
    vb = evaluate.bootstrap(valid["late"], val_p, valid["purchased_at"], "ap")
    tb = evaluate.bootstrap(test["late"], test_p, test["purchased_at"], "ap")
    point = {m: evaluate.metrics(valid["late"], val_p[m])["ap"] for m in order}
    chosen, detail = evaluate.one_se_choice(point, vb, order)
    rows = {}
    for m in order:
        rows[m] = {"valid": evaluate.metrics(valid["late"], val_p[m]),
                   "test": evaluate.metrics(test["late"], test_p[m]),
                   "valid_ap_ci": evaluate.interval(vb[m]),
                   "test_ap_ci": evaluate.interval(tb[m])}
    return {"chosen": chosen, "selection": detail, "models": rows, "val_p": val_p,
            "test_p": test_p, "fitted": fitted, "test_boot": tb}


def run(raw_dir: Path = RAW_DIR, out_dir: Path = OUTPUT_DIR, asset_dir: Path = ASSET_DIR,
        quick: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    fetch(raw_dir)
    tables = read_raw(raw_dir)
    orders = build_orders(tables)
    full = add_handoff_features(add_checkout_features(orders))
    sp = split_frame(full)
    train, valid, test = sp["train"], sp["valid"], sp["test"]
    dev = pd.concat([train, valid])
    log(f"orders {len(full):,}; train {len(train):,} valid {len(valid):,} test {len(test):,}")

    summary: dict = {"data": {
        "labelled_orders": int(len(full)), "late_rate_all": float(full["late"].mean()),
        "never_arrived": int(full["never_arrived"].sum()),
        "splits": {k: {"n": int(len(v)), "late_rate": float(v["late"].mean()),
                       "start": str(v["purchased_at"].min().date()),
                       "end": str(v["purchased_at"].max().date())} for k, v in sp.items()},
        "median_promise_days": float(full["promise_days"].median()),
        "median_delivery_days": float(full["delivery_days"].median()),
        "median_slack_days": float((full["promise_days"] - full["delivery_days"]).median()),
    }}

    # Checkout decision point -------------------------------------------------
    grid = LGBM_GRID[:1] if quick else LGBM_GRID
    lgbm_params, tuning = tune_lightgbm(train, grid)
    summary["lightgbm_tuning"] = tuning
    log(f"lightgbm settings {lgbm_params}")
    order = MODEL_ORDER
    ck = compare_stage("checkout", train, valid, test, CHECKOUT_NUMERIC, order, lgbm_params, log)
    ck_name = ck["chosen"]
    ck_model = ck["fitted"][ck_name]
    tb = ck["test_boot"]
    gain = tb[ck_name] - tb["promise_only"]
    summary["checkout"] = {"chosen": ck_name, "selection": ck["selection"],
                           "models": ck["models"],
                           "test_ap_gain_vs_promise": float(
                               ck["models"][ck_name]["test"]["ap"]
                               - ck["models"]["promise_only"]["test"]["ap"]),
                           "test_ap_gain_ci": evaluate.interval(gain)}
    log(f"checkout choice: {ck_name}")

    # Handoff decision point ---------------------------------------------------
    has = lambda d: d.loc[d["carrier_at"].notna()]  # noqa: E731
    tr_h, va_h, te_h = has(train), has(valid), has(test)
    h_num = CHECKOUT_NUMERIC + HANDOFF_NUMERIC
    h_order = ["logistic", "hist_gbm", "lightgbm"]
    ho = compare_stage("handoff", tr_h, va_h, te_h, h_num, h_order, lgbm_params, log)
    ho_name = ho["chosen"]
    ho_model = ho["fitted"][ho_name]
    summary["handoff"] = {"chosen": ho_name, "selection": ho["selection"],
                          "models": ho["models"],
                          "orders_without_handoff_in_test": int(len(test) - len(te_h)),
                          "late_rate_without_handoff": float(
                              test.loc[test["carrier_at"].isna(), "late"].mean()),
                          "median_days_left_at_handoff_late": float(
                              te_h.loc[te_h["late"] == 1, "days_left_at_handoff"].median())}
    log(f"handoff choice: {ho_name}")

    # Same-subset comparison of both decision points on the test window.
    ck_on_h = ck_model.predict_proba(feature_frame(te_h))[:, 1]
    po_on_h = ck["fitted"]["promise_only"].predict_proba(feature_frame(te_h))[:, 1]
    ho_p = ho["test_p"][ho_name]
    rng = np.random.default_rng(0)
    rand_p = rng.random(len(te_h))
    y_h = te_h["late"].to_numpy()
    stage_cmp = {"random": evaluate.metrics(y_h, rand_p),
                 "promise_only": evaluate.metrics(y_h, po_on_h),
                 "checkout": evaluate.metrics(y_h, ck_on_h),
                 "handoff": evaluate.metrics(y_h, ho_p)}
    sb = evaluate.bootstrap(y_h, {"checkout": ck_on_h, "handoff": ho_p},
                            te_h["purchased_at"], "ap")
    summary["stage_comparison"] = {"metrics": stage_cmp, "handoff_minus_checkout_ap_ci":
                                   evaluate.interval(sb["handoff"] - sb["checkout"])}

    # Calibration under base-rate drift ----------------------------------------
    hist_p = ck_model.predict_proba(feature_frame(full))[:, 1]
    test_raw = ck["test_p"][ck_name]
    test_cal, monthly = monthly_recalibration(test, test_raw, full,
                                              pd.Series(hist_p, index=full.index))
    summary["calibration"] = {
        "before": evaluate.metrics(test["late"], test_raw),
        "after": evaluate.metrics(test["late"], test_cal),
        "monthly": monthly.to_dict(orient="records"),
    }
    hist_ph = ho_model.predict_proba(feature_frame(has(full), h_num))[:, 1]
    latest = matured_window(full, SPLITS.test_end)
    hp_series = pd.Series(hist_p, index=full.index)
    ck_shift = intercept_shift(hp_series.loc[latest.index].to_numpy(), latest["late"].mean())
    latest_h = matured_window(has(full), SPLITS.test_end)
    hh_series = pd.Series(hist_ph, index=has(full).index)
    ho_shift = intercept_shift(hh_series.loc[latest_h.index].to_numpy(), latest_h["late"].mean())
    log("calibration done")

    # Economics on the handoff-eligible test orders ----------------------------
    ho_cal, monthly_h = monthly_recalibration(te_h, ho_p, has(full), hh_series)
    summary["calibration"]["handoff_before"] = evaluate.metrics(y_h, ho_p)
    summary["calibration"]["handoff_after"] = evaluate.metrics(y_h, ho_cal)
    summary["calibration"]["handoff_monthly"] = monthly_h.to_dict(orient="records")
    policies = {
        "random 10% of each day": economics.daily_topk(te_h, rand_p, 0.10),
        "checkout model, top 10% of each day": economics.daily_topk(te_h, ck_on_h, 0.10),
        "handoff model, top 10% of each day": economics.daily_topk(te_h, ho_p, 0.10),
        "handoff model, top 2% of each day": economics.daily_topk(te_h, ho_p, 0.02),
        "handoff threshold, raw probabilities": economics.threshold_flags(ho_p),
        "handoff threshold, recalibrated monthly": economics.threshold_flags(ho_cal),
    }
    econ = {k: economics.value_per_thousand(y_h, f) for k, f in policies.items()}
    sens = economics.sensitivity(y_h, ho_cal)
    summary["economics"] = {
        "assumptions": {"late_order_cost_usd": COSTS.late_order_cost,
                        "action_cost_usd": COSTS.action_cost,
                        "action_efficacy": COSTS.action_efficacy,
                        "threshold": economics.threshold()},
        "per_policy": econ,
        "test_orders": int(len(te_h)),
        "sensitivity_threshold_policy": sens.to_dict(orient="records"),
    }
    log("economics done")

    # Promise quotes -----------------------------------------------------------
    deliv_tr = train.loc[train["delivered_at"].notna()]
    qm = promise.make_quantile_model(0.9).fit(feature_frame(deliv_tr),
                                              promise.realised_days(deliv_tr))
    pred_va = qm.predict(feature_frame(valid))
    pred_te = qm.predict(feature_frame(test))
    act_va, act_te = promise.realised_days(valid), promise.realised_days(test)
    olist_va = valid["promise_days"].to_numpy(float)
    olist_te = test["promise_days"].to_numpy(float)
    offset = promise.conformal_offset(pred_va, act_va, PROMISE_TARGET)
    cut = promise.constant_shift(olist_va, act_va, PROMISE_TARGET)
    q_te = promise.quote(pred_te, offset)
    s_te = np.maximum(olist_te - cut, 2)
    # Matched comparison: on the test window, how short can each rule go while
    # still matching Olist's own on-time rate there?
    olist_ot = promise.on_time_rate(olist_te, act_te)
    curve = promise.tradeoff_curve(olist_te, pred_te, act_te, np.arange(-6, 30, 0.5))
    matched = {}
    for rule in ("quantile model", "shifted Olist quote"):
        c = curve.loc[(curve["rule"] == rule) & (curve["on_time"] >= olist_ot)]
        matched[rule] = float(c["mean_days"].min()) if len(c) else None
    summary["promise"] = {
        "target_on_time": PROMISE_TARGET, "conformal_offset_days": offset,
        "olist_cut_days": cut,
        "test": {"olist": {"mean_days": float(olist_te.mean()), "on_time": olist_ot},
                 "shifted_olist": {"mean_days": float(s_te.mean()),
                                   "on_time": promise.on_time_rate(s_te, act_te)},
                 "quantile_model": {"mean_days": float(q_te.mean()),
                                    "on_time": promise.on_time_rate(q_te, act_te)}},
        "valid": {"olist_on_time": promise.on_time_rate(olist_va, act_va)},
        "matched_to_olist_on_time": matched,
    }
    log("promise quotes done")

    # Review impact ------------------------------------------------------------
    labelled = full.loc[(full["purchased_at"] >= SPLITS.train_start)
                        & (full["purchased_at"] < SPLITS.test_end)]
    frame = causal.design(labelled)
    n_est = 80 if quick else 300
    effects = [causal.effect(frame, "score", n_estimators=n_est),
               causal.effect(frame, "low", n_estimators=n_est)]
    summary["review_impact"] = effects
    log("review impact done")

    # Audits -------------------------------------------------------------------
    a1 = audit.outcome_leak(dev, test)
    a2 = audit.whole_period_history(dev, test, full)
    a3 = audit.random_vs_forward(dev, test)
    summary["audits"] = {"outcome_columns": a1, "whole_period_history": a2,
                         "random_vs_forward": a3}
    audit.feature_inventory().to_csv(out_dir / "feature_inventory.csv", index=False)
    log("audits done")

    # Explanations ---------------------------------------------------------------
    ale_ck = explain.ale_curves(ck_model, feature_frame(test),
                                ["expected_slack", "distance_km", "seller_late_rate",
                                 "zone_late_rate"], sample=2000 if quick else 6000)
    ale_ho = explain.ale_curves(ho_model, feature_frame(te_h, h_num),
                                ["days_left_at_handoff", "handling_days"],
                                sample=2000 if quick else 6000)
    ref_ck = reference_row(dev, CHECKOUT_NUMERIC, CHECKOUT_CATEGORICAL)
    ref_ho = reference_row(has(dev), h_num, CHECKOUT_CATEGORICAL)
    top_idx = np.argsort(-ho_p)[:3]
    examples = []
    for i in top_idx:
        row = feature_frame(te_h.iloc[[i]], h_num)
        why = explain.reasons(ho_model, row, ref_ho)
        examples.append({"risk": float(apply_shift(np.array([ho_p[i]]), ho_shift)[0]),
                         "late": int(y_h[i]), "reasons": [w["text"] for w in why]})
    summary["example_reasons"] = examples

    # Bundle -----------------------------------------------------------------------
    cut_ck = {str(q): float(np.quantile(test_raw, q)) for q in (0.75, 0.90)}
    cut_ho = {str(q): float(np.quantile(ho_p, q)) for q in (0.75, 0.90)}
    bundle = {"checkout_model": ck_model, "handoff_model": ho_model,
              "checkout_reference": ref_ck, "handoff_reference": ref_ho,
              "checkout_shift": ck_shift, "handoff_shift": ho_shift,
              "checkout_cutoffs": cut_ck, "handoff_cutoffs": cut_ho,
              "states": sorted(full["customer_state"].dropna().unique().tolist()),
              "categories": sorted(dev["main_category"].value_counts().head(30).index.tolist())}
    joblib.dump(bundle, out_dir / "bundle.joblib")
    summary["bundle"] = {"checkout_shift": ck_shift, "handoff_shift": ho_shift}

    # Charts ---------------------------------------------------------------------
    plots.monthly_late_rate(full.loc[full["purchased_at"] >= SPLITS.train_start],
                            asset_dir / "monthly_late_rate.png")
    table = pd.DataFrame([{
        "label": LABELS[m],
        "valid_ap": r["valid"]["ap"], "valid_ap_lo": r["valid_ap_ci"][0],
        "valid_ap_hi": r["valid_ap_ci"][1], "valid_base": r["valid"]["base_rate"],
        "test_ap": r["test"]["ap"], "test_ap_lo": r["test_ap_ci"][0],
        "test_ap_hi": r["test_ap_ci"][1], "test_base": r["test"]["base_rate"],
    } for m, r in ck["models"].items()])
    plots.model_comparison(table, asset_dir / "checkout_models.png")
    plots.capture_curves({
        "random": capture_curve(y_h, rand_p),
        "promise + recent rate": capture_curve(y_h, po_on_h),
        f"checkout ({LABELS[ck_name]})": capture_curve(y_h, ck_on_h),
        f"carrier handoff ({LABELS[ho_name]})": capture_curve(y_h, ho_p),
    }, asset_dir / "capture_curves.png")
    plots.calibration(evaluate.calibration_table(test["late"], test_raw),
                      evaluate.calibration_table(test["late"], test_cal),
                      asset_dir / "calibration.png")
    plots.ale_panel({**ale_ck, **ale_ho}, {
        "expected_slack": "promise minus lane's usual days",
        "distance_km": "distance (km)", "seller_late_rate": "seller's past late rate",
        "zone_late_rate": "zip area's past late rate",
        "days_left_at_handoff": "days left at handoff (handoff model)",
        "handling_days": "seller handling days (handoff model)"}, asset_dir / "ale.png")
    chosen_quotes = {"conformal quantile quote": (float(q_te.mean()),
                                                  promise.on_time_rate(q_te, act_te))}
    if cut > 0:
        chosen_quotes[f"Olist quote - {cut} days"] = (float(s_te.mean()),
                                                      promise.on_time_rate(s_te, act_te))
    plots.promise_tradeoff(curve, (float(olist_te.mean()), olist_ot), chosen_quotes,
                           asset_dir / "promise_tradeoff.png")
    plots.causal_bars(effects, asset_dir / "review_impact.png")
    plots.audit_bars([
        ("post-delivery columns", a1["honest"]["auc"], a1["with_outcome_columns"]["auc"]),
        ("whole-period seller rate", a2["as_of"]["auc"], a2["whole_period"]["auc"]),
        ("shuffled k-fold", a3["forward"]["auc"], a3["random_kfold"]["auc"]),
    ], asset_dir / "audits.png")
    plots.sensitivity_heatmap(sens, asset_dir / "economics_sensitivity.png")
    plots.policy_bars(econ, asset_dir / "policies.png")
    log("charts done")

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(summary), indent=2))
    log(f"wrote {out_dir / 'metrics.json'}")
    return summary
