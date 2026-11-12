"""Gradio front end. All logic lives in ``score.score_order``."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import OUTPUT_DIR
from .score import score_order


def load_bundle(path: Path = OUTPUT_DIR / "bundle.joblib") -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `promiseline run` first")
    return joblib.load(path)


def format_result(result: dict) -> str:
    lines = [f"**Late-delivery risk: {result['risk']:.0%}** ({result['band']} for the "
             f"{result['stage']} stage)"]
    if result["reasons"]:
        lines.append("")
        lines.append("Main reasons, compared with a typical order:")
        lines.extend(f"- {r}" for r in result["reasons"])
    else:
        lines.append("\nNothing about this order stands out from a typical one.")
    return "\n".join(lines)


def build(bundle: dict):
    import gradio as gr

    def predict(stage, customer_state, seller_state, distance_km, promise_days, price, freight,
                weight_g, seller_late_rate, route_mean_days, handling_days, missed_limit):
        order = {
            "customer_state": customer_state, "seller_state": seller_state,
            "distance_km": distance_km, "promise_days": promise_days, "price_total": price,
            "freight_total": freight, "weight_g": weight_g,
            "seller_late_rate": seller_late_rate, "route_mean_days": route_mean_days,
            "route_recent_days": route_mean_days,
        }
        if stage == "handoff":
            order.update({"handling_days": handling_days, "missed_ship_limit": int(missed_limit)})
        return format_result(score_order(bundle, order, stage=stage))

    states = bundle["states"]
    with gr.Blocks(title="promiseline") as demo:
        gr.Markdown("## promiseline\nScore one marketplace order for the risk of missing its "
                    "promised delivery date.")
        with gr.Row():
            with gr.Column():
                stage = gr.Radio(["checkout", "handoff"], value="checkout", label="Decision point")
                cs = gr.Dropdown(states, value="RJ", label="Customer state")
                ss = gr.Dropdown(states, value="SP", label="Seller state")
                dist = gr.Slider(0, 3500, value=350, label="Distance seller to customer (km)")
                prom = gr.Slider(3, 60, value=24, step=1, label="Promised days")
                price = gr.Number(value=120.0, label="Order value")
                freight = gr.Number(value=18.0, label="Freight")
                weight = gr.Number(value=1200, label="Weight (g)")
            with gr.Column():
                slr = gr.Slider(0, 0.5, value=0.07, label="Seller's past late rate")
                rmd = gr.Slider(2, 40, value=12, label="Lane's usual delivery days")
                hd = gr.Slider(0, 30, value=2.5, label="Days the seller took to hand over")
                ml = gr.Checkbox(value=False, label="Seller missed its shipping deadline")
                out = gr.Markdown()
                gr.Button("Score order").click(
                    predict, [stage, cs, ss, dist, prom, price, freight, weight, slr, rmd, hd, ml],
                    out)
    return demo


def launch(port: int = 7860) -> None:
    build(load_bundle()).launch(server_port=port)
