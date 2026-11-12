"""Command line entry point: ``promiseline fetch | run | app``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="promiseline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the Olist tables from the pinned mirror")
    run = sub.add_parser("run", help="train, evaluate and write outputs/ and assets/")
    run.add_argument("--quick", action="store_true", help="smaller grid, for smoke runs")
    app = sub.add_parser("app", help="launch the Gradio scoring app")
    app.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        from .fetch import fetch
        print(f"tables in {fetch()}")
    elif args.cmd == "run":
        from .pipeline import run as run_pipeline
        run_pipeline(quick=args.quick)
    else:
        from .app import launch
        launch(port=args.port)


if __name__ == "__main__":
    main()
